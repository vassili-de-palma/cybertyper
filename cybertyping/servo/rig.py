"""The physical rig: robot, kinematic chain, camera and the saved scan pose.

Every visual-servo script needs the same handful of objects. :class:`Rig`
owns them and their lifecycle::

    with Rig(camera_index=1) as rig:
        rig.go_to_scan_pose()
        ...

The scan pose comes from ``configs/click_homography.json``, written by
:mod:`cybertyping.calibration.click_homography`. Only three fields are used
by the servo loop: the joint angles of the overhead pose, the gripper
opening, and ``plane_z`` (table height in the robot frame). The homography
in that file is not used for servoing; it only serves the click-to-touch
diagnostic.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cybertyping.core import primitives as prim
from cybertyping.servo import camera as cam

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAL_PATH = REPO_ROOT / "configs" / "click_homography.json"

# Tool offset in the gripper frame. The click-homography calibration records
# gripper_frame_link itself, so the servo targets the same URDF endpoint.
ZERO_TIP_OFFSET = np.zeros(3, dtype=float)


@dataclass
class ScanPose:
    q_deg: np.ndarray        # body joints at the overhead scan pose
    gripper_pos: float       # gripper opening to hold during the task
    plane_z: float           # table height (m, robot frame)
    source: Path


def load_scan_pose(path: str | Path = DEFAULT_CAL_PATH) -> ScanPose:
    path = Path(path)
    with open(path) as f:
        cal = json.load(f)
    q = np.array([float(cal["joint_angles_deg"][n]) for n in prim.BODY_MOTOR_ORDER],
                 dtype=float)
    return ScanPose(
        q_deg=q,
        gripper_pos=float(cal.get("gripper_pos", 100.0)),
        plane_z=float(cal.get("plane_z", -0.014)),
        source=path,
    )


class Rig:
    """Robot + chain + camera + scan pose, with safe connect/disconnect.

    Parameters
    ----------
    cal_path:      click_homography.json to read the scan pose from.
    camera_index:  OpenCV index of the wrist camera (default: CAMERA_INDEX
                   env var, else 1).
    port, robot_id: serial port / lerobot id (default: SO101_PORT and
                   SO101_ROBOT_ID env vars).
    window:        name of the OpenCV window used for the live overlay.
    """

    def __init__(self, cal_path: str | Path = DEFAULT_CAL_PATH,
                 camera_index: int | None = None,
                 port: str | None = None, robot_id: str | None = None,
                 window: str = "cybertyping"):
        self.scan = load_scan_pose(cal_path)
        self.camera_index = (camera_index if camera_index is not None
                             else cam.default_camera_index())
        self.port = port or os.environ.get("SO101_PORT", "COM5")
        self.robot_id = robot_id or os.environ.get("SO101_ROBOT_ID", "team08_follower_arm")
        self.window = window
        self.tip_offset = ZERO_TIP_OFFSET
        self.chain = prim.build_chain(prim.URDF_PATH, locked_joints=prim.LOCKED_JOINTS)
        # Gripper +Z direction at the scan pose. The servo keeps the camera
        # at exactly this orientation so the image-to-robot gain stays valid.
        self.camera_dir = prim.fk_T(self.chain, self.scan.q_deg)[:3, 2].copy()
        self.robot = None
        self.cap: cv2.VideoCapture | None = None

        print(f"[rig] scan pose q = {np.round(self.scan.q_deg, 2).tolist()} "
              f"(from {self.scan.source.name})")
        print(f"[rig] plane_z = {self.scan.plane_z:+.4f} m, "
              f"camera dir = {np.round(self.camera_dir, 3).tolist()}")

    # ------------------------------------------------------------------ lifecycle

    def connect(self) -> "Rig":
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        cfg = SO101FollowerConfig(port=self.port, id=self.robot_id,
                                  max_relative_target=None)
        self.robot = SO101Follower(cfg)
        print(f"[rig] connecting {self.port} id={self.robot_id}")
        self.robot.connect()
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        return self

    def open_camera(self) -> None:
        self.cap = cam.open_camera(self.camera_index)
        if self.cap is None:
            raise RuntimeError(
                f"could not open camera index {self.camera_index}; "
                f"run scripts/probe_cameras.py to find the wrist camera")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        if self.robot is not None:
            try:
                self.robot.bus.enable_torque()
            except Exception:
                pass
            try:
                self.robot.disconnect()
            except Exception:
                pass
            self.robot = None
            print("[rig] disconnected")

    def __enter__(self) -> "Rig":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ helpers

    @property
    def gripper_pos(self) -> float:
        return self.scan.gripper_pos

    @property
    def plane_z(self) -> float:
        return self.scan.plane_z

    def read_joints_deg(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([float(obs[f"{n}.pos"]) for n in prim.BODY_MOTOR_ORDER],
                        dtype=float)

    def send_joints(self, q_deg: np.ndarray) -> None:
        action = {f"{n}.pos": float(q_deg[i]) for i, n in enumerate(prim.BODY_MOTOR_ORDER)}
        action["gripper.pos"] = self.gripper_pos
        self.robot.send_action(action)

    def slew_to(self, q_target: np.ndarray,
                max_step_deg: float = prim.MAX_STEP_DEG_PER_TICK_APPROACH,
                settle_s: float = 0.3) -> np.ndarray:
        """Joint-space slew from the current pose to ``q_target``."""
        q_now = self.read_joints_deg()
        ticks = prim.slew_segment(q_now, np.asarray(q_target, dtype=float), max_step_deg)
        prim.stream(self.robot, ticks, self.gripper_pos)
        time.sleep(settle_s)
        return self.read_joints_deg()

    def go_to_scan_pose(self) -> np.ndarray:
        print("[rig] slewing to scan pose")
        return self.slew_to(self.scan.q_deg)

    def read_frame(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        return cam.read_frame(self.cap)

    def show(self, frame: np.ndarray, wait_ms: int = 1) -> int:
        """Display ``frame`` and return the pressed key code (or -1)."""
        cv2.imshow(self.window, frame)
        return cv2.waitKey(wait_ms) & 0xFF
