#!/usr/bin/env python
"""Pixel <-> robot-XY homography calibration via click-then-touch.

The "robot point" we record is the URDF endpoint -- the origin of
gripper_frame_link in the robot base frame -- with NO TOOL_TIP_OFFSET
arithmetic. Downstream IK/test scripts should also target
gripper_frame_link directly so the calibration and the runtime use the
same reference point.

Workflow:

  1. Live camera GUI. Move the robot to a good scan pose.
     Press ENTER to lock the joint angles (motors hold position).
  2. Click >= 4 points on the checkerboard in the camera view.
     Press 'r' to undo the last click. Press ENTER when done.
  3. Torque is released. For each clicked point, manually bring
     gripper_frame_link onto that point and press ENTER. The frame's
     (x, y, z) in the robot base is read from forward kinematics.
  4. A 3x3 homography H mapping pixel -> robot XY is fit
     (cv2.findHomography) and saved alongside the locked joint angles.

Output:
    configs/click_homography.json
        {
          "joint_angles_deg": {motor: deg, ...},
          "image_points_px":  [[u, v], ...],
          "robot_points_xy":  [[x, y], ...],
          "H_pixel_to_robot": [[...3x3...]],
          "rmse_robot_m":     float,
          "rmse_robot_mm":    float
        }

USAGE:
    python scripts/click_homography_calibrate.py
    python scripts/click_homography_calibrate.py --camera 1 --out configs/my.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from cybertyping.core.primitives import (
    BODY_MOTOR_ORDER,
    URDF_PATH,
    build_chain,
    fk_tip,
)

# Use the URDF endpoint (gripper_frame_link) directly -- no TOOL_TIP_OFFSET arithmetic.
# Pass a zero offset to fk_tip so it returns T_base->gripper_frame_link[:3, 3].
_ZERO_TIP_OFFSET = np.zeros(3, dtype=float)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


WINDOW = "homography calibration"
MAX_DISPLAY_W = 1280

# Same camera-open logic as scripts/camera_check.py.
_BACKENDS: dict[str, int | None] = {
    "auto":  None,
    "msmf":  cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "v4l2":  cv2.CAP_V4L2,
    "any":   cv2.CAP_ANY,
}


def _backend_name(b: int) -> str:
    return {
        cv2.CAP_MSMF: "MSMF",
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_V4L2: "V4L2",
        cv2.CAP_AVFOUNDATION: "AVFOUNDATION",
        cv2.CAP_ANY: "ANY",
    }.get(b, str(b))


def _auto_backend_chain() -> list[int]:
    """Backends to try, in preference order, when --backend auto."""
    sysname = platform.system()
    if sysname == "Windows":
        return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    if sysname == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def _try_open(index: int, backend: int, width: int, height: int,
              warmup_frames: int = 20) -> cv2.VideoCapture | None:
    """Mirror scripts/live_dots.py::_open_and_verify. A backend is accepted
    only if it returns frames with max_pixel > 0 (i.e. not all-zero black)."""
    print(f"[cam] try index={index} backend={_backend_name(backend)} "
          f"{width}x{height}")
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"[cam]   isOpened=False")
        try:
            cap.release()
        except Exception:
            pass
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    rep_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rep_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[cam]   reports {rep_w}x{rep_h}")
    last_ok = None
    for _ in range(warmup_frames):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            last_ok = frame
        time.sleep(0.03)
    if last_ok is None:
        print(f"[cam]   no frames read in {warmup_frames} attempts -- releasing")
        try:
            cap.release()
        except Exception:
            pass
        return None
    max_pix = int(last_ok.max())
    avg = float(last_ok.mean())
    if max_pix == 0:
        print(f"[cam]   all-zero frames (backend not really streaming) -- releasing")
        try:
            cap.release()
        except Exception:
            pass
        return None
    print(f"[cam]   OK frame {last_ok.shape[1]}x{last_ok.shape[0]} "
          f"mean={avg:.1f} max={max_pix}")
    return cap


class SimpleCamera:
    """Camera wrapper that mirrors scripts/camera_check.py but with:
      - retry-on-read-failure (no crash on a single bad frame)
      - optional backend fallback when --backend auto
    """

    def __init__(self, index: int, backend: int | None, width: int, height: int):
        self.index = index
        self.backend_pref = backend  # None => try the auto chain
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None
        self._used_backend: int | None = None

    def __enter__(self):
        candidates = ([self.backend_pref] if self.backend_pref is not None
                      else _auto_backend_chain())
        for b in candidates:
            cap = _try_open(self.index, b, self.width, self.height)
            if cap is not None:
                self._cap = cap
                self._used_backend = b
                print(f"[cam] using backend={_backend_name(b)}")
                return self
        raise RuntimeError(
            f"Could not get any frames from camera index {self.index} "
            f"with backends {[_backend_name(b) for b in candidates]}. "
            f"Run scripts/camera_check.py to debug."
        )

    def __exit__(self, *exc):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def grab(self, flush: int = 0, max_attempts: int = 30) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError("Camera not opened.")
        for _ in range(max(0, flush)):
            self._cap.read()
        for _ in range(max_attempts):
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                return frame
            time.sleep(0.01)
        raise RuntimeError(f"Camera read failed after {max_attempts} attempts")


def _read_joints_deg(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.array(
        [float(obs[f"{n}.pos"]) for n in BODY_MOTOR_ORDER], dtype=float
    )


def _hold_pose(robot, q_deg: np.ndarray, gripper: float) -> None:
    action = {f"{n}.pos": float(q_deg[i]) for i, n in enumerate(BODY_MOTOR_ORDER)}
    action["gripper.pos"] = float(gripper)
    robot.send_action(action)


def _draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    h = 22 * len(lines) + 12
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


def _resize_for_display(frame: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY_W / float(w))
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale))), scale
    return frame.copy(), 1.0


def phase_lock_pose(cam: SimpleCamera, robot, chain) -> bool:
    """Disable torque so the user can hand-position the arm.
    ENTER re-enables torque and locks the pose. Returns True on confirm,
    False on abort. On return, torque is enabled and the arm is holding
    its current joint angles."""
    tip_off = _ZERO_TIP_OFFSET

    print("[robot] Disabling torque -- hand-position the robot.")
    try:
        robot.bus.disable_torque()
    except Exception as e:
        print(f"[robot] disable_torque failed: {e!r}")

    confirmed = False
    try:
        while True:
            frame = cam.grab(flush=0)
            disp, _ = _resize_for_display(frame)
            q_now = _read_joints_deg(robot)
            tip_now = fk_tip(chain, q_now, tip_off)
            disp = _draw_hud(disp, [
                "Phase 1/3  TORQUE OFF -- move the robot by hand.",
                f"tip XYZ = ({tip_now[0]:+.4f}, {tip_now[1]:+.4f}, "
                f"{tip_now[2]:+.4f}) m",
                "ENTER = lock pose here   q/ESC = abort",
            ])
            cv2.imshow(WINDOW, disp)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                return False
            if key in (10, 13):
                confirmed = True
                return True
    finally:
        # Re-enable torque so we hold whatever pose the arm is in. The caller
        # will read joints + send_action right after this returns to anchor it.
        try:
            robot.bus.enable_torque()
            q_now = _read_joints_deg(robot)
            grip = float(robot.get_observation().get("gripper.pos", 100.0))
            _hold_pose(robot, q_now, grip)
        except Exception as e:
            print(f"[robot] re-enable failed: {e!r}")
        if confirmed:
            print("[robot] Torque re-enabled; pose locked.")


def phase_pick_pixels(cam: SimpleCamera) -> list[tuple[float, float]] | None:
    """Click points on the camera view. ENTER finishes. Returns original-image coords."""
    clicks_disp: list[tuple[int, int]] = []
    state = {"new": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["new"] = (x, y)

    cv2.setMouseCallback(WINDOW, on_mouse)
    base_frame = cam.grab(flush=3)   # freeze a single frame to click on

    while True:
        disp, scale = _resize_for_display(base_frame)
        if state["new"] is not None:
            clicks_disp.append(state["new"])
            state["new"] = None
        for i, (cx, cy) in enumerate(clicks_disp):
            cv2.drawMarker(disp, (cx, cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 22, 2)
            cv2.circle(disp, (cx, cy), 12, (0, 255, 255), 2)
            cv2.putText(disp, str(i + 1), (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        disp = _draw_hud(disp, [
            f"Phase 2/3  Click >= 4 points on the checkerboard.  ({len(clicks_disp)} so far)",
            "LMB=add  r=undo  c=clear  ENTER=done (>=4)  q/ESC=abort",
        ])
        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.setMouseCallback(WINDOW, lambda *a: None)
            return None
        if key == ord("r") and clicks_disp:
            clicks_disp.pop()
        if key == ord("c"):
            clicks_disp.clear()
        if key in (10, 13):
            if len(clicks_disp) >= 4:
                break
            print(f"[clicks] need >= 4, have {len(clicks_disp)}")

    cv2.setMouseCallback(WINDOW, lambda *a: None)
    # Map display-coords back to original-image coords.
    return [(float(x / scale), float(y / scale)) for (x, y) in clicks_disp]


def phase_touch_points(
    robot,
    chain,
    cam: SimpleCamera,
    image_points: list[tuple[float, float]],
    locked_q: np.ndarray,
    gripper: float,
) -> list[tuple[float, float, float]] | None:
    """Disable torque. For each point, wait for ENTER then read tip XYZ. Returns
    list of (robot_x, robot_y, robot_z), or None on abort."""
    tip_off = _ZERO_TIP_OFFSET

    print("[robot] Disabling torque -- move the tip onto each highlighted point.")
    robot.bus.disable_torque()

    robot_xyz: list[tuple[float, float, float]] = []
    try:
        for idx, (ix, iy) in enumerate(image_points):
            while True:
                frame = cam.grab(flush=0)
                disp, scale = _resize_for_display(frame)
                # Draw all clicked points (past=green, current=red, future=yellow).
                for j, (jx, jy) in enumerate(image_points):
                    cx, cy = int(round(jx * scale)), int(round(jy * scale))
                    if j < idx:
                        color = (0, 200, 0)
                    elif j == idx:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 255)
                    cv2.drawMarker(disp, (cx, cy), color,
                                   cv2.MARKER_CROSS, 28, 2)
                    cv2.circle(disp, (cx, cy), 16, color, 2)
                    cv2.putText(disp, str(j + 1), (cx + 12, cy - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Live tip XYZ readout.
                q_now = _read_joints_deg(robot)
                tip_now = fk_tip(chain, q_now, tip_off)
                disp = _draw_hud(disp, [
                    f"Phase 3/3  Move tip onto point {idx + 1}/{len(image_points)} "
                    f"(red).",
                    f"tip XYZ = ({tip_now[0]:+.4f}, {tip_now[1]:+.4f}, "
                    f"{tip_now[2]:+.4f}) m",
                    "ENTER = record this point   s = skip   q/ESC = abort",
                ])
                cv2.imshow(WINDOW, disp)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    return None
                if key == ord("s"):
                    print(f"[touch] skipping point {idx + 1}")
                    break
                if key in (10, 13):
                    q_at = _read_joints_deg(robot)
                    tip_at = fk_tip(chain, q_at, tip_off)
                    rx, ry, rz = float(tip_at[0]), float(tip_at[1]), float(tip_at[2])
                    robot_xyz.append((rx, ry, rz))
                    print(f"[touch] point {idx + 1}: pixel=({ix:.1f},{iy:.1f}) "
                          f"-> robot=({rx:+.4f},{ry:+.4f},{rz:+.4f}) m")
                    break
    finally:
        # Re-enable torque and try to hold whatever pose the arm is currently
        # in so it doesn't fall when we return.
        print("[robot] Re-enabling torque (will hold current pose).")
        try:
            robot.bus.enable_torque()
            q_now = _read_joints_deg(robot)
            _hold_pose(robot, q_now, gripper)
        except Exception as e:
            print(f"[robot] re-enable failed: {e!r}")

    return robot_xyz


def fit_homography_pixel_to_robot(
    image_points: list[tuple[float, float]],
    robot_points: list[tuple[float, float]],
) -> tuple[np.ndarray, float]:
    src = np.array(image_points, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.array(robot_points, dtype=np.float64).reshape(-1, 1, 2)
    if len(image_points) == 4:
        H = cv2.getPerspectiveTransform(
            src.reshape(-1, 2).astype(np.float32),
            dst.reshape(-1, 2).astype(np.float32),
        ).astype(np.float64)
    else:
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            raise RuntimeError("findHomography returned None")
    pred = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    obs = dst.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((pred - obs) ** 2, axis=1))))
    return H, rmse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="UVC index")
    ap.add_argument("--backend", choices=list(_BACKENDS), default="any",
                    help="OpenCV capture backend (any worked for live_dots.py)")
    ap.add_argument("--width", type=int, default=640,
                    help="640x480 is the wrist cam's native mode; "
                         "MSMF/DSHOW pad or fail above that")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default=str(REPO_ROOT / "configs" / "click_homography.json"))
    args = ap.parse_args()
    # None => SimpleCamera tries the platform's auto chain w/ fallback.
    backend = None if args.backend == "auto" else _BACKENDS[args.backend]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chain = build_chain(URDF_PATH, locked_joints=[])  # all joints needed for FK

    port = os.environ.get("SO101_PORT", "COM7")
    robot_id = os.environ.get("SO101_ROBOT_ID", "team08_follower_arm")
    cfg = SO101FollowerConfig(port=port, id=robot_id, max_relative_target=None)
    robot = SO101Follower(cfg)
    print(f"[connect] {port} id={robot_id}")
    robot.connect()
    print("[connect] OK")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    try:
        with SimpleCamera(index=args.camera, backend=backend,
                          width=args.width, height=args.height) as cam:
            # ---------- PHASE 1: lock pose ----------
            if not phase_lock_pose(cam, robot, chain):
                print("[abort] user cancelled before locking pose")
                return 1
            locked_q = _read_joints_deg(robot)
            gripper = float(robot.get_observation().get("gripper.pos", 100.0))
            _hold_pose(robot, locked_q, gripper)
            print(f"[pose] locked q_deg = {[round(q, 2) for q in locked_q.tolist()]}")

            # ---------- PHASE 2: click points ----------
            image_points = phase_pick_pixels(cam)
            if image_points is None:
                print("[abort] user cancelled during click phase")
                return 1
            print(f"[clicks] {len(image_points)} points captured")

            # ---------- PHASE 3: touch points ----------
            robot_points = phase_touch_points(
                robot, chain, cam, image_points, locked_q, gripper,
            )
            if robot_points is None:
                print("[abort] user cancelled during touch phase")
                return 1
            if len(robot_points) != len(image_points):
                # User skipped some points -- trim image_points to match.
                image_points = image_points[:len(robot_points)]
            if len(robot_points) < 4:
                print(f"[abort] need >= 4 touched points, got {len(robot_points)}")
                return 1

        # ---------- FIT HOMOGRAPHY ----------
        robot_xy_only = [(p[0], p[1]) for p in robot_points]
        robot_z_only = [p[2] for p in robot_points]
        plane_z = float(np.median(robot_z_only))
        H, rmse = fit_homography_pixel_to_robot(image_points, robot_xy_only)
        print(f"[fit] homography pixel->robot RMSE = {rmse * 1000:.2f} mm")
        print(f"[fit] plane_z (median of touched z) = {plane_z:+.4f} m  "
              f"(spread {(max(robot_z_only) - min(robot_z_only)) * 1000:.1f} mm)")

        out = {
            "joint_angles_deg": {n: float(locked_q[i])
                                 for i, n in enumerate(BODY_MOTOR_ORDER)},
            "gripper_pos": float(gripper),
            "frame": "gripper_frame_link",  # URDF endpoint, no tool offset applied
            "image_points_px": [list(p) for p in image_points],
            "robot_points_xy": [list(p) for p in robot_xy_only],
            "robot_points_xyz": [list(p) for p in robot_points],
            "plane_z": plane_z,
            "H_pixel_to_robot": H.tolist(),
            "rmse_robot_m": rmse,
            "rmse_robot_mm": rmse * 1000.0,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[out] -> {out_path}")

    finally:
        cv2.destroyAllWindows()
        try:
            robot.bus.enable_torque()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        print("[robot] disconnected")

    return 0


if __name__ == "__main__":
    sys.exit(main())
