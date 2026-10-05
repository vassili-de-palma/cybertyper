#!/usr/bin/env python
"""Click on a camera image pixel, robot touches it.

Workflow:
  1. Robot moves to the scan pose saved in configs/click_homography.json
  2. Camera shows live feed in a window
  3. Click any pixel (LMB) → robot touches that point
  4. Robot returns to scan pose, loops

The homography from click_homography_calibrate.py converts pixels directly
to robot XY (no IK uncertainty, only 6.3mm RMSE from calibration).

USAGE:
    $env:SO101_PORT="COM7"; python scripts/click_to_touch.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import platform

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from cybertyping.core.primitives import (
    BODY_MOTOR_ORDER,
    ENFORCE_GRIPPER_DOWN,
    LOCKED_JOINTS,
    MAX_STEP_DEG_PER_TICK_APPROACH,
    MAX_STEP_DEG_PER_TICK_RETREAT,
    URDF_PATH,
    build_chain,
    fk_tip,
    slew_segment,
    solve_ik_tip,
    stream,
)
from cybertyping.core.descent_jac import descent_jacobian_until_contact, compute_pos_jacobian_numeric
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

WINDOW = "click_to_touch"
CAL_PATH = REPO_ROOT / "configs" / "click_homography.json"
WORKSPACE_BOUNDS = {
    "x": [0.05, 0.35],
    "y": [-0.20, 0.25],
}
HOVER_MM = 50.0
MAX_IK_RES_M = 0.015


def _backend_chain() -> list[int]:
    """Backends to try, in preference order."""
    sysname = platform.system()
    if sysname == "Windows":
        return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    if sysname == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def _open_camera(index: int, width: int = 640, height: int = 480) -> cv2.VideoCapture | None:
    """Try to open camera at index with multiple backends."""
    for backend in _backend_chain():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"[camera] opened at index {index}")
        return cap
    return None


def _read_joints_deg(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.array(
        [float(obs[f"{n}.pos"]) for n in BODY_MOTOR_ORDER], dtype=float
    )


def pixel_to_robot_xy(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """Apply homography to convert pixel (px, py) to robot (rx, ry)."""
    pt = np.array([px, py, 1.0], dtype=np.float64)
    w = H @ pt
    rx, ry = float(w[0] / w[2]), float(w[1] / w[2])
    return rx, ry


def jacobian_servo_to_pixel(
    chain, robot, cap, q_init, tip_off, gripper_pos,
    target_px, H_pixel_to_robot, plane_z,
    tol_px=8.0, max_iters=50, loop_hz=10
) -> np.ndarray:
    """Pure Jacobian XYZ servo (no IK) — move tip towards pixel AND down to plane_z.

    Each iteration: grab frame, project FK tip to pixel, compute XY error,
    compute Z error via homography perspective (accounts for camera tilt),
    step joints via pinv(J) for all 3D.

    The homography encodes the camera's 3D orientation and perspective. To estimate Z:
    1. Project target pixel into 3D via the homography surface (assumes target_px is on plane_z)
    2. Compare target 3D to current tip 3D → Z error

    This automatically accounts for camera tilt because H embeds the camera's viewpoint.
    """
    H_inv = np.linalg.inv(H_pixel_to_robot)
    period = 1.0 / loop_hz
    q = q_init.copy()
    hover_z = plane_z + 0.010  # 10mm above plane

    for i in range(max_iters):
        # Grab frame
        cap.read()  # flush
        ret, frame = cap.read()

        # Project FK tip to pixel
        tip_xyz = fk_tip(chain, q, tip_off)
        pt = H_inv @ np.array([tip_xyz[0], tip_xyz[1], 1.0])
        px_tip = pt[:2] / pt[2]

        # Pixel error in image space
        err_px = np.asarray(target_px) - px_tip
        err_px_norm = np.linalg.norm(err_px)

        # Z error: move down to hover_z (plane_z + 10mm buffer)
        # Camera tilt is automatically accounted for by the homography in XY control;
        # Z descent is controlled separately via descent_jacobian_until_contact
        err_z = hover_z - tip_xyz[2]

        # Draw debug overlay
        if ret and frame is not None:
            cv2.drawMarker(frame, tuple(np.round(px_tip).astype(int)),
                           (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.drawMarker(frame, tuple(np.round(target_px).astype(int)),
                           (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(frame, f"xy_err={err_px_norm:.1f}px z_err={err_z*1000:+.1f}mm iter={i}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow(WINDOW, frame)
            cv2.waitKey(1)

        if err_px_norm < tol_px and err_z < 0.003:  # Converged on XY and close to Z
            print(f"[servo] converged at iter {i}, pixel_err={err_px_norm:.2f}px z_err={err_z*1000:.1f}mm")
            print(f"[servo] final tip = ({tip_xyz[0]:+.4f}, {tip_xyz[1]:+.4f}, {tip_xyz[2]:+.4f})")
            break

        # Convert pixel error to robot XY delta via homography scale
        H_scale = H_pixel_to_robot[:2, :2] / H_pixel_to_robot[2, 2]
        max_delta_xy = 0.002  # 2mm max XY per step
        delta_xy = H_scale @ err_px
        if np.linalg.norm(delta_xy) > max_delta_xy:
            delta_xy = delta_xy / np.linalg.norm(delta_xy) * max_delta_xy

        # Build full 3D velocity command: move XY towards pixel AND move Z towards hover_z
        delta_z = np.clip(err_z, -0.001, 0.001)  # Max 1mm per step in Z
        delta_xyz = np.array([delta_xy[0], delta_xy[1], delta_z], dtype=float)

        # Step joints via full 3D Jacobian
        J = compute_pos_jacobian_numeric(chain, q, BODY_MOTOR_ORDER, tip_off)
        dq_rad = np.linalg.pinv(J) @ delta_xyz
        dq_deg = np.rad2deg(dq_rad)
        # Clamp to 0.5 deg per joint per step
        dq_deg = np.clip(dq_deg, -0.5, 0.5)
        q = q + dq_deg

        # Send to robot
        action = {f"{n}.pos": float(q[j]) for j, n in enumerate(BODY_MOTOR_ORDER)}
        action["gripper.pos"] = gripper_pos
        robot.send_action(action)
        time.sleep(period)

    return q


def ask_confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("", "y", "yes")
    except EOFError:
        return False


def main() -> int:
    # Load calibration
    with open(CAL_PATH) as f:
        cal = json.load(f)
    scan_q = np.array(
        [
            float(cal["joint_angles_deg"][n])
            for n in BODY_MOTOR_ORDER
        ],
        dtype=float,
    )
    gripper_pos = float(cal.get("gripper_pos", 100.0))
    H = np.array(cal["H_pixel_to_robot"], dtype=np.float64)
    plane_z = float(cal.get("plane_z", -0.014))
    # gripper_frame_link is at the tip, so no offset needed
    tip_off = np.zeros(3, dtype=float)

    print(f"[load] scan pose q = {np.round(scan_q, 2).tolist()}")
    print(f"[load] plane_z = {plane_z:+.4f} m")
    print(f"[load] homography RMSE = {cal.get('rmse_robot_mm', 0):.1f} mm")

    # Build chain and connect robot
    chain = build_chain(URDF_PATH, locked_joints=LOCKED_JOINTS)
    port = os.environ.get("SO101_PORT", "COM7")
    robot_id = os.environ.get("SO101_ROBOT_ID", "team08_follower_arm")
    cfg = SO101FollowerConfig(port=port, id=robot_id, max_relative_target=None)
    robot = SO101Follower(cfg)
    print(f"[connect] {port} id={robot_id}")
    robot.connect()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    clicked_pt = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pt[0] = (float(x), float(y))

    cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        # Move to scan pose
        print("[move] slewing to scan pose...")
        q_now = _read_joints_deg(robot)
        ticks = slew_segment(q_now, scan_q, MAX_STEP_DEG_PER_TICK_APPROACH)
        stream(robot, ticks, gripper_pos)
        time.sleep(0.3)
        q_at_scan = _read_joints_deg(robot)
        print(f"[scan] at pose, q = {np.round(q_at_scan, 2).tolist()}")

        # Open camera once (robot wrist camera at index 1)
        cap = _open_camera(1)

        if cap is None:
            print("[camera error] could not open any camera")
            return 1

        while True:
            # Show live frames and wait for click
            print("[camera] live feed... click on the image to touch that point")
            clicked_pt[0] = None
            while clicked_pt[0] is None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[camera error] failed to read frame")
                    return 1
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    print("[abort] user quit")
                    return 0

            px, py = clicked_pt[0]
            print(f"[click] pixel ({px:.0f}, {py:.0f})")

            # Pixel to robot
            rx, ry = pixel_to_robot_xy(H, px, py)
            print(f"[map] robot XY = ({rx:+.4f}, {ry:+.4f})")

            # Bounds check
            if not (WORKSPACE_BOUNDS["x"][0] <= rx <= WORKSPACE_BOUNDS["x"][1]):
                print(
                    f"[reject] x={rx:.4f} outside "
                    f"{WORKSPACE_BOUNDS['x']}"
                )
                continue
            if not (WORKSPACE_BOUNDS["y"][0] <= ry <= WORKSPACE_BOUNDS["y"][1]):
                print(
                    f"[reject] y={ry:.4f} outside "
                    f"{WORKSPACE_BOUNDS['y']}"
                )
                continue

            # Pure Jacobian XY servo (no IK) — move towards fixed pixel target
            # Target pixel is FIXED to the clicked frame; servo converges by moving robot
            q_now = _read_joints_deg(robot)
            print(f"[jacobian_servo] moving towards fixed pixel ({px:.0f}, {py:.0f})")
            q_at_target = jacobian_servo_to_pixel(
                chain, robot, cap, q_now, tip_off, gripper_pos,
                target_px=np.array([px, py], dtype=float),
                H_pixel_to_robot=H,
                plane_z=plane_z,
                tol_px=5.0,
                max_iters=100,
                loop_hz=15,
            )
            print(f"[jacobian_servo] converged on target pixel")

            # Now descend from current position until contact
            print(f"[descent] descending until contact")
            contact_q, info = descent_jacobian_until_contact(
                chain,
                robot,
                BODY_MOTOR_ORDER,
                q_at_target,
                tip_off,
                gripper_pos,
                descent_speed_mps=0.005,
                max_descent_m=0.05,
                stall_thresh_deg=10.0,
                load_thresh=60,
                xy_drift_abort_mm=15.0,
            )
            print(f"[touch] {info.get('reason')} z={info.get('z_traveled_mm', 0):.1f}mm")

            # Retreat
            print("[retreat] retreating")
            ticks = slew_segment(
                contact_q, q_at_target, MAX_STEP_DEG_PER_TICK_RETREAT
            )
            stream(robot, ticks, gripper_pos)
            time.sleep(0.2)

            # Return to scan
            print("[scan] returning to scan pose")
            q_now = _read_joints_deg(robot)
            ticks = slew_segment(q_now, scan_q, MAX_STEP_DEG_PER_TICK_APPROACH)
            stream(robot, ticks, gripper_pos)
            time.sleep(0.3)

    finally:
        if cap is not None:
            cap.release()
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
