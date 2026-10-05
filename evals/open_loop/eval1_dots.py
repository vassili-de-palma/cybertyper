"""Eval 1 (50pts), Reaching four colored points on an A4 sheet.

RUBRIC (from the project brief):
    Hover the gripper above each of four colored points (#FF0000, #00FF00,
    #0000FF, #7FFFFF) in that order. Hover = within 3 cm above the point,
    overlap at least 1/2 of the (~1.5 cm) dot. Must finish in 25 s.

Strategy:
    1. Move to the SCAN POSE (overhead wrist-cam view of the sheet).
    2. Detect the four dots with HSV segmentation (perception/dots.py).
    3. Pixel -> robot XY via wrist-cam extrinsics (eye-in-hand).
    4. For each dot in the rubric order:
         - IK to (rx, ry, HOVER_HEIGHT)
         - slew there, dwell a beat to clearly satisfy "hovers at most 3 cm"
         - move on.
    5. Return to a safe home pose.

NO contact-sensing required for this eval, we never touch the paper.

USAGE:
    python -m evals.open_loop.eval1_dots
    # dry-run (no robot, just plan from a saved image):
    CYBERTYPING_DRY_RUN=1 python -m evals.open_loop.eval1_dots --image data/dots.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from cybertyping.core.runtime import (
    banner, new_log, load_world, load_scan_pose, maybe_robot, is_dry_run, URDF_PATH,
)
from cybertyping.perception.capture import open_camera
from cybertyping.perception.detect.color_blobs import (
    EVAL1_DOT_ORDER, detect_dots, annotate_dots,
)
from cybertyping.perception.extrinsics.fov_approx import (
    PlaneModel, legacy_fov_pixel_to_robot,
)
from cybertyping.perception.extrinsics import calibrated as _cal
from cybertyping import core

# ----------------------------- CONFIG ---------------------------------------

# 1.5 cm above the dot. The rubric allows up to 3 cm. We choose 1.5 cm to give
# headroom for FK error and pen length.
HOVER_HEIGHT_M = 0.015

# Total budget is 25 s. With 4 hovers and ~3 s of approach/hover each, we have
# plenty of slack. Use joint-space slew at moderate speed.
# Slower than main's MAX_STEP_DEG_PER_TICK_APPROACH=4.0 because eval1 just
# hovers (no contact) and the rubric tolerance is generous (1.5 cm dot,
# 3 cm hover height). Erring on the side of smoother motion for the demo.
MAX_STEP_DEG_PER_TICK = 1.6
DWELL_AT_HOVER_S      = 0.5

# Fallback HFOV for the wrist camera if no intrinsics file exists.
DEFAULT_HFOV_DEG = 60.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image",  type=str, default=None,
                    help="Use a saved JPEG instead of a live camera (dev/dry-run).")
    ap.add_argument("--save-annotated", type=str, default="runs/eval1_dots_annot.jpg")
    args = ap.parse_args()

    banner("Eval 1, four colored dots")

    # 0) World / scan pose preconditions
    world = load_world()
    if world is None:
        print("ERROR: world_settings_load.json not found. "
              "Run touch_sequence_load.py once to capture the plane + bbox.")
        return 2
    plane = PlaneModel.from_world_settings(world)
    scan  = load_scan_pose()
    if scan is None and not is_dry_run():
        print("ERROR: configs/scan_pose.json not found. "
              "Capture one with `python -m cybertyping.calibration.session_calibration`.")
        return 2

    log = new_log("eval1_dots")

    cal = _cal.RigCalibration.try_load(_cal.DEFAULT_PATH)
    if cal is not None:
        print(f"[cal] loaded rig_calibration.json (reproj="
              f"{cal.reproj_error_px:.2f} px, bias_samples={len(cal.bias_samples)})")

    with maybe_robot() as robot:
        # 1) Move to scan pose (or skip in dry-run)
        chain = core.build_chain(URDF_PATH, locked_joints=core.LOCKED_JOINTS)
        T_base_camera = None
        if robot is not None:
            print("[scan] Moving to scan pose...")
            # If we have an exact joint-space scan pose saved, use that for repeatability.
            # Otherwise IK to the saved (x,y,z).
            target_xyz = np.array([scan["x"], scan["y"], scan["z"]], dtype=float)
            obs   = robot.get_observation()
            q_now = np.array([float(obs[f"{n}.pos"]) for n in core.BODY_MOTOR_ORDER])
            q_scan = core.ik.solve_origin(chain, target_xyz, q_now).q
            traj   = core.slew_segment(q_now, q_scan, MAX_STEP_DEG_PER_TICK)
            core.stream(robot, traj, gripper_pos=float(obs["gripper.pos"]))
            time.sleep(0.3)
            obs   = robot.get_observation()
            q_at  = np.array([float(obs[f"{n}.pos"]) for n in core.BODY_MOTOR_ORDER])
            cam_xyz = core.fk_tip(chain, q_at, core.TOOL_TIP_OFFSET_LOCAL_XYZ)
            if cal is not None:
                T_base_camera = cal.T_base_camera(core.fk_T(chain, q_at))
            print(f"[scan] tip @ ({cam_xyz[0]:+.3f}, {cam_xyz[1]:+.3f}, {cam_xyz[2]:+.3f}) m")
        else:
            cam_xyz = np.array([scan["x"] if scan else 0.20,
                                scan["y"] if scan else 0.00,
                                scan["z"] if scan else 0.30])

        # 2) Capture a frame and detect dots
        print("[capture] Grabbing frame...")
        with open_camera(mock_path=args.image, index=args.camera) as cam:
            frame = cam.grab()
        h, w = frame.shape[:2]
        print(f"[capture] {w}x{h}")

        dots = detect_dots(frame)
        for color in EVAL1_DOT_ORDER:
            if color not in dots:
                print(f"  [MISS] {color}: not found, re-check lighting / HSV bands.")
            else:
                px, py = dots[color].pixel_xy
                print(f"  {color}: pixel ({px:.1f}, {py:.1f})  area={dots[color].area_px:.0f}")

        # 3) Convert pixels -> robot XY. Prefer the calibrated path (proper
        # pinhole undistort + plane intersection + bias map from
        # rig_calibration.json) when available; fall back to the legacy FOV
        # approx for dry-runs / un-calibrated rigs.
        use_calibrated = cal is not None and T_base_camera is not None
        if use_calibrated:
            print(f"[map] using calibrated pixel->robot")
        else:
            print(f"[map] using legacy FOV approx (no rig_calibration.json "
                  f"or dry-run)")

        ordered_targets: list[tuple[str, tuple[float, float, float]]] = []
        for color in EVAL1_DOT_ORDER:
            if color not in dots:
                continue
            px, py = dots[color].pixel_xy
            if use_calibrated:
                hit = _cal.pixel_to_robot(px, py, core.fk_T(chain, q_at), cal)
                if hit is None:
                    print(f"  {color}: calibrated back-projection failed, "
                          f"falling back to FOV approx")
                    rx, ry, rz = legacy_fov_pixel_to_robot(
                        px, py, w, h, tuple(cam_xyz), plane, DEFAULT_HFOV_DEG,
                    )
                else:
                    rx, ry, rz = hit
            else:
                rx, ry, rz = legacy_fov_pixel_to_robot(
                    px, py, w, h, tuple(cam_xyz), plane, DEFAULT_HFOV_DEG,
                )
            ordered_targets.append((color, (rx, ry, rz)))
            print(f"  {color} -> robot ({rx:+.4f}, {ry:+.4f}, {rz:+.4f})")
            log.add(event="dot_detected", color=color,
                    pixel=[float(px), float(py)], robot=[rx, ry, rz])

        if len(ordered_targets) < 4:
            print(f"WARNING: only {len(ordered_targets)}/4 dots detected. Proceeding.")

        # 4) Save annotated image for debugging
        out_path = Path(args.save_annotated)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        cv2.imwrite(str(out_path), annotate_dots(frame, dots))
        print(f"[debug] Annotated frame -> {out_path}")

        # 5) Execute hovers
        if robot is None:
            print("[dry-run] skipping motion.")
            log.save(f"runs/eval1_{int(time.time())}.json")
            return 0

        for color, (rx, ry, rz) in ordered_targets:
            hover_xyz = np.array([rx, ry, rz + HOVER_HEIGHT_M])
            obs   = robot.get_observation()
            q_now = np.array([float(obs[f"{n}.pos"]) for n in core.BODY_MOTOR_ORDER])
            q_tgt = core.ik.solve_tip(chain, hover_xyz, q_now, core.TOOL_TIP_OFFSET_LOCAL_XYZ).q
            traj  = core.slew_segment(q_now, q_tgt, MAX_STEP_DEG_PER_TICK)
            print(f"[hover] {color}: -> ({rx:+.4f}, {ry:+.4f}, {rz + HOVER_HEIGHT_M:+.4f})")
            core.stream(robot, traj, gripper_pos=float(obs["gripper.pos"]))
            time.sleep(DWELL_AT_HOVER_S)
            log.add(event="hovered", color=color)

        log.save(f"runs/eval1_{int(time.time())}.json")
        print("[done] Eval 1 sequence complete.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
