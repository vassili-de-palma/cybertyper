#!/usr/bin/env python
"""Move the robot tip to (x, y) and descend until z stops changing.

Exposes one function:

    contact_xyz = touch_xy(robot, chain, x, y, plane_z, ...)

It does:
  1. IK to a hover pose at (x, y, plane_z + hover_mm).
  2. Slew there in joint space.
  3. Descend along -z via Jacobian velocity control until contact is
     detected (motor stall OR shoulder_lift load spike) -- i.e. until
     z is no longer changing.
  4. Retreat back to the hover pose.

Also runnable as a script:

    python scripts/touch_xy.py --x 0.18 --y -0.03 --plane-z -0.02

Used by scripts/click_homography_test.py to actually touch each clicked
pixel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

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
from cybertyping.core.descent_jac import descent_jacobian_until_contact

# The click_homography calibration records gripper_frame_link directly (no
# TOOL_TIP_OFFSET arithmetic). To stay consistent, target the same URDF
# endpoint here too -- pass a zero "tip offset" everywhere, which makes
# fk_tip and solve_ik_tip operate on gripper_frame_link itself.
_ZERO_TIP_OFFSET = np.zeros(3, dtype=float)


def _read_joints_deg(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.array(
        [float(obs[f"{n}.pos"]) for n in BODY_MOTOR_ORDER], dtype=float,
    )


def touch_xy(
    robot,
    chain,
    x: float,
    y: float,
    plane_z: float,
    *,
    gripper_pos: float = 100.0,
    hover_mm: float = 50.0,
    descent_speed_mps: float = 0.005,
    max_descent_m: float = 0.10,
    stall_thresh_deg: float = 10.0,
    load_thresh: int = 60,
    xy_drift_abort_mm: float = 4.0,
    dwell_s: float = 0.4,
    retreat: bool = True,
) -> dict:
    """Slew to (x, y, plane_z + hover) then descend until contact.

    Args:
        robot:           connected SO101Follower.
        chain:           ikpy chain (build_chain(URDF_PATH, locked_joints=LOCKED_JOINTS)).
        x, y:            target tip XY in robot frame (m).
        plane_z:         estimated table/checkerboard z (m). Hover starts at
                         plane_z + hover_mm.
        hover_mm:        height above plane_z to start the descent from.
        descent_speed_mps, ..., xy_drift_abort_mm:
                         passed through to descent_jacobian_until_contact.
        dwell_s:         how long to sit at contact before retreating.
        retreat:         if True, return to the hover pose after contact.

    Returns:
        dict with keys: hover_q, hover_tip, contact_q, contact_tip,
        descent_info (the return dict from descent_jacobian_until_contact).
    """
    tip_off = _ZERO_TIP_OFFSET
    hover_z = plane_z + hover_mm / 1000.0
    hover_xyz = np.array([x, y, hover_z], dtype=float)

    # ---------- SLEW TO HOVER ----------
    q_now = _read_joints_deg(robot)
    q_hover, res_h = solve_ik_tip(
        chain, hover_xyz, q_now, tip_off,
        enforce_gripper_down=ENFORCE_GRIPPER_DOWN,
    )
    print(f"[touch_xy] hover IK -> ({x:+.4f}, {y:+.4f}, {hover_z:+.4f}) "
          f"res={res_h * 1000:.1f} mm")
    ticks = slew_segment(q_now, q_hover, MAX_STEP_DEG_PER_TICK_APPROACH)
    stream(robot, ticks, gripper_pos)
    # Settle for half a second so the load reading is stable.
    settle_action = {f"{n}.pos": float(q_hover[i])
                     for i, n in enumerate(BODY_MOTOR_ORDER)}
    settle_action["gripper.pos"] = gripper_pos
    for _ in range(15):
        robot.send_action(settle_action)
        time.sleep(1.0 / 30)

    q_at_hover = _read_joints_deg(robot)
    tip_at_hover = fk_tip(chain, q_at_hover, tip_off)
    print(f"[touch_xy] at hover: tip = {tip_at_hover.round(4).tolist()}")

    # ---------- DESCEND UNTIL CONTACT ----------
    contact_q, info = descent_jacobian_until_contact(
        chain, robot, BODY_MOTOR_ORDER, q_at_hover, tip_off, gripper_pos,
        descent_speed_mps=descent_speed_mps,
        max_descent_m=max_descent_m,
        stall_thresh_deg=stall_thresh_deg,
        load_thresh=load_thresh,
        xy_drift_abort_mm=xy_drift_abort_mm,
    )
    print()
    print(f"[touch_xy] descent stop: {info.get('reason')}  "
          f"z_traveled={info.get('z_traveled_mm', 0):.1f} mm")
    contact_tip = fk_tip(chain, contact_q, tip_off)
    print(f"[touch_xy] contact tip = {contact_tip.round(4).tolist()}")
    print(f"[touch_xy] horizontal miss: "
          f"({(contact_tip[0] - x) * 1000:+.2f}, "
          f"{(contact_tip[1] - y) * 1000:+.2f}) mm")

    # Brief dwell.
    if dwell_s > 0:
        dwell_action = {f"{n}.pos": float(contact_q[i])
                        for i, n in enumerate(BODY_MOTOR_ORDER)}
        dwell_action["gripper.pos"] = gripper_pos
        n = max(1, int(dwell_s * 30))
        for _ in range(n):
            robot.send_action(dwell_action)
            time.sleep(1.0 / 30)

    # ---------- RETREAT TO HOVER ----------
    if retreat:
        print(f"[touch_xy] retreating to hover")
        ticks = slew_segment(contact_q, q_hover, MAX_STEP_DEG_PER_TICK_RETREAT)
        stream(robot, ticks, gripper_pos)

    return {
        "hover_q": q_hover,
        "hover_tip": tip_at_hover,
        "contact_q": contact_q,
        "contact_tip": contact_tip,
        "descent_info": info,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, required=True, help="target tip x (m)")
    ap.add_argument("--y", type=float, required=True, help="target tip y (m)")
    ap.add_argument("--plane-z", type=float, required=True,
                    help="estimated table z (m). Hover starts plane_z+hover_mm.")
    ap.add_argument("--hover-mm", type=float, default=50.0)
    ap.add_argument("--gripper-pos", type=float, default=100.0)
    ap.add_argument("--descent-speed-mps", type=float, default=0.005)
    ap.add_argument("--max-descent-m", type=float, default=0.10)
    ap.add_argument("--stall-thresh-deg", type=float, default=1.5)
    ap.add_argument("--load-thresh", type=int, default=60)
    ap.add_argument("--xy-drift-abort-mm", type=float, default=4.0)
    args = ap.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    port = os.environ.get("SO101_PORT", "COM5")
    robot_id = os.environ.get("SO101_ROBOT_ID", "team08_follower_arm")
    cfg = SO101FollowerConfig(port=port, id=robot_id, max_relative_target=None)
    robot = SO101Follower(cfg)
    print(f"[connect] {port} id={robot_id}")
    robot.connect()

    chain = build_chain(URDF_PATH, locked_joints=LOCKED_JOINTS)
    try:
        result = touch_xy(
            robot, chain, args.x, args.y, args.plane_z,
            gripper_pos=args.gripper_pos,
            hover_mm=args.hover_mm,
            descent_speed_mps=args.descent_speed_mps,
            max_descent_m=args.max_descent_m,
            stall_thresh_deg=args.stall_thresh_deg,
            load_thresh=args.load_thresh,
            xy_drift_abort_mm=args.xy_drift_abort_mm,
        )
        print(json.dumps({
            "contact_tip": [float(v) for v in result["contact_tip"]],
            "reason": result["descent_info"].get("reason"),
            "z_traveled_mm": result["descent_info"].get("z_traveled_mm"),
        }, indent=2))
    finally:
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
