"""CLI entry point.

Usage:
    python -m cybertyping.analytic_ik.main --x 0.20 --y 0.05
    python -m cybertyping.analytic_ik.main --x 0.20 --y 0.05 --z-hover 0.15 --v-desc 0.01

Dry-run (no robot motion, just print the IK solution):
    $env:CYBERTYPING_DRY_RUN = "1"
    python -m cybertyping.analytic_ik.main --x 0.20 --y 0.05
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# Allow `python -m cybertyping.analytic_ik.main` from the repo root or any cwd.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from .config import (
    BODY_MOTOR_ORDER, ELBOW_NAME, SHOULDER_NAME, TOUCH_PERSIST,
    TOUCH_THRESHOLD, TOUCH_WARMUP_TICKS, TOUCH_WINDOW, V_DESCENT,
    Z_FLOOR, Z_HOVER,
)
from .kinematics import (
    JointLimitViolation, OutOfReach, forward_kinematics,
    lerobot_deg_to_urdf_rad, solve_ik,
)
from .motion import descend_until_touch, reach_hover
from .touch import TouchDetector


def _is_dry_run() -> bool:
    return os.environ.get("CYBERTYPING_DRY_RUN", "").lower() in ("1", "true", "yes")


@contextmanager
def _robot_context():
    """Lazy import of cybertyping.core.runtime — the dry-run path does not
    touch it, so the package's heavy transitive dependencies (ikpy etc.)
    don't have to be installed to verify CLI plumbing."""
    if _is_dry_run():
        yield None
        return
    from cybertyping.core.runtime import maybe_robot
    with maybe_robot() as robot:
        yield robot


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--x", type=float, required=True, help="target x (m, base frame)")
    p.add_argument("--y", type=float, required=True, help="target y (m, base frame)")
    p.add_argument("--z-hover", type=float, default=Z_HOVER,
                   help=f"hover height above target (m, default {Z_HOVER})")
    p.add_argument("--z-floor", type=float, default=Z_FLOOR,
                   help=f"safety floor (m, default {Z_FLOOR})")
    p.add_argument("--v-desc", type=float, default=V_DESCENT,
                   help=f"descent speed (m/s, default {V_DESCENT})")
    p.add_argument("--touch-sign", type=int, default=+1, choices=(-1, +1),
                   help="sign of the contact load deviation (calibrate empirically)")
    p.add_argument("--reach-only", action="store_true",
                   help="reach to hover and stop — skip descent + touch")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Pre-flight: validate the WHOLE descent range, not just the hover pose.
    # If any z between z_hover and z_floor is unreachable, abort before
    # connecting to the robot.
    try:
        q_plan = solve_ik(args.x, args.y, args.z_hover)
        unreachable_z = []
        for z_check in np.linspace(args.z_hover, args.z_floor, 50):
            try:
                solve_ik(args.x, args.y, float(z_check))
            except (OutOfReach, JointLimitViolation):
                unreachable_z.append(float(z_check))
        if unreachable_z:
            print(f"[ik] descent range contains {len(unreachable_z)}/50 "
                  f"unreachable z values "
                  f"(first bad z = {unreachable_z[0]:+.4f} m)",
                  file=sys.stderr)
            return 2
    except (OutOfReach, JointLimitViolation) as e:
        print(f"[ik] hover pose unreachable: {e}", file=sys.stderr)
        return 2

    q_deg_plan = np.rad2deg(q_plan)
    fk_xyz = forward_kinematics(q_plan)
    fk_err_mm = 1000.0 * float(np.linalg.norm(
        [args.x - fk_xyz[0], args.y - fk_xyz[1], args.z_hover - fk_xyz[2]]
    ))
    print(f"[ik] hover at ({args.x:+.3f}, {args.y:+.3f}, {args.z_hover:+.3f}) m")
    print(f"[ik]   q_deg = [{q_deg_plan[0]:+7.2f}, {q_deg_plan[1]:+7.2f}, "
          f"{q_deg_plan[2]:+7.2f}, {q_deg_plan[3]:+7.2f}, {q_deg_plan[4]:+7.2f}]")
    print(f"[ik]   FK round-trip err = {fk_err_mm:.3f} mm")
    print(f"[ik]   full descent range [{args.z_hover:+.3f} -> "
          f"{args.z_floor:+.3f}] reachable")

    with _robot_context() as robot:
        if robot is None:
            print("[dry-run] CYBERTYPING_DRY_RUN set — not connecting to robot.")
            return 0

        obs = robot.get_observation()
        grip = float(obs.get("gripper.pos", 100.0))

        # 1. Reach hover.
        print(f"[reach] hovering at ({args.x:+.3f}, {args.y:+.3f}, "
              f"{args.z_hover:+.3f}) …")
        reach_hover(robot, args.x, args.y, args.z_hover,
                    current_gripper_pos=grip)

        if args.reach_only:
            print("[reach-only] hover complete; skipping descent.")
            return 0

        # 2. Prime the touch detector while the arm is stationary at hover.
        detector = TouchDetector(
            robot, [SHOULDER_NAME, ELBOW_NAME],
            window=TOUCH_WINDOW, threshold=TOUCH_THRESHOLD,
            persist=TOUCH_PERSIST, sign=args.touch_sign,
        )
        print(f"[touch] priming baseline ({TOUCH_WARMUP_TICKS} ticks) …")
        detector.prime(ticks=TOUCH_WARMUP_TICKS)

        # 3. Descend.
        z_touch = descend_until_touch(
            robot, args.x, args.y,
            z_start=args.z_hover, z_floor=args.z_floor,
            v_desc=args.v_desc, touch_fn=detector,
            current_gripper_pos=grip,
        )

        if z_touch is None:
            print("[result] no contact detected by floor; aborting.")
            return 1

        # 4. Report true contact z from encoder FK.
        obs = robot.get_observation()
        q_deg = np.array(
            [float(obs[f"{n}.pos"]) for n in BODY_MOTOR_ORDER], dtype=float,
        )
        q_rad = lerobot_deg_to_urdf_rad(q_deg)
        x_true, y_true, z_true = forward_kinematics(q_rad)
        print(f"[result] commanded z_touch = {z_touch:+.4f} m")
        print(f"[result] encoder FK    EE = ({x_true:+.4f}, {y_true:+.4f}, "
              f"{z_true:+.4f}) m")

        # 5. Retreat 5 mm.
        print("[retreat] +5 mm above contact …")
        reach_hover(robot, args.x, args.y, z_touch + 0.005,
                    current_gripper_pos=grip)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
