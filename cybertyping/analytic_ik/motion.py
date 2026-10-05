"""High-level motion routines: reach-to-hover and descend-until-touch.

Pure motion logic built on :func:`kinematics.solve_ik` (closed-form) and the
joint-space slew/stream helpers from :mod:`cybertyping.core.primitives`.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .config import BODY_MOTOR_ORDER, CONTROL_HZ
from .kinematics import solve_ik, urdf_rad_to_lerobot_deg


def _read_joints_deg(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.array([float(obs[f"{n}.pos"]) for n in BODY_MOTOR_ORDER], dtype=float)


def reach_hover(
    robot,
    x: float,
    y: float,
    z_hover: float,
    *,
    current_gripper_pos: float,
    max_step_deg_per_tick: float = 2.0,
    loop_hz: float = 30.0,
    hold_seconds: float = 0.5,
    verbose: bool = True,
) -> np.ndarray:
    """Solve IK for the hover pose and slew the arm there in joint space.

    Returns the body-joint vector (lerobot degrees) read back after settling.
    """
    from cybertyping.core.primitives import slew_segment  # lazy: needs ikpy

    q_rad = solve_ik(x, y, z_hover)
    q_deg = urdf_rad_to_lerobot_deg(q_rad)
    if verbose:
        print(f"[reach] target ({x:+.3f}, {y:+.3f}, {z_hover:+.3f}) -> "
              f"q_deg={np.array2string(q_deg, precision=2)}")

    q_now = _read_joints_deg(robot)
    period = 1.0 / float(loop_hz)
    for q in slew_segment(q_now, np.asarray(q_deg, dtype=float), max_step_deg_per_tick):
        t0 = time.perf_counter()
        action = {f"{n}.pos": float(q[i]) for i, n in enumerate(BODY_MOTOR_ORDER)}
        action["gripper.pos"] = float(current_gripper_pos)
        robot.send_action(action)
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)
    time.sleep(hold_seconds)
    return _read_joints_deg(robot)


def descend_until_touch(
    robot,
    x: float,
    y: float,
    *,
    z_start: float,
    z_floor: float,
    v_desc: float,
    touch_fn: Callable[[], bool],
    current_gripper_pos: float,
    control_hz: int = CONTROL_HZ,
    verbose: bool = True,
) -> float | None:
    """Constant-velocity Cartesian-z descent. Re-solves IK each tick from
    closed-form, so there is no Jacobian linearisation drift.

    Returns:
        z_touch (float) if contact was detected before reaching the floor,
        None otherwise.
    """
    # Pre-validate the entire descent range: fails fast if any z is
    # geometrically out of reach or violates a joint limit.
    for z_check in np.linspace(z_start, z_floor, 50):
        solve_ik(x, y, float(z_check))

    period = 1.0 / float(control_hz)
    t0 = time.monotonic()
    if verbose:
        print(f"[descend] {x:+.3f},{y:+.3f}  z: {z_start:+.3f} -> {z_floor:+.3f}  "
              f"v={v_desc * 1000:.1f} mm/s  hz={control_hz}")

    while True:
        t = time.monotonic() - t0
        z = max(z_start - v_desc * t, z_floor)

        if touch_fn():
            if verbose:
                print(f"[descend] CONTACT at z={z:+.4f}")
            return float(z)

        if z <= z_floor:
            if verbose:
                print(f"[descend] reached floor z={z_floor:+.4f} without contact")
            return None

        q_rad = solve_ik(x, y, float(z))
        q_deg = urdf_rad_to_lerobot_deg(q_rad)
        action = {f"{n}.pos": float(q_deg[i])
                  for i, n in enumerate(BODY_MOTOR_ORDER)}
        action["gripper.pos"] = float(current_gripper_pos)
        robot.send_action(action)
        time.sleep(period)
