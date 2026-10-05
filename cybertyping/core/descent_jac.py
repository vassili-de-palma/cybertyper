"""Jacobian velocity-based descent with stall + load contact detection.

IK-per-tick descent re-targets (fixed_x, fixed_y, decreasing_z) on every step.
At the workspace edge, the IK residual VARIES with z, so the achieved pen-tip
xy drifts even though the IK xy *command* is constant. The pen ends up sliding
forward as z decreases — fatal for keyboards.

This module instead does VELOCITY control in joint space:
    1. Compute FK Jacobian J_pos at current q.
    2. Solve for joint velocities producing pure -z velocity at pen tip:
       dq = pinv(J_pos) @ [0, 0, -step_z_per_tick].
    3. Send q + dq (no re-targeting of xy).
    4. Stop on either:
        a) stall (commanded vs actual joint position gap > threshold = pen
           physically blocked by surface),
        b) load spike (shoulder_lift |Δload| > threshold from in-flight
           baseline; same primitive as touch_sequence_load.descent_until_contact),
        c) max-descent safety.

Pseudo-inverse handles arm redundancy gracefully. Lateral drift is bounded
by IK linearization error, not by IK saturation. In practice <1 mm at
workspace edge for sub-cm descents.
"""
from __future__ import annotations

import statistics
import time
from collections.abc import Sequence

import numpy as np


def _fk_origin(chain, q_body_deg, body_motor_order):
    """Origin of gripper_frame_link (without tool offset)."""
    full = np.zeros(len(chain.links))
    for i, name in enumerate(body_motor_order):
        full[chain._motor_link_idx[name]] = np.deg2rad(float(q_body_deg[i]))
    return np.asarray(chain.forward_kinematics(full))[:3, 3]


def _fk_tip(chain, q_body_deg, body_motor_order, tip_offset_local):
    full = np.zeros(len(chain.links))
    for i, name in enumerate(body_motor_order):
        full[chain._motor_link_idx[name]] = np.deg2rad(float(q_body_deg[i]))
    T = np.asarray(chain.forward_kinematics(full))
    return (T @ np.append(np.asarray(tip_offset_local, dtype=float), 1.0))[:3]


def compute_pos_jacobian_numeric(
    chain, q_body_deg, body_motor_order, tip_offset_local, eps_deg: float = 0.5
) -> np.ndarray:
    """3xN position Jacobian of the pen tip wrt body joint angles (m / rad)."""
    n = len(body_motor_order)
    J = np.zeros((3, n))
    tip0 = _fk_tip(chain, q_body_deg, body_motor_order, tip_offset_local)
    eps_rad = np.deg2rad(eps_deg)
    for i in range(n):
        q_plus = q_body_deg.copy()
        q_plus[i] += eps_deg
        tip_plus = _fk_tip(chain, q_plus, body_motor_order, tip_offset_local)
        J[:, i] = (tip_plus - tip0) / eps_rad
    return J


def descent_jacobian_until_contact(
    chain,
    robot,
    body_motor_order: Sequence[str],
    hover_q: np.ndarray,
    tip_offset_local: Sequence[float],
    gripper_pos: float,
    *,
    descent_speed_mps: float = 0.005,    # 5 mm/s (slow for stable readings)
    loop_hz: int = 30,
    max_descent_m: float = 0.10,
    stall_thresh_deg: float = 1.5,
    xy_drift_abort_mm: float = 2.0,      # ABORT if pen tip slides laterally > this
    load_thresh: int = 60,
    contact_motor: str = "shoulder_lift",
    skip_first_ticks: int = 8,
    baseline_n_ticks: int = 10,
):
    """Lower the pen tip by stepping joints along the inverse Jacobian of FK.

    Returns (q_at_contact, info_dict).
    """
    q = np.array(hover_q, dtype=float)
    tip_off = np.asarray(tip_offset_local, dtype=float)
    initial_tip = _fk_tip(chain, q, body_motor_order, tip_off)
    period = 1.0 / loop_hz
    step_z = descent_speed_mps / loop_hz   # m per tick

    baseline_buf: list[int] = []
    baseline: int | None = None
    consec_load = 0
    consec_stall = 0
    tick = 0
    z_traveled = 0.0

    while True:
        t0 = time.perf_counter()
        z_traveled += step_z

        if z_traveled > max_descent_m:
            return q, {
                "reason": f"max descent {max_descent_m*1000:.0f} mm reached",
                "ticks": tick,
                "z_traveled_mm": z_traveled * 1000,
                "peak_delta": 0,
            }

        # Jacobian @ current q
        J_pos = compute_pos_jacobian_numeric(chain, q, body_motor_order, tip_off)
        # Pseudo-inverse for pure -z velocity
        v = np.array([0.0, 0.0, -step_z / period])   # m/s
        try:
            dq_rad_per_s = np.linalg.pinv(J_pos) @ v
        except np.linalg.LinAlgError:
            return q, {"reason": "Jacobian singular", "ticks": tick,
                       "z_traveled_mm": z_traveled * 1000, "peak_delta": 0}
        dq_deg = np.rad2deg(dq_rad_per_s * period)
        q_new = q + dq_deg

        # Joint bounds clamp (safety)
        for i, name in enumerate(body_motor_order):
            lo, hi = chain.links[chain._motor_link_idx[name]].bounds
            if np.isfinite(lo) and np.isfinite(hi):
                q_new[i] = float(np.clip(q_new[i], np.rad2deg(lo), np.rad2deg(hi)))

        # Send action
        action = {f"{n}.pos": float(q_new[i]) for i, n in enumerate(body_motor_order)}
        action["gripper.pos"] = gripper_pos
        try:
            robot.send_action(action)
        except Exception:
            pass

        # Read state
        load_now = 0
        actual_q = q_new
        try:
            obs = robot.get_observation()
            actual_q = np.array(
                [float(obs[f"{n}.pos"]) for n in body_motor_order], dtype=float,
            )
            loads = robot.bus.sync_read("Present_Load")
            load_now = int(loads.get(contact_motor, 0))
        except Exception:
            pass

        stall_err = float(np.max(np.abs(q_new - actual_q)))
        # Measure ACTUAL pen tip xy drift from descent start. Crucial backstop:
        # once pen touches the table, further descent commands can no longer
        # lower z, so joints find a lateral direction to "satisfy" the
        # commanded velocity -> pen slides on the surface. Catching this drift
        # early stops sliding.
        actual_tip_now = _fk_tip(chain, actual_q, body_motor_order, tip_off)
        xy_drift_x_mm = (actual_tip_now[0] - initial_tip[0]) * 1000
        xy_drift_y_mm = (actual_tip_now[1] - initial_tip[1]) * 1000
        xy_drift_norm_mm = float(np.hypot(xy_drift_x_mm, xy_drift_y_mm))
        if tick >= skip_first_ticks and xy_drift_norm_mm > xy_drift_abort_mm:
            return q_new, {
                "reason": f"XY DRIFT {xy_drift_norm_mm:.2f} mm > {xy_drift_abort_mm} mm "
                          f"(pen sliding on surface)",
                "ticks": tick + 1,
                "z_traveled_mm": z_traveled * 1000,
                "peak_delta": 0,
                "xy_drift_mm": [xy_drift_x_mm, xy_drift_y_mm],
            }

        if tick < skip_first_ticks:
            phase = "warmup"
            delta = 0
        elif baseline is None:
            baseline_buf.append(load_now)
            phase = "baseline"
            delta = 0
            if len(baseline_buf) >= baseline_n_ticks:
                baseline = int(statistics.median(baseline_buf))
                print(f"\n  [base] in-flight baseline ({len(baseline_buf)} samples) = "
                      f"{baseline:+d}", flush=True)
        else:
            delta = abs(load_now - baseline)
            phase = "monitor"

            # Stall: motor physically blocked
            if stall_err > stall_thresh_deg:
                consec_stall += 1
                if consec_stall >= 2:
                    return q_new, {
                        "reason": f"STALL {stall_err:.1f}° > {stall_thresh_deg}°",
                        "ticks": tick + 1,
                        "z_traveled_mm": z_traveled * 1000,
                        "peak_delta": delta,
                    }
            else:
                consec_stall = 0

            # High load (real contact force)
            if delta > load_thresh:
                consec_load += 1
                if consec_load >= 2:
                    return q_new, {
                        "reason": f"LOAD |Δ|={delta} > {load_thresh}",
                        "ticks": tick + 1,
                        "z_traveled_mm": z_traveled * 1000,
                        "peak_delta": delta,
                    }
            else:
                consec_load = 0

        # Status line
        actual_tip = _fk_tip(chain, actual_q, body_motor_order, tip_off)
        print(
            f"  [{phase:8s}] tick {tick:3d}  z_actual={actual_tip[2]*1000:+6.1f}mm  "
            f"xy_drift_mm=({(actual_tip[0]-initial_tip[0])*1000:+.2f},"
            f"{(actual_tip[1]-initial_tip[1])*1000:+.2f})  "
            f"load={load_now:+5d} Δ={delta:+4d}  stall={stall_err:.2f}°    ",
            end="\r", flush=True,
        )

        # Keep commanded q as the planner anchor (motors lag a few degrees;
        # re-anchoring to actual_q causes the planner to fight the motor PID).
        q = q_new
        tick += 1
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)
