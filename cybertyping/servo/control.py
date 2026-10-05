"""Servo loop, camera-orientation hold and the two descents.

All motion here is *velocity-style*: each tick computes a small Cartesian
step for the tool tip, converts it to joint increments through the
pseudo-inverse of the numeric position Jacobian, clips, and sends. Nothing
re-targets an absolute pose, so the IK residual cannot drift the tip
sideways during a descent (the failure mode of the open-loop pipeline).

Tunables are module constants; the values are the ones used on demo day.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from cybertyping.core import primitives as prim
from cybertyping.core.descent_jac import compute_pos_jacobian_numeric
from cybertyping.servo.camera import draw_cross, put_status
from cybertyping.servo.rig import Rig
from cybertyping.servo.targets import Target, find_tip

# Pixel error -> robot XY step (m per pixel). The wrist camera is mounted so
# image +x is roughly robot -y and image +y is roughly robot -x, hence the
# swapped, negative layout. Flip a sign if the tip moves away from the
# target. The magnitude only sets the loop gain; steps are clipped below.
IMAGE_TO_ROBOT_GAIN_MAT = np.array([
    [0.0,   -3e-4],
    [-3e-4,  0.0],
], dtype=float)

MAX_XY_STEP_M = 0.002        # per-tick cap on the servo XY step
MAX_SERVO_DQ_DEG = 0.5       # per-tick joint clip during servoing
MAX_DESCENT_DQ_DEG = 2.0     # per-tick joint clip during descents

WRIST_FLEX_IDX = prim.BODY_MOTOR_ORDER.index("wrist_flex")
WRIST_FLEX_LIMITS_DEG = (-95.0, 95.0)
CAMERA_HOLD_MAX_DELTA_PER_TICK_DEG = 2.0

GREEN, BLUE, YELLOW, RED = (0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 0, 255)


# ---------------------------------------------------------------------------
# Orientation hold
# ---------------------------------------------------------------------------

def enforce_camera_dir(chain, q_deg: np.ndarray, target_dir: np.ndarray,
                       max_iters: int = 6, tol: float = 1e-3) -> np.ndarray:
    """Adjust wrist_flex so the gripper +Z axis points along ``target_dir``.

    One-dimensional Gauss-Newton on wrist_flex with a numeric derivative of
    the gripper +Z direction. The total change per call is rate-limited so
    a single tick cannot snap the wrist.
    """
    q = q_deg.astype(float).copy()
    eps_deg = 1.0
    wf_start = q[WRIST_FLEX_IDX]
    for _ in range(max_iters):
        z = prim.fk_T(chain, q)[:3, 2]
        err = target_dir - z
        if np.linalg.norm(err) < tol:
            break
        q_pert = q.copy()
        q_pert[WRIST_FLEX_IDX] += eps_deg
        dz = (prim.fk_T(chain, q_pert)[:3, 2] - z) / eps_deg
        denom = float(dz @ dz)
        if denom < 1e-9:
            break
        q[WRIST_FLEX_IDX] += float(np.clip(float(err @ dz) / denom, -5.0, 5.0))
    delta = float(np.clip(q[WRIST_FLEX_IDX] - wf_start,
                          -CAMERA_HOLD_MAX_DELTA_PER_TICK_DEG,
                          CAMERA_HOLD_MAX_DELTA_PER_TICK_DEG))
    q[WRIST_FLEX_IDX] = float(np.clip(wf_start + delta, *WRIST_FLEX_LIMITS_DEG))
    return q


def _step_joints(chain, q: np.ndarray, delta_xyz: np.ndarray, tip_off,
                 clip_deg: float) -> np.ndarray | None:
    """q + pinv(J) @ delta_xyz, clipped. None if the Jacobian is singular."""
    J = compute_pos_jacobian_numeric(chain, q, prim.BODY_MOTOR_ORDER, tip_off)
    try:
        dq_rad = np.linalg.pinv(J) @ delta_xyz
    except np.linalg.LinAlgError:
        return None
    return q + np.clip(np.rad2deg(dq_rad), -clip_deg, clip_deg)


def _user_abort(key: int) -> bool:
    return key in (ord("q"), 27)


# ---------------------------------------------------------------------------
# Visual servo (phases 1 and 2)
# ---------------------------------------------------------------------------

def servo_to_target(
    rig: Rig, q_init: np.ndarray, target: Target, tip_color: str, *,
    target_z: float | None = None,
    z_step_m_per_tick: float = 0.003,
    z_tol_m: float = 0.001,
    tol_px: float = 8.0,
    max_iters: int = 400,
    loop_hz: float = 10.0,
    exit_mode: str = "both",
    phase_label: str = "servo",
    wait_for_signals: bool = True,
    lock_camera_dir: bool = True,
    initial_tip_px=None,
    initial_target_px=None,
) -> np.ndarray:
    """Drive the detected tip onto the target pixel; optionally descend in Z.

    Each tick: detect tip, ask the target for its pixel, map the pixel error
    through :data:`IMAGE_TO_ROBOT_GAIN_MAT`, add a capped Z step toward
    ``target_z`` (if given), and apply both through one Jacobian solve.

    exit_mode:
        "both"     exit when |err_px| < tol_px AND |err_z| < z_tol_m
        "z_only"   exit as soon as Z is reached (diagonal approach phase)
        "xy_only"  exit as soon as the pixel error is below tol_px

    wait_for_signals:
        True   hold position until both tip and target are seen (ESC/q
               aborts). Used for keyboard keys, where the OCR may need a
               few frames.
        False  start immediately; while a signal is missing the XY step is
               zero but the Z descent still runs. Used for coloured dots.

    Returns the final joint vector.
    """
    period = 1.0 / loop_hz
    q = np.asarray(q_init, dtype=float).copy()
    tip_px = tuple(initial_tip_px) if initial_tip_px is not None else None
    target_px = tuple(initial_target_px) if initial_target_px is not None else None

    for i in range(max_iters):
        frame = rig.read_frame()
        if frame is not None:
            pt = target.update(frame)
            if pt is not None:
                target_px = pt
            tp = find_tip(frame, tip_color, prior_xy=tip_px)
            if tp is not None:
                tip_px = tp

        tip_xyz = prim.fk_tip(rig.chain, q, rig.tip_offset)
        have_both = tip_px is not None and target_px is not None

        if not have_both and wait_for_signals:
            if frame is not None:
                missing = []
                if target_px is None:
                    missing.append(f"no target '{target.label}'")
                if tip_px is None:
                    missing.append(f"no {tip_color} tip")
                put_status(frame, f"[{phase_label}] waiting: {', '.join(missing)} (ESC aborts)", RED)
                target.draw(frame)
                if _user_abort(rig.show(frame, 20)):
                    print(f"[{phase_label}] aborted by user while waiting")
                    return q
            else:
                time.sleep(period)
            continue

        if have_both:
            err_px = np.array(target_px, dtype=float) - np.array(tip_px, dtype=float)
            err_px_norm = float(np.linalg.norm(err_px))
        else:
            err_px = np.zeros(2)
            err_px_norm = float("inf")   # blocks XY convergence until seen

        if target_z is not None:
            err_z = target_z - tip_xyz[2]
            delta_z = float(np.clip(err_z, -z_step_m_per_tick, z_step_m_per_tick))
        else:
            err_z, delta_z = 0.0, 0.0

        if frame is not None:
            if tip_px is not None:
                draw_cross(frame, tip_px, GREEN)
            target.draw(frame)
            err_txt = f"{err_px_norm:.1f}px" if np.isfinite(err_px_norm) else "-"
            z_txt = f" z_err={err_z * 1000:+.1f}mm" if target_z is not None else ""
            put_status(frame, f"[{phase_label}] xy_err={err_txt}{z_txt} iter={i}")
            if _user_abort(rig.show(frame, 1)):
                print(f"[{phase_label}] aborted by user")
                return q

        xy_done = err_px_norm < tol_px
        z_done = target_z is None or abs(err_z) < z_tol_m
        done = {"z_only": z_done, "xy_only": xy_done}.get(exit_mode, xy_done and z_done)
        if done:
            print(f"[{phase_label}] converged at iter {i}: pixel_err={err_px_norm:.2f}px "
                  f"z_err={err_z * 1000:+.2f}mm tip=({tip_xyz[0]:+.4f}, "
                  f"{tip_xyz[1]:+.4f}, {tip_xyz[2]:+.4f})")
            break

        delta_xy = IMAGE_TO_ROBOT_GAIN_MAT @ err_px
        n = float(np.linalg.norm(delta_xy))
        if n > MAX_XY_STEP_M:
            delta_xy = delta_xy / n * MAX_XY_STEP_M
        delta_xyz = np.array([delta_xy[0], delta_xy[1], delta_z], dtype=float)

        q_new = _step_joints(rig.chain, q, delta_xyz, rig.tip_offset, MAX_SERVO_DQ_DEG)
        if q_new is None:
            print(f"[{phase_label}] singular Jacobian, stopping")
            return q
        q = q_new
        if lock_camera_dir:
            q = enforce_camera_dir(rig.chain, q, rig.camera_dir)
        rig.send_joints(q)
        time.sleep(period)
    else:
        print(f"[{phase_label}] max_iters={max_iters} reached without convergence")

    return q


# ---------------------------------------------------------------------------
# Descents (phase 3 variants)
# ---------------------------------------------------------------------------

def descend_to_z(
    rig: Rig, q_init: np.ndarray, target: Target, target_z: float, *,
    descent_speed_mps: float = 0.005,
    loop_hz: float = 15.0,
    max_descent_m: float = 0.20,
    tol_m: float = 0.001,
    max_xy_correction_per_tick_m: float = 0.0005,
) -> np.ndarray:
    """Straight vertical descent to ``target_z`` with a closed-loop XY hold.

    Z steps open-loop at a constant speed. At the start the target's pixel is
    snapshotted as a reference; each tick the drift of the target in the
    image is mapped through the servo gain into a small XY correction, so
    the tip stays over the target while descending. wrist_flex is left free
    so the Jacobian can use it. Used by Eval 1 (hover, no contact).
    """
    period = 1.0 / loop_hz
    q = np.asarray(q_init, dtype=float).copy()
    step_z = descent_speed_mps / loop_hz

    reference_px = None
    for _ in range(30):
        frame = rig.read_frame()
        if frame is not None:
            pt = target.update(frame)
            if pt is not None:
                reference_px = np.array(pt, dtype=float)
                break
    if reference_px is None:
        print("[descend_to_z] no reference target pixel; XY hold disabled")
    else:
        print(f"[descend_to_z] reference pixel = ({reference_px[0]:.0f}, {reference_px[1]:.0f})")

    start_z = float(prim.fk_tip(rig.chain, q, rig.tip_offset)[2])
    print(f"[descend_to_z] z {start_z * 1000:+.1f}mm -> {target_z * 1000:+.1f}mm")

    z_traveled = 0.0
    last_px = reference_px
    while True:
        t0 = time.perf_counter()
        tip_xyz = prim.fk_tip(rig.chain, q, rig.tip_offset)
        err_z = target_z - float(tip_xyz[2])
        if abs(err_z) < tol_m:
            print(f"[descend_to_z] reached z={tip_xyz[2] * 1000:+.2f}mm")
            return q
        if z_traveled > max_descent_m:
            print(f"[descend_to_z] aborted after {z_traveled * 1000:.0f} mm without reaching target")
            return q

        delta_z = float(np.clip(err_z, -step_z, step_z))

        frame = rig.read_frame()
        current_px = target.update(frame) if frame is not None else None
        if current_px is not None:
            last_px = np.array(current_px, dtype=float)

        drift_norm = 0.0
        xy_corr = np.zeros(2)
        if reference_px is not None and last_px is not None:
            drift = last_px - reference_px
            drift_norm = float(np.linalg.norm(drift))
            xy_corr = IMAGE_TO_ROBOT_GAIN_MAT @ drift
            n = float(np.linalg.norm(xy_corr))
            if n > max_xy_correction_per_tick_m:
                xy_corr = xy_corr / n * max_xy_correction_per_tick_m

        q_new = _step_joints(rig.chain, q, np.array([xy_corr[0], xy_corr[1], delta_z]),
                             rig.tip_offset, MAX_DESCENT_DQ_DEG)
        if q_new is None:
            print("[descend_to_z] singular Jacobian, aborting")
            return q
        q = q_new
        rig.send_joints(q)
        z_traveled += abs(delta_z)

        if frame is not None:
            if reference_px is not None:
                draw_cross(frame, reference_px, YELLOW, 30, 3)
            target.draw(frame)
            put_status(frame, f"[descend->{target_z * 1000:+.0f}mm] z={tip_xyz[2] * 1000:+.1f}mm "
                              f"err={err_z * 1000:+.1f}mm drift={drift_norm:.1f}px "
                              f"wf={q[WRIST_FLEX_IDX]:+.1f}")
            rig.show(frame, 1)

        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)


def descend_until_contact(
    rig: Rig, q_init: np.ndarray, *,
    target: Target | None = None,
    descent_speed_mps: float = 0.005,
    max_descent_m: float = 0.05,
    loop_hz: float = 30.0,
    stall_thresh_deg: float = 10.0,
    load_thresh: int = 60,
    contact_motor: str = "shoulder_lift",
    skip_first_ticks: int = 8,
    baseline_n_ticks: int = 10,
) -> tuple[np.ndarray, dict]:
    """Straight vertical descent until contact (phase 3 of a key press).

    XY is frozen: the commanded step is pure -Z and the pseudo-inverse
    distributes it over all five joints (wrist_flex included), so the
    planned tip motion has no lateral component. Contact is declared when
    the ``contact_motor`` load deviates from its in-motion baseline by more
    than ``load_thresh`` for two consecutive ticks, or when the commanded
    and measured joints disagree by more than ``stall_thresh_deg`` (stall).

    Returns ``(q_at_contact, info)`` with ``info["reason"]`` explaining why
    the descent stopped.
    """
    period = 1.0 / loop_hz
    q = np.asarray(q_init, dtype=float).copy()
    step_z = descent_speed_mps / loop_hz

    baseline_buf: list[int] = []
    baseline: int | None = None
    consec_load = consec_stall = 0
    tick = 0
    z_traveled = 0.0

    def _info(reason: str, n: int) -> dict:
        return {"reason": reason, "ticks": n, "z_traveled_mm": z_traveled * 1000}

    while True:
        t0 = time.perf_counter()
        z_traveled += step_z
        if z_traveled > max_descent_m:
            return q, _info(f"max descent {max_descent_m * 1000:.0f} mm reached", tick)

        frame = rig.read_frame()

        q_new = _step_joints(rig.chain, q, np.array([0.0, 0.0, -step_z]),
                             rig.tip_offset, MAX_DESCENT_DQ_DEG)
        if q_new is None:
            return q, _info("Jacobian singular", tick)
        rig.send_joints(q_new)

        actual_q = q_new
        load_now = 0
        try:
            actual_q = rig.read_joints_deg()
            loads = rig.robot.bus.sync_read("Present_Load")
            load_now = int(loads.get(contact_motor, 0))
        except Exception:
            pass
        stall_err = float(np.max(np.abs(q_new - actual_q)))

        if frame is not None:
            phase = "warmup" if tick < skip_first_ticks else ("baseline" if baseline is None else "monitor")
            base_txt = f" base={baseline:+d}" if baseline is not None else ""
            put_status(frame, f"[descent {phase}] z={z_traveled * 1000:.1f}mm "
                              f"load={load_now:+d}{base_txt} stall={stall_err:.1f}")
            if target is not None:
                target.draw(frame)
            rig.show(frame, 1)

        if tick < skip_first_ticks:
            pass
        elif baseline is None:
            baseline_buf.append(load_now)
            if len(baseline_buf) >= baseline_n_ticks:
                baseline = int(statistics.median(baseline_buf))
                print(f"  [descent] in-flight load baseline = {baseline:+d}")
        else:
            delta = abs(load_now - baseline)
            consec_stall = consec_stall + 1 if stall_err > stall_thresh_deg else 0
            if consec_stall >= 2:
                return q_new, _info(f"stall {stall_err:.1f} deg", tick + 1)
            consec_load = consec_load + 1 if delta > load_thresh else 0
            if consec_load >= 2:
                return q_new, _info(f"load |d|={delta} > {load_thresh}", tick + 1)

        q = q_new
        tick += 1
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)
