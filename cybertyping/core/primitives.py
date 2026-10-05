"""Hardware-validated SO-101 motion primitives.

This is the single source of truth for everything that touches the arm at
the lowest level: the ikpy kinematic chain, forward/inverse kinematics for
the gripper frame and the tool tip, joint-space slewing, 30 Hz streaming,
and load-based contact detection. Every higher-level module (visual servo,
analytic IK, the legacy open-loop pipeline) builds on these functions.

The code was validated on the real robot during the project and is kept
deliberately conservative. Tune the constants below rather than
restructuring the control loops.

Conventions
-----------
* Joint vectors ``q`` are body-motor degrees in ``BODY_MOTOR_ORDER``
  (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll).
* ``fk_origin`` returns the URDF ``gripper_frame_link`` origin;
  ``fk_tip`` applies a tool offset expressed in that local frame.
* All positions are metres in the robot base frame.
"""

import json
import os
import statistics
import time
from collections import deque
from pathlib import Path

import numpy as np
from ikpy.chain import Chain

# ====== ROBOT ======
PORT = os.environ.get("SO101_PORT", "COM5")
ROBOT_ID = os.environ.get("SO101_ROBOT_ID", "team08_follower_arm")
URDF_PATH = str(
    Path(__file__).resolve().parents[2] / "SO101" / "so101_new_calib.urdf"
)
WORLD_SETTING_PATH = Path(__file__).resolve().parents[2] / "world_settings_load.json"

# ====== TIP OFFSET (measured with a caliper on the mounted pen) ======
TOOL_TIP_OFFSET_LOCAL_XYZ = (-0.02635, +0.00821, +0.00090)

# ====== CONTACT SENSING ======
CONTACT_MOTOR = "shoulder_lift"
CONTACT_THRESHOLD = 50         # |Δload| above baseline -> contact signal
CONTACT_N_CONSEC = 2           # this many consecutive ticks above threshold
# Baseline is measured IN-FLIGHT (while the arm is already descending at
# steady speed), NOT at pre-touch stationary. Reason: Present_Load encodes
# the PID's PWM output, which changes sign/magnitude when the motor switches
# from "hold against gravity" to "track a descending setpoint". A static
# baseline ends up huge-Δ away from the in-motion load and triggers false
# contact in the first ticks of every descent.
SKIP_FIRST_TICKS = 6           # discard PID transient ticks before sampling baseline
BASELINE_INFLIGHT_TICKS = 8    # collect this many in-motion ticks for baseline median
DESCENT_SPEED_M_PER_S = 0.010  # 1 cm/s
MAX_DESCENT_M = 0.12           # max travel from pre-touch -> safety stop
LOAD_HARD_LIMIT = 350          # |Δload| above this -> abort instantly (saturation guard)

# ====== SAFETY ======
Z_MIN_M = -0.10                # hard floor in robot Z
LOOP_HZ = 30
MAX_STEP_DEG_PER_TICK_APPROACH = 4.0
MAX_STEP_DEG_PER_TICK_RETREAT  = 4.0
IK_POS_TOL_M = 0.035           # loose: descent runs until contact, not to a fixed z;
                               # +ENFORCE_GRIPPER_DOWN can push to ~20 mm near workspace
                               # edges where wrist_flex (limit ±95°) saturates
MAX_JOINT_FLIP_DEG = 25.0
MAX_EE_JUMP_M = 1.25
MAX_RELATIVE_TARGET_DEG = None
LOCK_WRIST_FLEX = False
LOCKED_JOINTS: list[str] = ["wrist_roll"] + (
    ["wrist_flex"] if LOCK_WRIST_FLEX else []
)

# Force the gripper to point STRAIGHT DOWN during IK.
# Adds an orientation constraint (gripper_frame_link +Z aligned with base -Z).
# IMPORTANT: the constraint consumes one effective DOF (the pitch sum). On
# SO-101 it is exactly satisfiable ONLY IF wrist_flex is free to rotate ->
# set LOCK_WRIST_FLEX = False when enabling this flag, otherwise the IK is
# over-constrained and the residual will grow (the script will likely abort).
# wrist_roll can stay locked: it spins around the gripper's own axis, so it
# does NOT help satisfy either position or +Z direction constraints.
ENFORCE_GRIPPER_DOWN = True
GRIPPER_DOWN_DIRECTION = (0.0, 0.0, -1.0)   # base-frame direction for +Z_local

# ====== LIVE LOAD PLOT ======
PLOT_LIVE_LOAD = False
PLOT_WINDOW_SAMPLES = 600
PLOT_UPDATE_EVERY = 3
PLOT_MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
# ============================

BODY_MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


# ====== CHAIN ======
def build_chain(urdf_path, locked_joints=()):
    tmp = Chain.from_urdf_file(urdf_path)
    movable = [
        getattr(l, "joint_type", None) in ("revolute", "prismatic", "continuous")
        for l in tmp.links
    ]
    movable_idx = [i for i, m in enumerate(movable) if m]
    if len(movable_idx) != len(BODY_MOTOR_ORDER):
        raise SystemExit(
            f"[chain] {len(movable_idx)} movable joints, "
            f"expected {len(BODY_MOTOR_ORDER)}. Check URDF_PATH."
        )
    motor_link_idx = dict(zip(BODY_MOTOR_ORDER, movable_idx))
    mask = list(movable)
    for name in locked_joints:
        if name not in motor_link_idx:
            raise SystemExit(
                f"[chain] locked joint '{name}' not in {BODY_MOTOR_ORDER}"
            )
        mask[motor_link_idx[name]] = False
    chain = Chain.from_urdf_file(urdf_path, active_links_mask=mask)
    chain._motor_link_idx = motor_link_idx
    return chain


def motor_deg_to_full_rad(chain, q_body_deg):
    full = np.zeros(len(chain.links))
    for i, name in enumerate(BODY_MOTOR_ORDER):
        full[chain._motor_link_idx[name]] = np.deg2rad(float(q_body_deg[i]))
    return full


def full_rad_to_body_deg(chain, full_rad):
    return np.array(
        [np.rad2deg(float(full_rad[chain._motor_link_idx[n]])) for n in BODY_MOTOR_ORDER],
        dtype=float,
    )


def fk_T(chain, q_body_deg):
    return np.asarray(
        chain.forward_kinematics(motor_deg_to_full_rad(chain, q_body_deg))
    )


def fk_origin(chain, q_body_deg):
    return fk_T(chain, q_body_deg)[:3, 3]


def fk_tip(chain, q_body_deg, tip_local):
    T = fk_T(chain, q_body_deg)
    return (T @ np.append(np.asarray(tip_local, dtype=float), 1.0))[:3]


def clamp_initial_to_bounds(chain, full_rad):
    out = full_rad.copy()
    for name in BODY_MOTOR_ORDER:
        i = chain._motor_link_idx[name]
        lo, hi = chain.links[i].bounds
        if np.isfinite(lo) and np.isfinite(hi):
            out[i] = np.clip(out[i], lo, hi)
    return out


# ====== IK ======
def solve_ik_origin(chain, target_xyz, q_init_deg, enforce_gripper_down=False):
    """
    IK so gripper_frame_link's origin lands at `target_xyz`.

    If `enforce_gripper_down` is True, also constrain the gripper's +Z axis
    to align with GRIPPER_DOWN_DIRECTION (base frame, default = [0, 0, -1] =
    point straight down). Uses ikpy `orientation_mode="Z"` which only
    constrains the Z axis direction (2 effective rotational DOF -> 1 on
    SO-101 because the arm geometry already constrains the yaw of +Z).
    """
    init_full = clamp_initial_to_bounds(chain, motor_deg_to_full_rad(chain, q_init_deg))
    kwargs = {"target_position": target_xyz, "initial_position": init_full}
    if enforce_gripper_down:
        kwargs["target_orientation"] = np.asarray(GRIPPER_DOWN_DIRECTION, dtype=float)
        kwargs["orientation_mode"] = "Z"
    sol_full = chain.inverse_kinematics(**kwargs)
    q_body = full_rad_to_body_deg(chain, sol_full)
    residual = float(np.linalg.norm(fk_origin(chain, q_body) - target_xyz))
    return q_body, residual


def solve_ik_tip(chain, target_tip_xyz, q_warm_deg, tip_local, max_iter=5,
                 enforce_gripper_down=False):
    target = np.asarray(target_tip_xyz, dtype=float)
    tip = np.asarray(tip_local, dtype=float)
    q = np.asarray(q_warm_deg, dtype=float)
    residual = float("inf")
    for _ in range(max_iter):
        R = fk_T(chain, q)[:3, :3]
        q, _ = solve_ik_origin(
            chain, target - R @ tip, q, enforce_gripper_down=enforce_gripper_down,
        )
        residual = float(np.linalg.norm(fk_tip(chain, q, tip) - target))
        if residual < 0.5 * IK_POS_TOL_M:
            break
    return q, residual


# ====== JOINT-SPACE STREAMING ======
def slew_segment(q_a, q_b, max_step_deg):
    d = q_b - q_a
    n = max(int(np.ceil(np.max(np.abs(d)) / max_step_deg)), 1)
    return [q_a + d * (i / n) for i in range(1, n + 1)]


def slew_through(q_init, waypoints, max_step_deg):
    out = []
    prev = q_init
    for q in waypoints:
        out.extend(slew_segment(prev, q, max_step_deg))
        prev = q
    return out


def stream(robot, q_traj, gripper_pos, plot_state=None):
    """Approach/retreat streaming (no contact check)."""
    period = 1.0 / LOOP_HZ
    for q in q_traj:
        t0 = time.perf_counter()
        action = {f"{n}.pos": float(q[i]) for i, n in enumerate(BODY_MOTOR_ORDER)}
        action["gripper.pos"] = gripper_pos
        robot.send_action(action)
        if plot_state is not None:
            try:
                loads = robot.bus.sync_read("Present_Load")
                plot_state["tick"][0] += 1
                plot_state["t"].append(plot_state["tick"][0])
                for m in PLOT_MOTORS:
                    plot_state["data"][m].append(int(loads.get(m, 0)))
                if plot_state["tick"][0] % PLOT_UPDATE_EVERY == 0:
                    update_live_plot(plot_state)
            except Exception:
                pass
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)


def measure_baseline(robot, n_ticks, plot_state=None):
    """Hold whatever Goal_Position was last commanded; sample CONTACT_MOTOR load."""
    period = 1.0 / LOOP_HZ
    samples = []
    for _ in range(n_ticks):
        t0 = time.perf_counter()
        try:
            loads = robot.bus.sync_read("Present_Load")
            samples.append(int(loads.get(CONTACT_MOTOR, 0)))
            if plot_state is not None:
                plot_state["tick"][0] += 1
                plot_state["t"].append(plot_state["tick"][0])
                for m in PLOT_MOTORS:
                    plot_state["data"][m].append(int(loads.get(m, 0)))
                if plot_state["tick"][0] % PLOT_UPDATE_EVERY == 0:
                    update_live_plot(plot_state)
        except Exception:
            pass
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)
    if not samples:
        raise SystemExit("[abort] could not read any baseline samples")
    return int(statistics.median(samples))


def descent_until_contact(
    chain, robot, pre_tip_xyz, q_pre, tip_off, gripper_pos, plot_state=None
):
    """
    Open-ended Cartesian descent along base -Z from pre_tip_xyz with IN-FLIGHT
    baseline. Phases (all at the same descent speed, no hold):
      1) SKIP_FIRST_TICKS warmup ticks: descend, ignore load (PID transient).
      2) BASELINE_INFLIGHT_TICKS sampling ticks: descend, collect load samples,
         compute median = baseline.
      3) monitor: descend, |load - baseline| compared to CONTACT_THRESHOLD for
         N consecutive ticks -> contact.

    Returns (contact_q, descent_qs, info).
    """
    pre = np.asarray(pre_tip_xyz, dtype=float)
    step_z = DESCENT_SPEED_M_PER_S / LOOP_HZ
    period = 1.0 / LOOP_HZ

    descent_qs = []
    q_warm = np.asarray(q_pre, dtype=float)
    z_cmd = float(pre[2])
    consec = 0
    skipped = 0
    baseline_buf = []
    baseline = None              # set after baseline phase
    contact_q = None
    contact_reason = None
    tick = 0
    peak_delta = 0

    while True:
        t0 = time.perf_counter()
        z_cmd -= step_z

        # Safety backstops (no-contact case)
        if z_cmd < Z_MIN_M:
            contact_reason = f"reached Z_MIN_M = {Z_MIN_M:.3f} without contact"
            break
        if (pre[2] - z_cmd) > MAX_DESCENT_M:
            contact_reason = f"max travel {MAX_DESCENT_M*1000:.0f} mm without contact"
            break

        target = np.array([pre[0], pre[1], z_cmd])
        q, res = solve_ik_tip(
            chain, target, q_warm, tip_off,
            enforce_gripper_down=ENFORCE_GRIPPER_DOWN,
        )
        if res > IK_POS_TOL_M:
            contact_reason = f"IK residual {res*1000:.1f} mm > tol"
            break
        flip = float(np.max(np.abs(q - q_warm)))
        if flip > MAX_JOINT_FLIP_DEG:
            contact_reason = f"joint flip {flip:.1f}°"
            break

        action = {f"{n}.pos": float(q[i]) for i, n in enumerate(BODY_MOTOR_ORDER)}
        action["gripper.pos"] = gripper_pos
        robot.send_action(action)

        try:
            loads = robot.bus.sync_read("Present_Load")
            load_now = int(loads.get(CONTACT_MOTOR, 0))
            if plot_state is not None:
                plot_state["tick"][0] += 1
                plot_state["t"].append(plot_state["tick"][0])
                for m in PLOT_MOTORS:
                    plot_state["data"][m].append(int(loads.get(m, 0)))
                if plot_state["tick"][0] % PLOT_UPDATE_EVERY == 0:
                    update_live_plot(plot_state)
        except Exception:
            load_now = 0

        descent_qs.append(q)

        # Phase 1: warmup — let the PID settle into the descent regime
        if skipped < SKIP_FIRST_TICKS:
            skipped += 1
            phase = "warmup"
            delta = 0
        # Phase 2: collect in-flight baseline samples
        elif baseline is None:
            baseline_buf.append(load_now)
            phase = "baseline"
            delta = 0
            if len(baseline_buf) >= BASELINE_INFLIGHT_TICKS:
                baseline = int(statistics.median(baseline_buf))
                print(f"\n  [base] in-flight baseline ({len(baseline_buf)} samples) "
                      f"= {baseline:+d}", flush=True)
        # Phase 3: monitor for contact
        else:
            delta = abs(load_now - baseline)
            peak_delta = max(peak_delta, delta)
            phase = "monitor"

            # Hard load limit (motor saturation guard) — only after baseline known
            if delta > LOAD_HARD_LIMIT:
                contact_q = q
                contact_reason = f"|Δload|={delta} > LOAD_HARD_LIMIT={LOAD_HARD_LIMIT}"
                break

            if delta > CONTACT_THRESHOLD:
                consec += 1
                if consec >= CONTACT_N_CONSEC:
                    contact_q = q
                    contact_reason = (
                        f"|Δload|={delta} > {CONTACT_THRESHOLD} "
                        f"for {CONTACT_N_CONSEC} ticks"
                    )
                    break
            else:
                consec = 0

        q_warm = q

        # Live one-line status
        print(
            f"  [{phase:8s}] tick {tick:3d}  z_cmd={z_cmd*1000:+6.1f}mm  "
            f"{CONTACT_MOTOR} load={load_now:+5d}  Δ={delta:+4d}  "
            f"(peak {peak_delta})   ",
            end="\r", flush=True,
        )

        tick += 1
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)
    print()
    if contact_q is None:
        contact_q = descent_qs[-1] if descent_qs else q_pre
    info = {
        "ticks": tick + 1,
        "z_traveled_mm": (pre[2] - z_cmd) * 1000,
        "peak_delta": peak_delta,
        "reason": contact_reason or "contact",
    }
    return contact_q, descent_qs, info


# ====== LIVE LOAD PLOT ======
def setup_live_plot():
    import matplotlib.pyplot as plt
    plt.ion()
    fig, axes = plt.subplots(len(PLOT_MOTORS), 1, figsize=(9, 8), sharex=True)
    if len(PLOT_MOTORS) == 1:
        axes = [axes]
    lines = []
    data = {m: deque(maxlen=PLOT_WINDOW_SAMPLES) for m in PLOT_MOTORS}
    times = deque(maxlen=PLOT_WINDOW_SAMPLES)
    for ax, m in zip(axes, PLOT_MOTORS):
        (line,) = ax.plot([], [], lw=1.0)
        ax.set_ylabel(m, fontsize=8)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        lines.append(line)
    axes[-1].set_xlabel("tick")
    fig.suptitle("Present_Load (live)")
    fig.tight_layout()
    plt.show(block=False)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return {
        "fig": fig, "axes": axes, "lines": lines,
        "t": times, "data": data, "tick": [0],
    }


def update_live_plot(plot_state):
    xs = list(plot_state["t"])
    for m, line in zip(PLOT_MOTORS, plot_state["lines"]):
        line.set_data(xs, list(plot_state["data"][m]))
    for ax in plot_state["axes"]:
        ax.relim()
        ax.autoscale_view()
    plot_state["fig"].canvas.draw_idle()
    plot_state["fig"].canvas.flush_events()


# ====== GEOMETRY ======
def point_in_convex_quad(xy, corners):
    p = np.asarray(xy, dtype=float)
    sign = 0
    n = len(corners)
    for i in range(n):
        edge = corners[(i + 1) % n] - corners[i]
        to_p = p - corners[i]
        c = edge[0] * to_p[1] - edge[1] * to_p[0]
        if abs(c) < 1e-12:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def bbox_quad(bb):
    return np.array([
        [bb["top_left"]["x"],     bb["top_left"]["y"]],
        [bb["top_right"]["x"],    bb["top_right"]["y"]],
        [bb["bottom_right"]["x"], bb["bottom_right"]["y"]],
        [bb["bottom_left"]["x"],  bb["bottom_left"]["y"]],
    ], dtype=float)


# ====== JSON I/O ======
def load_world(path):
    with open(path, "r") as f:
        return json.load(f)


def save_world(path, world):
    with open(path, "w") as f:
        json.dump(world, f, indent=2)
