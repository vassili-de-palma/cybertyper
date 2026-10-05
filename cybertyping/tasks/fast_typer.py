"""Pre-IK keymap, sentence trajectory planner, speed presets.

prepare_keymap() solves IK once for every key using a warm-started chain,
so per-press cost is just slew + dwell + lift. plan_sentence_trajectory()
stitches per-key trajectories with an inter-key lift that adapts to the
distance between consecutive keys.

Presets in PRESETS: SAFE (contact-sense), FAST (open-loop + verify+retry),
BLITZ (open-loop, no verify), ADAPTIVE (per-key behaviour based on
success history; see ARCHITECTURE.md for when to use which).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np



@dataclass(frozen=True)
class SpeedPreset:
    name: str
    max_step_deg_per_tick: float
    approach_height_m: float
    inter_key_lift_adjacent_m: float
    inter_key_lift_near_m: float
    inter_key_lift_far_m: float
    near_distance_m: float           # robot-XY distance below which "near"
    far_distance_m: float            # below this stays in same-row; above = full lift
    use_contact_sense: bool          # True = use descent_until_contact for every press
    use_visual_servo: bool           # True = one tip-detector correction per press
    use_keystroke_verify: bool       # True = pynput verify each press
    allow_retry: bool                # True = retry once if a press doesn't register
    settle_dwell_s: float            # tiny pause between segments (PID convergence)
    # When True, the policy uses per-key state on PreparedKey to dynamically
    # adjust lift / dwell / depth as we go. Proven keys (3+ first-try
    # successes) get BLITZ-style speed; problematic keys (2+ misses) get
    # SAFE-style depth + extra dwell.
    adaptive: bool = False
    # Retry depth: how much deeper than the calibrated press depth to descend
    # on a missed press. Successful retries persist this as the new baseline.
    retry_extra_depth_m: float = 0.002


# NOTE: Eval keyboard is not plugged into the laptop running this script,
# so pynput sees zero keystrokes. `use_keystroke_verify` and `allow_retry`
# are forced off for every preset -- otherwise every "verify" returns
# False, every key triggers a +2mm retry, and the time budget evaporates.
# Reliability has to come from contact-sense descent + calibrated depth
# instead.

PRESET_SAFE = SpeedPreset(
    name="SAFE",
    max_step_deg_per_tick=1.6,
    approach_height_m=0.050,
    inter_key_lift_adjacent_m=0.050,    # full retreat between keys
    inter_key_lift_near_m=0.050,
    inter_key_lift_far_m=0.050,
    near_distance_m=0.030,
    far_distance_m=0.070,
    use_contact_sense=True,
    use_visual_servo=True,
    use_keystroke_verify=False,
    allow_retry=False,
    settle_dwell_s=0.05,
)

PRESET_FAST = SpeedPreset(
    name="FAST",
    max_step_deg_per_tick=2.2,
    approach_height_m=0.035,
    inter_key_lift_adjacent_m=0.005,    # 5mm
    inter_key_lift_near_m=0.010,        # 1cm
    inter_key_lift_far_m=0.030,         # 3cm
    near_distance_m=0.025,
    far_distance_m=0.070,
    use_contact_sense=False,
    use_visual_servo=False,
    use_keystroke_verify=False,
    allow_retry=False,
    settle_dwell_s=0.02,
)

PRESET_BLITZ = SpeedPreset(
    name="BLITZ",
    max_step_deg_per_tick=3.5,
    approach_height_m=0.020,
    inter_key_lift_adjacent_m=0.005,    # 5mm: clear typical 6-8mm keycaps;
                                        # 3mm grazed adjacent caps during
                                        # joint-space slew between presses
    inter_key_lift_near_m=0.008,        # 8mm
    inter_key_lift_far_m=0.020,         # 2cm
    near_distance_m=0.030,
    far_distance_m=0.090,
    use_contact_sense=False,
    use_visual_servo=False,
    use_keystroke_verify=False,         # eval keyboard not plugged into our laptop
    allow_retry=False,                  # ditto: no signal to gate retry on
    settle_dwell_s=0.0,
)

# ADAPTIVE: started life as a FAST + per-key online-learning hybrid that
# back-off depth/dwell on misses observed via pynput. Since the eval
# keyboard is not plugged into our laptop, the learning signal is dead;
# ADAPTIVE degrades to "FAST with extra bookkeeping". Kept for parity
# with the older eval scripts, but not recommended for eval day.
PRESET_ADAPTIVE = SpeedPreset(
    name="ADAPTIVE",
    max_step_deg_per_tick=2.6,           # between FAST (2.2) and BLITZ (3.5)
    approach_height_m=0.030,
    inter_key_lift_adjacent_m=0.005,
    inter_key_lift_near_m=0.010,
    inter_key_lift_far_m=0.025,
    near_distance_m=0.025,
    far_distance_m=0.080,
    use_contact_sense=False,
    use_visual_servo=False,
    use_keystroke_verify=False,
    allow_retry=False,
    settle_dwell_s=0.015,
    adaptive=True,
    retry_extra_depth_m=0.002,
)

PRESETS = {p.name: p for p in [PRESET_SAFE, PRESET_FAST, PRESET_BLITZ, PRESET_ADAPTIVE]}


# Per-key effective parameter overrides that the ADAPTIVE policy applies.
# Confidence buckets:
#   PROVEN     (3+ successes, 0 misses)        -> BLITZ-style lift/dwell
#   NEUTRAL    (default, no history)            -> preset defaults
#   PROBLEMATIC (2+ misses or <50% success)     -> SAFE-ish: deeper press + extra dwell

@dataclass(frozen=True)
class _EffectiveParams:
    """Per-key resolved parameters used by the streaming loop."""
    press_offset_m: float            # extra Z below calibrated press_z
    press_dwell_ticks: int           # how many ticks to hold at q_press
    lift_scale: float                # multiply the preset's inter-key lift by this
    max_step_deg_per_tick: float


def _resolve_per_key(preset: SpeedPreset, key: "PreparedKey") -> _EffectiveParams:
    """Pick effective per-key parameters for the ADAPTIVE policy.

    For non-adaptive presets, returns the preset defaults unchanged."""
    if not preset.adaptive:
        return _EffectiveParams(
            press_offset_m=0.0,
            press_dwell_ticks=2,
            lift_scale=1.0,
            max_step_deg_per_tick=preset.max_step_deg_per_tick,
        )
    if key.is_proven:
        # Collapse to BLITZ-style on this specific key.
        return _EffectiveParams(
            press_offset_m=key.learned_press_offset_m,
            press_dwell_ticks=1,
            lift_scale=0.6,
            max_step_deg_per_tick=PRESET_BLITZ.max_step_deg_per_tick,
        )
    if key.is_problematic:
        # Be conservative: deeper press, longer dwell, slower slew.
        # Use the learned offset if any, plus an extra 1mm margin.
        return _EffectiveParams(
            press_offset_m=key.learned_press_offset_m + 0.001,
            press_dwell_ticks=4,
            lift_scale=1.2,
            max_step_deg_per_tick=PRESET_SAFE.max_step_deg_per_tick,
        )
    # Neutral: preset defaults + any learned per-key offset.
    return _EffectiveParams(
        press_offset_m=key.learned_press_offset_m,
        press_dwell_ticks=2,
        lift_scale=1.0,
        max_step_deg_per_tick=preset.max_step_deg_per_tick,
    )


def preset_from_env(default: str = "FAST") -> SpeedPreset:
    name = os.environ.get("CYBERTYPING_PRESET", default).upper()
    if name not in PRESETS:
        raise ValueError(f"Unknown CYBERTYPING_PRESET={name!r}; "
                         f"valid: {list(PRESETS)}")
    return PRESETS[name]



@dataclass
class PreparedKey:
    label: str
    robot_xy: tuple[float, float]
    plane_z: float
    press_z: float                   # plane_z + press_offset_m
    q_pre: np.ndarray                # IK at (rx, ry, plane_z + APPROACH_HEIGHT_M)
    q_press: np.ndarray              # IK at (rx, ry, press_z)
    # Inter-key lift IK results, keyed by lift height in millimeters.
    # Filled on demand by `prepare_keymap` based on the preset's lift set.
    q_lifts: dict[int, np.ndarray] = field(default_factory=dict)
    # IK residuals for diagnostics
    res_pre_m: float = 0.0
    res_press_m: float = 0.0

    # ----- Adaptive state (updated online by ADAPTIVE preset) -----
    # success_count: number of times pressing this key registered first-try
    # miss_count:    number of times the first press did NOT register
    # learned_press_offset_m: extra depth beyond the global press_offset that
    #     this specific key needs to reliably register. Bounded to
    #     [LEARNED_OFFSET_MIN_M, LEARNED_OFFSET_MAX_M] to prevent runaway.
    # last_correction_m: optional XY correction we learned from visual servo
    #     (currently unused but available for future learning).
    success_count: int = 0
    miss_count: int = 0
    learned_press_offset_m: float = 0.0
    last_correction_xy_m: tuple[float, float] = (0.0, 0.0)

    @property
    def confidence(self) -> float:
        """Fraction of attempts that succeeded first-try. Returns 0.5
        (neutral) if we've never tried this key."""
        n = self.success_count + self.miss_count
        if n == 0:
            return 0.5
        return self.success_count / n

    @property
    def is_proven(self) -> bool:
        """Heuristic: 3+ successful presses with 100% first-try registers
        means we trust this key enough to use BLITZ-style behavior."""
        return self.success_count >= 3 and self.miss_count == 0

    @property
    def is_problematic(self) -> bool:
        """Conversely: 2+ misses or <50% success rate -> back off."""
        n = self.success_count + self.miss_count
        return self.miss_count >= 2 or (n >= 3 and self.confidence < 0.5)


# Bounds on online-learned press depth: never deeper than +4mm beyond the
# calibrated global offset (sanity guard against runaway from spurious misses).
LEARNED_OFFSET_MIN_M = -0.001       # 1 mm shallower
LEARNED_OFFSET_MAX_M = +0.004       # 4 mm deeper


@dataclass
class PreparedKeymap:
    keys: dict[str, PreparedKey]
    press_offset_m: float
    preset: SpeedPreset
    q_scan: np.ndarray | None = None   # joint config at scan pose (return-to-home)

    def __contains__(self, label: str) -> bool:
        return label in self.keys

    def __getitem__(self, label: str) -> PreparedKey:
        return self.keys[label]


def _solve_ik_safe(chain, tip_xyz, q_warm, tip_off):
    """Thin shim that returns (q_ndarray, residual) for callers in this
    module that haven't been migrated to IKResult yet. New code should
    call `core.ik.solve_tip(...)` directly.
    """
    from cybertyping import core
    res = core.ik.solve_tip(chain, tip_xyz, q_warm, tip_off)
    return res.q, res.residual


def prepare_keymap(
    chain, keymap, press_offset_m: float,
    preset: SpeedPreset, q_warm_start: np.ndarray | None = None,
    target_labels: Iterable[str] | None = None,
) -> PreparedKeymap:
    """Solve IK for every key in the keymap, plus the inter-key lift Zs.

    Args:
        chain:           ikpy chain.
        keymap:          perception.homography.KeyMap (with robot_xy + robot_z).
        press_offset_m:  from press-depth calibration.
        preset:          which speed preset (we precompute lifts only for the
                         heights this preset will use).
        q_warm_start:    starting joint config. If None, defaults to zeros.
        target_labels:   restrict to these labels (default: every key with robot_xy).
    """
    from cybertyping import core

    tip_off = np.asarray(core.TOOL_TIP_OFFSET_LOCAL_XYZ, dtype=float)
    q_warm = (np.asarray(q_warm_start, dtype=float)
              if q_warm_start is not None
              else np.zeros(len(core.BODY_MOTOR_ORDER), dtype=float))
    q_chain = q_warm.copy()

    # Order targets by canonical xy proximity so the warm-start chain stays
    # smooth and each IK converges quickly. Iterate row by row, left to right.
    if target_labels is None:
        target_labels = [l for l, e in keymap.entries.items()
                         if e.robot_xy is not None]
    target_labels = list(target_labels)
    # canonical sort: (y row, then x within row)
    def _canon_xy(label):
        e = keymap.entries[label]
        return (e.canonical_xy[1], e.canonical_xy[0])
    target_labels.sort(key=_canon_xy)

    keys: dict[str, PreparedKey] = {}
    lift_heights_m = sorted({
        preset.inter_key_lift_adjacent_m,
        preset.inter_key_lift_near_m,
        preset.inter_key_lift_far_m,
        preset.approach_height_m,
    })

    for label in target_labels:
        entry = keymap.entries[label]
        if entry.robot_xy is None:
            continue
        rx, ry = entry.robot_xy
        plane_z = float(entry.robot_z)
        press_z = plane_z + press_offset_m

        q_pre, res_pre = _solve_ik_safe(
            chain, np.array([rx, ry, plane_z + preset.approach_height_m]),
            q_chain, tip_off,
        )
        q_press, res_press = _solve_ik_safe(
            chain, np.array([rx, ry, press_z]), q_pre, tip_off,
        )

        q_lifts: dict[int, np.ndarray] = {}
        for lift_m in lift_heights_m:
            q_lift, _ = _solve_ik_safe(
                chain, np.array([rx, ry, plane_z + lift_m]),
                q_pre, tip_off,
            )
            q_lifts[int(round(lift_m * 1000))] = q_lift

        keys[label] = PreparedKey(
            label=label, robot_xy=(rx, ry),
            plane_z=plane_z, press_z=press_z,
            q_pre=q_pre, q_press=q_press,
            q_lifts=q_lifts,
            res_pre_m=res_pre, res_press_m=res_press,
        )
        # warm-start next IK from this key's pre pose
        q_chain = q_pre

    return PreparedKeymap(
        keys=keys, press_offset_m=press_offset_m, preset=preset,
        q_scan=q_warm,
    )



def _inter_key_lift_m(prev: PreparedKey, curr: PreparedKey,
                      preset: SpeedPreset) -> float:
    """Decide how high to lift between two consecutive keys based on
    their robot-XY separation."""
    dx = curr.robot_xy[0] - prev.robot_xy[0]
    dy = curr.robot_xy[1] - prev.robot_xy[1]
    d = float(np.hypot(dx, dy))
    if d <= preset.near_distance_m:
        return preset.inter_key_lift_adjacent_m
    if d <= preset.far_distance_m:
        return preset.inter_key_lift_near_m
    return preset.inter_key_lift_far_m


def plan_sentence_trajectory(
    prepared: PreparedKeymap,
    sentence_labels: list[str],
    start_q: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Build a single joint-space trajectory that types `sentence_labels`.

    For each key:
        1. Slew from current pose to q_pre.
        2. Slew q_pre -> q_press   (descent).
        3. Tiny dwell at q_press   (so the keypress registers).
        4. Slew q_press -> q_lift  (the inter-key lift -- much smaller
           than q_pre when next key is near).
        5. Slew q_lift -> next q_pre.

    Returns the full sequence of joint configs to stream at LOOP_HZ.
    """
    from cybertyping import core

    preset = prepared.preset
    traj: list[np.ndarray] = []
    if not sentence_labels:
        return traj

    q_cur = (start_q if start_q is not None
             else prepared.q_scan if prepared.q_scan is not None
             else prepared.keys[sentence_labels[0]].q_pre.copy())

    # Initial slew to the first key's pre-touch.
    first = prepared.keys[sentence_labels[0]]
    traj.extend(core.slew_segment(q_cur, first.q_pre,
                                  preset.max_step_deg_per_tick))
    if preset.settle_dwell_s > 0:
        traj.extend([first.q_pre] * max(1, int(preset.settle_dwell_s * 30)))

    for i, label in enumerate(sentence_labels):
        if label not in prepared.keys:
            continue
        key = prepared.keys[label]
        # descend pre -> press
        traj.extend(core.slew_segment(traj[-1], key.q_press,
                                      preset.max_step_deg_per_tick))
        # tiny dwell at press to let the keystroke register
        traj.extend([key.q_press] * 2)
        # lift back to pre OR to the adaptive inter-key lift if next exists
        next_label = (sentence_labels[i + 1] if i + 1 < len(sentence_labels)
                      else None)
        if next_label is None or next_label not in prepared.keys:
            # last key: lift back to full pre-touch
            traj.extend(core.slew_segment(key.q_press, key.q_pre,
                                          preset.max_step_deg_per_tick))
            continue
        nxt = prepared.keys[next_label]
        lift_m = _inter_key_lift_m(key, nxt, preset)
        lift_mm = int(round(lift_m * 1000))
        q_lift = key.q_lifts.get(lift_mm, key.q_pre)
        # press -> lift over current key
        traj.extend(core.slew_segment(key.q_press, q_lift,
                                      preset.max_step_deg_per_tick))
        # lift -> next pre-touch (lateral motion at lift height)
        # We pick the lower of (this key's lift over its xy) and (next key's
        # equivalent lift over ITS xy) so the lateral motion happens at the
        # smaller of the two lift heights. Practically we just slew straight
        # to nxt.q_pre and let the joint-space interpolation handle it.
        traj.extend(core.slew_segment(q_lift, nxt.q_pre,
                                      preset.max_step_deg_per_tick))
        if preset.settle_dwell_s > 0:
            traj.extend([nxt.q_pre] * max(1, int(preset.settle_dwell_s * 30)))

    return traj



def type_sentence(
    robot, chain, prepared: PreparedKeymap,
    sentence_labels: list[str],
    keystroke_listener=None,
    log=None,
    verify_mode: str = "stream_then_verify",
) -> tuple[str, list[bool]]:
    """Stream the trajectory for `sentence_labels`, optionally verify each
    press via the keystroke listener, optionally retry misses.

    verify_mode:
      "off"               -- open-loop blast; no verification.
      "per_key"          , stop after each key, check listener, retry if miss.
                             Reliable but slow because motion never overlaps
                             verification.
      "stream_then_verify" -- (default) stream the whole sentence in one
                             blast, snapshot listener timestamps as we go,
                             then check at the end which keys registered.
                             For any that didn't, plan a fixup trajectory
                             that re-presses just the missed keys.
                             Best of both worlds for FAST/BLITZ.

    Returns:
        (typed_string, per_key_verified_bool_list)
    """
    from cybertyping import core
    preset = prepared.preset

    obs = robot.get_observation()
    q_now = np.array([float(obs[f"{n}.pos"]) for n in core.BODY_MOTOR_ORDER])
    gripper_pos = float(obs.get("gripper.pos", 0.0))

    has_listener = (preset.use_keystroke_verify
                    and keystroke_listener is not None
                    and getattr(keystroke_listener, "available", False))

    # ----- "off": pure open-loop blast -----
    if not has_listener or verify_mode == "off":
        traj = plan_sentence_trajectory(prepared, sentence_labels, start_q=q_now)
        if traj:
            core.stream(robot, traj, gripper_pos)
        typed = [_label_to_char(l) for l in sentence_labels if l in prepared.keys]
        verified = [True] * len(typed)
        if log is not None:
            log.add(event="fast_blast", n_keys=len(sentence_labels),
                    n_ticks=len(traj))
        return "".join(typed), verified

    # ----- "per_key": slow but safe (legacy SAFE-preset path) -----
    if verify_mode == "per_key":
        typed = []
        verified = []
        for label in sentence_labels:
            if label not in prepared.keys:
                continue
            key = prepared.keys[label]
            press_window_start = time.time()
            key_traj = _plan_one_key(prepared, q_now, label, preset)
            core.stream(robot, key_traj, gripper_pos)
            q_now = key_traj[-1].copy() if key_traj else q_now
            ok = keystroke_listener.was_received(
                label, press_window_start, window_s=1.5,
            )
            # Record first-try result
            update_key_after_press(key, registered=ok, was_retry=False)
            if not ok and preset.allow_retry:
                retry_traj, extra_depth = _plan_one_key_retry(
                    prepared, q_now, label, preset, chain=chain)
                t_retry = time.time()
                core.stream(robot, retry_traj, gripper_pos)
                q_now = retry_traj[-1].copy() if retry_traj else q_now
                ok = keystroke_listener.was_received(
                    label, t_retry, window_s=2.0,
                )
                update_key_after_press(key, registered=ok, was_retry=True,
                                       retry_extra_depth_m=extra_depth)
            typed.append(_label_to_char(label))
            verified.append(ok)
            if log is not None:
                log.add(event="fast_press", key=label, verified=ok,
                        learned_offset_mm=key.learned_press_offset_m * 1000)
        return "".join(typed), verified

    # ----- "stream_then_verify": the new default -----
    # We need to know APPROXIMATELY when each key was pressed in real time so
    # we can correlate listener events to expected presses. Plan per-key
    # trajectories and accumulate them, recording the expected press time
    # per key, then stream the whole concatenated trajectory.
    typed_buffer: list[tuple[str, float]] = []   # (label, expected_press_time_s)
    full_traj: list[np.ndarray] = []

    # Initial slew to first key's pre-touch
    first = None
    for label in sentence_labels:
        if label in prepared.keys:
            first = prepared.keys[label]
            break
    if first is None:
        return "", []
    full_traj.extend(core.slew_segment(q_now, first.q_pre,
                                        preset.max_step_deg_per_tick))

    for i, label in enumerate(sentence_labels):
        if label not in prepared.keys:
            continue
        key = prepared.keys[label]
        # descend: pre -> press
        full_traj.extend(core.slew_segment(full_traj[-1], key.q_press,
                                           preset.max_step_deg_per_tick))
        # Expected press time = approx halfway through the press dwell.
        expected_press_idx = len(full_traj)
        typed_buffer.append((label, expected_press_idx))
        # dwell
        full_traj.extend([key.q_press] * 2)
        # next-key lift planning
        next_label = next((l for l in sentence_labels[i + 1:]
                           if l in prepared.keys), None)
        if next_label is None:
            full_traj.extend(core.slew_segment(key.q_press, key.q_pre,
                                               preset.max_step_deg_per_tick))
        else:
            nxt = prepared.keys[next_label]
            lift_m = _inter_key_lift_m(key, nxt, preset)
            q_lift = key.q_lifts.get(int(round(lift_m * 1000)), key.q_pre)
            full_traj.extend(core.slew_segment(key.q_press, q_lift,
                                                preset.max_step_deg_per_tick))
            full_traj.extend(core.slew_segment(q_lift, nxt.q_pre,
                                                preset.max_step_deg_per_tick))

    # Convert expected_press_idx -> expected_press_time. We know LOOP_HZ=30.
    LOOP_HZ = 30
    t_blast_start = time.time()
    if log is not None:
        log.add(event="blast_start", n_keys=len(typed_buffer),
                n_ticks=len(full_traj))
    core.stream(robot, full_traj, gripper_pos)

    # Verify each press against the listener using expected timestamps.
    verified: list[bool] = []
    missed: list[tuple[str, int]] = []
    for i, (label, idx) in enumerate(typed_buffer):
        expected_t = t_blast_start + idx / LOOP_HZ
        # The listener clock may be slightly off; allow generous window.
        ok = keystroke_listener.was_received(
            label, since_t=expected_t - 0.5, window_s=1.5,
        )
        verified.append(ok)
        if not ok:
            missed.append((label, i))
        if log is not None:
            log.add(event="blast_verify", key=label, verified=ok)

    # Update per-key state for the first-try results we got from the blast.
    for (label, _), was_ok in zip(typed_buffer, verified):
        if label in prepared.keys:
            update_key_after_press(prepared.keys[label],
                                   registered=was_ok, was_retry=False)

    # ----- Fixup pass for missed presses -----
    if missed and preset.allow_retry:
        print(f"  [fixup] re-pressing {len(missed)} missed keys: "
              f"{[m[0] for m in missed]}")
        obs = robot.get_observation()
        q_now = np.array([float(obs[f"{n}.pos"]) for n in core.BODY_MOTOR_ORDER])
        for label, idx in missed:
            key = prepared.keys[label]
            t_retry = time.time()
            retry_traj, extra_depth = _plan_one_key_retry(
                prepared, q_now, label, preset, chain=chain)
            core.stream(robot, retry_traj, gripper_pos)
            q_now = retry_traj[-1].copy() if retry_traj else q_now
            ok = keystroke_listener.was_received(
                label, since_t=t_retry, window_s=2.0,
            )
            verified[idx] = ok
            update_key_after_press(key, registered=ok, was_retry=True,
                                   retry_extra_depth_m=extra_depth)
            if log is not None:
                log.add(event="fixup_press", key=label, verified=ok,
                        learned_offset_mm=key.learned_press_offset_m * 1000)

    typed = [_label_to_char(l) for l, _ in typed_buffer]
    return "".join(typed), verified


def _label_to_char(label: str) -> str:
    if label == "space": return " "
    if label == "enter": return "\n"
    return label


def _plan_one_key(prepared: PreparedKeymap, q_start: np.ndarray, label: str,
                  preset: SpeedPreset) -> list[np.ndarray]:
    """Trajectory for one press: q_start -> q_pre -> q_press -> q_pre."""
    from cybertyping import core
    key = prepared.keys[label]
    traj: list[np.ndarray] = []
    traj.extend(core.slew_segment(q_start, key.q_pre,
                                  preset.max_step_deg_per_tick))
    traj.extend(core.slew_segment(key.q_pre, key.q_press,
                                  preset.max_step_deg_per_tick))
    traj.extend([key.q_press] * 2)
    traj.extend(core.slew_segment(key.q_press, key.q_pre,
                                  preset.max_step_deg_per_tick))
    return traj


def _plan_one_key_retry(prepared, q_start, label, preset,
                        chain=None) -> tuple[list[np.ndarray], float]:
    """Retry trajectory: press DEEPER than the calibrated press_z.

    Returns (trajectory, attempted_extra_depth_m). If `chain` is provided
    we IK the deeper press pose; otherwise we fall back to dwell-longer
    at the calibrated depth (less effective but safer when chain unavailable).
    """
    from cybertyping import core
    key = prepared.keys[label]
    extra_depth = float(preset.retry_extra_depth_m)
    # Bound to LEARNED_OFFSET_MAX_M including any already-learned offset.
    total_offset = key.learned_press_offset_m + extra_depth
    total_offset = max(LEARNED_OFFSET_MIN_M,
                       min(LEARNED_OFFSET_MAX_M, total_offset))
    extra_depth_actual = total_offset - key.learned_press_offset_m

    if chain is not None and extra_depth_actual > 1e-5:
        # Solve IK for the deeper press pose
        tip_off = np.asarray(core.TOOL_TIP_OFFSET_LOCAL_XYZ, dtype=float)
        deeper_xyz = np.array([key.robot_xy[0], key.robot_xy[1],
                               key.press_z - extra_depth_actual])
        q_deeper, _ = _solve_ik_safe(chain, deeper_xyz, key.q_press, tip_off)
    else:
        q_deeper = key.q_press

    traj: list[np.ndarray] = []
    traj.extend(core.slew_segment(q_start, q_deeper,
                                  preset.max_step_deg_per_tick))
    traj.extend([q_deeper] * 4)      # longer dwell on retry
    traj.extend(core.slew_segment(q_deeper, key.q_pre,
                                  preset.max_step_deg_per_tick))
    return traj, extra_depth_actual



def update_key_after_press(key: PreparedKey, *,
                           registered: bool,
                           was_retry: bool = False,
                           retry_extra_depth_m: float = 0.0) -> None:
    """Mutate the PreparedKey's adaptive state after observing a press.

    Args:
        registered:          True if the keystroke listener saw the press.
        was_retry:           True if this was the retry (not the first try).
        retry_extra_depth_m: how much deeper than the calibrated press_z the
                             retry descended (0 if not a retry).

    Update rules:
      First-try success:    success_count += 1
      First-try miss:       miss_count += 1
      Retry success:        learned_press_offset_m += retry_extra_depth_m,
                            bounded to [LEARNED_OFFSET_MIN_M, LEARNED_OFFSET_MAX_M].
                            (We DON'T bump success_count -- the retry succeeded
                            but the first try missed; the first try counts as miss.)
      Retry miss:           (already counted as miss on first try). Bump
                            learned_press_offset_m by half the retry depth
                            (we know we're definitely short).
    """
    if not was_retry:
        if registered:
            key.success_count += 1
        else:
            key.miss_count += 1
        return
    # was_retry == True
    if registered:
        # The deeper press worked. Persist the learned offset so we start
        # at this depth next time we press this key.
        new_offset = key.learned_press_offset_m + retry_extra_depth_m
        key.learned_press_offset_m = max(
            LEARNED_OFFSET_MIN_M, min(LEARNED_OFFSET_MAX_M, new_offset))
    else:
        # Even the deeper press didn't register. Be conservative: bump
        # learned offset by half of the retry depth so we're closer next time
        # without overshooting based on one data point.
        new_offset = key.learned_press_offset_m + retry_extra_depth_m * 0.5
        key.learned_press_offset_m = max(
            LEARNED_OFFSET_MIN_M, min(LEARNED_OFFSET_MAX_M, new_offset))



def estimate_sentence_time_s(prepared: PreparedKeymap,
                             sentence_labels: list[str],
                             loop_hz: int = 30) -> float:
    """How long will this sentence take to stream? Useful for the bonus
    track planning (and to verify we fit inside the per-sentence budget
    before going live)."""
    traj = plan_sentence_trajectory(prepared, sentence_labels)
    return len(traj) / loop_hz



ADAPTIVE_STATE_SCHEMA_VERSION = 2
# v1: no version field, no press_offset fingerprint -> learned offsets
#     could be restored on top of a keymap built with a different base
#     press_offset_m, double-counting the depth.
# v2: adds "version" and uses press_offset_m as a fingerprint; restore
#     refuses to apply if the base offset has changed.

# Tolerance below which two press_offset_m values are considered "same"
# (i.e. came from the same keyboard / calibration regime).
_PRESS_OFFSET_MATCH_TOL_M = 5e-4   # 0.5 mm


def serialize_adaptive_state(prepared: PreparedKeymap) -> dict:
    """Pull just the adaptive-learning state from a PreparedKeymap.

    The IK fields aren't worth persisting (they're cheap to recompute and
    depend on the URDF + scan pose), but the learned offsets and per-key
    success/miss counters ARE precious, they encode the per-key
    calibration we won during rehearsal.
    """
    return {
        "version": ADAPTIVE_STATE_SCHEMA_VERSION,
        "preset_name": prepared.preset.name,
        "press_offset_m": prepared.press_offset_m,
        "per_key": {
            label: {
                "learned_press_offset_m": k.learned_press_offset_m,
                "success_count": k.success_count,
                "miss_count": k.miss_count,
                "last_correction_xy_m": list(k.last_correction_xy_m),
            }
            for label, k in prepared.keys.items()
        },
    }


def restore_adaptive_state(prepared: PreparedKeymap, state: dict) -> int:
    """Apply previously-saved per-key state back onto a fresh PreparedKeymap.

    Returns the number of keys whose state was restored. Refuses to
    restore (returns 0) if the schema version doesn't match or if the
    base `press_offset_m` differs by more than the match tolerance --
    learned offsets are deltas on top of the base depth and stop being
    meaningful when the base changes.

    Keys present in `state["per_key"]` but absent from the current
    keymap are silently skipped.
    """
    ver = state.get("version")
    if ver != ADAPTIVE_STATE_SCHEMA_VERSION:
        print(f"[adaptive] state version {ver!r} != "
              f"{ADAPTIVE_STATE_SCHEMA_VERSION}; refusing to restore.")
        return 0

    saved_off = state.get("press_offset_m")
    cur_off = prepared.press_offset_m
    if saved_off is None or abs(float(saved_off) - float(cur_off)) > _PRESS_OFFSET_MATCH_TOL_M:
        print(f"[adaptive] base press_offset_m changed "
              f"({saved_off} -> {cur_off}); refusing to restore "
              f"(would double-count depth).")
        return 0

    n = 0
    per_key = state.get("per_key", {})
    for label, fields in per_key.items():
        if label not in prepared.keys:
            continue
        k = prepared.keys[label]
        k.learned_press_offset_m = float(fields.get("learned_press_offset_m", 0.0))
        k.success_count = int(fields.get("success_count", 0))
        k.miss_count = int(fields.get("miss_count", 0))
        cxy = fields.get("last_correction_xy_m", (0.0, 0.0))
        k.last_correction_xy_m = (float(cxy[0]), float(cxy[1]))
        n += 1
    return n
