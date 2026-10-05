"""Open-loop descent to a calibrated fixed depth — fastest press option.

Reads `configs/press_depth.json` (written by
`cybertyping.calibration.press_depth`) to know how far below
`pre_tip_xyz[2]` to descend. No sensing during the descent — risk is
that bad calibration will under- or over-press; payoff is sub-second
per key for the bonus track.

If `configs/press_depth.json` is absent, defaults to 5 mm.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cybertyping import core as motion
from cybertyping.core import ik

_PRESS_DEPTH_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "press_depth.json"
)
_DEFAULT_DEPTH_M = 0.006   # 6mm — see session.py for rationale
_WARNED_NO_CONFIG = False


def _load_depth_m() -> float:
    global _WARNED_NO_CONFIG
    if not _PRESS_DEPTH_PATH.exists():
        if not _WARNED_NO_CONFIG:
            _WARNED_NO_CONFIG = True
            print(
                f"[open_loop] WARNING: {_PRESS_DEPTH_PATH.name} not found; "
                f"defaulting to {_DEFAULT_DEPTH_M*1000:.1f} mm. "
                f"Run --calibrate-press or populate configs/press_depth.json "
                f"to avoid shallow presses."
            )
        return _DEFAULT_DEPTH_M
    with open(_PRESS_DEPTH_PATH) as f:
        d = json.load(f)
    # `calibration/press_depth.py` writes {"offset_m": ..., "per_key": {...}, ...}
    if "offset_m" in d:
        return float(d["offset_m"])
    if not _WARNED_NO_CONFIG:
        _WARNED_NO_CONFIG = True
        print(
            f"[open_loop] WARNING: {_PRESS_DEPTH_PATH.name} has no 'offset_m'; "
            f"defaulting to {_DEFAULT_DEPTH_M*1000:.1f} mm."
        )
    return _DEFAULT_DEPTH_M


def press(
    robot,
    chain,
    pre_tip_xyz,
    q_pre,
    tip_off,
    gripper_pos: float,
    *,
    depth_m: float | None = None,
    max_step_deg: float = 2.2,
):
    """Slew from q_pre to (pre_tip_xyz - depth in -Z). No descent sensing.

    Returns (q_target, descent_qs, info) matching the shape of the
    load_contact press function so the two are swappable.
    """
    d = float(_load_depth_m()) if depth_m is None else float(depth_m)

    target_tip = np.asarray(pre_tip_xyz, dtype=float).copy()
    target_tip[2] -= d

    q_target = ik.solve_tip(chain, target_tip, q_pre, tip_off).q
    descent_qs = motion.slew_segment(q_pre, q_target, max_step_deg)
    motion.stream(robot, descent_qs, gripper_pos=gripper_pos)

    info = {
        "reason": "open_loop_fixed_depth",
        "depth_m": d,
        "ticks": len(descent_qs),
    }
    return q_target, descent_qs, info


__all__ = ["press"]
