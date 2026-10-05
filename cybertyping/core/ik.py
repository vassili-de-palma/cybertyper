"""Canonical IK wrappers.

The underlying primitives in `cybertyping/core/primitives.py` return
tuples `(q, residual)`. Callers that forgot to unpack the tuple have
crashed in the past (e.g. `slew_segment(q_now, q_scan, ...)` doing
tuple-minus-ndarray). Use the functions in this module everywhere
instead: they return an explicit dataclass that cannot be accidentally
treated as an array.

    from cybertyping.core import ik
    res = ik.solve_origin(chain, target_xyz, q_now)
    traj = motion.slew_segment(q_now, res.q, MAX_STEP_DEG_PER_TICK)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cybertyping.core import primitives as _ts


# 5 mm is the tolerance used for fixed-z targets.
# primitives.py uses 35 mm because it descends until contact
# rather than to a fixed z, so the IK residual doesn't need to be tight.
DEFAULT_TOL_M = 0.005


@dataclass
class IKResult:
    q: np.ndarray        # body-frame degrees, len(BODY_MOTOR_ORDER)
    residual: float      # |fk(q) - target| in meters
    ok: bool             # residual < tol

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=float)


def solve_origin(chain, target_xyz, q_init_deg,
                 enforce_gripper_down: bool = False,
                 tol_m: float = DEFAULT_TOL_M) -> IKResult:
    """IK so the gripper_frame_link **origin** lands at `target_xyz`.

    Use when you want the gripper itself at a pose (e.g. moving to a
    scan pose, hovering over a workspace point).
    """
    q, residual = _ts.solve_ik_origin(
        chain, target_xyz, q_init_deg,
        enforce_gripper_down=enforce_gripper_down,
    )
    return IKResult(q=q, residual=float(residual), ok=float(residual) < tol_m)


def solve_tip(chain, target_tip_xyz, q_warm_deg, tip_local,
              max_iter: int = 5,
              enforce_gripper_down: bool = False,
              tol_m: float = DEFAULT_TOL_M) -> IKResult:
    """IK so the **tool tip** (`tip_local` in the gripper frame) lands at
    `target_tip_xyz`.

    Use when you want a specific point on the tool to land somewhere
    (key center, dot center). `tip_local` is the tip's offset in the
    gripper's local frame; for gripper-only typing this should be close
    to (0, 0, 0). For pen typing it's the pen-tip offset.
    """
    q, residual = _ts.solve_ik_tip(
        chain, target_tip_xyz, q_warm_deg, tip_local,
        max_iter=max_iter,
        enforce_gripper_down=enforce_gripper_down,
    )
    return IKResult(q=q, residual=float(residual), ok=float(residual) < tol_m)


__all__ = ["IKResult", "solve_origin", "solve_tip", "DEFAULT_TOL_M"]
