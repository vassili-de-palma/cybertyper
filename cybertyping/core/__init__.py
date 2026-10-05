"""Shared kinematics, motion and contact-sensing primitives.

Everything is defined in :mod:`cybertyping.core.primitives` and re-exported
here so callers have one stable import path::

    from cybertyping.core import build_chain, fk_tip, solve_ik_tip, BODY_MOTOR_ORDER

Sub-modules:

* ``primitives``   ikpy chain, FK/IK, slewing, streaming, load-contact descent
* ``ik``           dataclass wrappers around the raw IK solvers
* ``descent_jac``  Jacobian velocity-control descent (no per-tick IK drift)
* ``runtime``      robot connection, dry-run switch, URDF path, logging helpers
"""

from . import primitives as _p

# Kinematics
build_chain             = _p.build_chain
motor_deg_to_full_rad   = _p.motor_deg_to_full_rad
full_rad_to_body_deg    = _p.full_rad_to_body_deg
fk_T                    = _p.fk_T
fk_origin               = _p.fk_origin
fk_tip                  = _p.fk_tip
clamp_initial_to_bounds = _p.clamp_initial_to_bounds
solve_ik_origin         = _p.solve_ik_origin
solve_ik_tip            = _p.solve_ik_tip

# Streaming / motion
slew_segment            = _p.slew_segment
slew_through            = _p.slew_through
stream                  = _p.stream

# Contact sensing
measure_baseline        = _p.measure_baseline
descent_until_contact   = _p.descent_until_contact

# Geometry
point_in_convex_quad    = _p.point_in_convex_quad
bbox_quad               = _p.bbox_quad

# Persistence
load_world              = _p.load_world
save_world              = _p.save_world

# Constants
BODY_MOTOR_ORDER            = _p.BODY_MOTOR_ORDER
LOCKED_JOINTS               = _p.LOCKED_JOINTS
TOOL_TIP_OFFSET_LOCAL_XYZ   = _p.TOOL_TIP_OFFSET_LOCAL_XYZ
URDF_PATH                   = _p.URDF_PATH
LOOP_HZ                     = _p.LOOP_HZ
MAX_STEP_DEG_PER_TICK_APPROACH = _p.MAX_STEP_DEG_PER_TICK_APPROACH
MAX_STEP_DEG_PER_TICK_RETREAT  = _p.MAX_STEP_DEG_PER_TICK_RETREAT

from . import ik       # noqa: E402, F401
from . import runtime  # noqa: E402, F401

__all__ = [
    "build_chain", "motor_deg_to_full_rad", "full_rad_to_body_deg",
    "fk_T", "fk_origin", "fk_tip", "clamp_initial_to_bounds",
    "solve_ik_origin", "solve_ik_tip", "ik",
    "slew_segment", "slew_through", "stream",
    "measure_baseline", "descent_until_contact",
    "point_in_convex_quad", "bbox_quad",
    "load_world", "save_world",
    "BODY_MOTOR_ORDER", "LOCKED_JOINTS", "TOOL_TIP_OFFSET_LOCAL_XYZ",
    "URDF_PATH", "LOOP_HZ",
    "MAX_STEP_DEG_PER_TICK_APPROACH", "MAX_STEP_DEG_PER_TICK_RETREAT",
]
