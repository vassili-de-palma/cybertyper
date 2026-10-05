"""Load-based contact descent with an open-loop fallback.

Primary: :func:`cybertyping.core.primitives.descent_until_contact`
(in-flight load baseline + threshold + n-consecutive-tick confirmation).

The primitive reads ``robot.bus.sync_read("Present_Load")`` directly. If the
driver does not expose the bus, or the load register is missing, the press
falls back to a fixed open-loop depth from ``configs/press_depth.json``.
Once the fallback triggers, every later call in the same process goes
straight to it, so a dead register does not cost a failed attempt per key.
"""

from __future__ import annotations

from cybertyping.core.primitives import descent_until_contact as _descent_load
from . import open_loop as _open_loop


_MODE: str = "load"


def _warn(msg: str) -> None:
    print(f"[load_contact] {msg}")


def press(chain, robot, pre_tip_xyz, q_pre, tip_off, gripper_pos,
          plot_state=None):
    """Contact descent with open-loop fallback.

    Signature matches ``descent_until_contact``. Returns
    ``(q_target, descent_qs, info)``; ``info["reason"]`` records which
    strategy actually ran (``load_contact`` or ``open_loop_fallback``).
    """
    global _MODE

    if _MODE == "load":
        try:
            return _descent_load(
                chain, robot, pre_tip_xyz, q_pre, tip_off, gripper_pos,
            )
        except Exception as e:  # AttributeError (no .bus) or driver errors
            _MODE = "openloop"
            _warn(f"load descent raised {e!r}; using open-loop depth for the "
                  f"rest of this session.")

    q, qs, info = _open_loop.press(
        robot, chain, pre_tip_xyz, q_pre, tip_off, gripper_pos,
    )
    info = dict(info) if isinstance(info, dict) else {"info": info}
    info.setdefault("reason", "open_loop_fallback")
    return q, qs, info


__all__ = ["press"]
