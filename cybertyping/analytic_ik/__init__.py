"""Closed-form reach-and-touch for the SO-101.

Analytical inverse kinematics (no ikpy, no Jacobian) plus a rolling-baseline
``Present_Load`` touch detector. Written as an independent cross-check of the
ikpy-based primitives in :mod:`cybertyping.core`; ``tests/test_analytic_ik.py``
verifies both forward kinematics agree to within 1 mm.

Entry point::

    CYBERTYPING_DRY_RUN=1 python -m cybertyping.analytic_ik.main --x 0.20 --y 0.05   # IK only
    python -m cybertyping.analytic_ik.main --x 0.20 --y 0.05 --reach-only            # hover, no descent
    python -m cybertyping.analytic_ik.main --x 0.20 --y 0.05                         # reach + touch

Before the first real run:

1. ``pytest tests/test_analytic_ik.py`` must pass; if not, the URDF-derived
   constants in ``config.py`` are stale.
2. Calibrate the touch threshold with a descent into empty air (set
   ``--z-floor`` well below any surface) and set ``config.TOUCH_THRESHOLD``
   to about five times the load noise.
3. If contact fires when the end effector is pushed *up* by hand, pass
   ``--touch-sign -1``.
"""
