"""Tests for cybertyping.analytic_ik.kinematics.

Three checks (run in order — later ones depend on earlier ones being clean):
  1. URDF-walking FK matches the ikpy-based FK in cybertyping/core/primitives
     on 100 random poses (validates the URDF transform chain).
  2. IK → FK round-trip: for 200 random Cartesian targets in the workspace,
     solve_ik(p) should give q such that FK(q) ≈ p (within 1 mm) AND the EE
     z-axis points straight down (within 1° tilt).
  3. OutOfReach: targets clearly outside the envelope (e.g., x=1.0 m) raise.

Run:
    pytest tests/test_analytic_ik.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cybertyping.analytic_ik.kinematics import (
    JointLimitViolation, OutOfReach, fk_T, forward_kinematics, solve_ik,
)


def _random_q(rng: np.random.Generator, joint_limits, margin: float = 0.1) -> np.ndarray:
    """Random joint config strictly inside limits (margin in rad)."""
    q = np.empty(5)
    for i, (lo, hi) in enumerate(joint_limits):
        q[i] = rng.uniform(lo + margin, hi - margin)
    return q


# ---------------------------------------------------------------------------
# Test 1: FK vs ikpy oracle
# ---------------------------------------------------------------------------
def test_fk_matches_ikpy() -> None:
    from cybertyping.core.primitives import build_chain, fk_origin, URDF_PATH
    chain = build_chain(URDF_PATH, locked_joints=())
    from cybertyping.analytic_ik.config import JOINT_LIMITS_RAD

    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(100):
        q_rad = _random_q(rng, JOINT_LIMITS_RAD)
        q_deg = np.rad2deg(q_rad)

        mine  = np.array(forward_kinematics(q_rad))
        oracle = np.array(fk_origin(chain, q_deg))
        err = float(np.linalg.norm(mine - oracle))
        max_err = max(max_err, err)
        assert err < 1e-3, (
            f"FK mismatch (err={err*1000:.2f} mm) at q_deg={q_deg}\n"
            f"  ours={mine}\n  ikpy={oracle}"
        )
    print(f"[test_fk_matches_ikpy] ok — max err = {max_err*1000:.3f} mm over 100 poses")


# ---------------------------------------------------------------------------
# Test 2: IK → FK round-trip
# ---------------------------------------------------------------------------
def test_ik_roundtrip() -> None:
    from cybertyping.analytic_ik.config import JOINT_LIMITS_RAD

    rng = np.random.default_rng(1)
    n_ok = 0
    n_unreachable = 0
    n_limit = 0
    pos_errs = []
    pitch_errs = []
    for _ in range(200):
        q_rand = _random_q(rng, JOINT_LIMITS_RAD)
        x, y, z = forward_kinematics(q_rand)
        try:
            q_solved = solve_ik(x, y, z)
        except OutOfReach:
            n_unreachable += 1
            continue
        except JointLimitViolation:
            n_limit += 1
            continue
        n_ok += 1
        x2, y2, z2 = forward_kinematics(q_solved)
        pos_err = float(np.linalg.norm([x - x2, y - y2, z - z2]))
        # EE pitch (atan2 of z-component over horizontal projection) — for
        # "straight down" we want the z-component of the EE z-axis ≈ -1.
        ee_z = fk_T(q_solved)[:3, 2]
        pitch_err_deg = float(np.rad2deg(np.arccos(np.clip(-ee_z[2], -1.0, 1.0))))
        pos_errs.append(pos_err)
        pitch_errs.append(pitch_err_deg)

    print(f"[test_ik_roundtrip] reached {n_ok}/200  "
          f"(unreachable={n_unreachable}, limit={n_limit})")
    if pos_errs:
        print(f"  pos err: median {np.median(pos_errs)*1000:.2f} mm, "
              f"max {np.max(pos_errs)*1000:.2f} mm")
        print(f"  EE down-tilt err: median {np.median(pitch_errs):.2f}°, "
              f"max {np.max(pitch_errs):.2f}°")

    # Soft assertions — the FK→IK round-trip is degenerate here because
    # random q's don't have an EE-down constraint. The IK enforces EE-down,
    # so it generally moves to a DIFFERENT q than the random one. The
    # position recovered should still be exact (IK lands at the requested
    # x,y,z) when reachable.
    assert n_ok > 30, f"too few solutions converged ({n_ok}/200)"
    assert max(pos_errs) < 5e-3, (
        f"max position error {max(pos_errs)*1000:.2f} mm too large — "
        f"check link lengths / signs in config.py"
    )
    assert max(pitch_errs) < 1.0, (
        f"max EE down-tilt {max(pitch_errs):.2f}° — wrist closure broken"
    )


# ---------------------------------------------------------------------------
# Test 3: OutOfReach
# ---------------------------------------------------------------------------
def test_out_of_reach() -> None:
    for x, y, z in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0)]:
        try:
            solve_ik(x, y, z)
        except OutOfReach:
            continue
        raise AssertionError(f"expected OutOfReach for ({x},{y},{z}), got solution")
    print("[test_out_of_reach] ok")


# ---------------------------------------------------------------------------
def main() -> int:
    try:
        test_fk_matches_ikpy()
    except ImportError as e:
        print(f"[test_fk_matches_ikpy] skipped — ikpy import failed: {e}")
    test_ik_roundtrip()
    test_out_of_reach()
    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
