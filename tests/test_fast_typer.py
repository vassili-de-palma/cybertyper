"""Tests for fast_typer.py speed presets, lift selection, and trajectory
generation.

We can't fully exercise prepare_keymap without ikpy (which depends on
the URDF and lerobot toolchain), so the trajectory tests use a hand-built
PreparedKeymap with synthetic per-key joint configs.
"""

import os
from unittest import mock

import numpy as np
import pytest

from cybertyping.tasks.fast_typer import (
    PRESETS, PRESET_SAFE, PRESET_FAST, PRESET_BLITZ,
    SpeedPreset, preset_from_env, _inter_key_lift_m,
    PreparedKey, PreparedKeymap,
)


def test_presets_registered():
    assert set(PRESETS.keys()) == {"SAFE", "FAST", "BLITZ", "ADAPTIVE"}
    for p in PRESETS.values():
        assert isinstance(p, SpeedPreset)
        # Sanity: SAFE should be slower than FAST should be slower than BLITZ.
    assert PRESET_SAFE.max_step_deg_per_tick < PRESET_FAST.max_step_deg_per_tick
    assert PRESET_FAST.max_step_deg_per_tick < PRESET_BLITZ.max_step_deg_per_tick
    # BLITZ has tinier lifts than SAFE.
    assert PRESET_BLITZ.inter_key_lift_adjacent_m < PRESET_SAFE.inter_key_lift_adjacent_m


def test_preset_from_env_default():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CYBERTYPING_PRESET", None)
        assert preset_from_env(default="FAST").name == "FAST"
        assert preset_from_env(default="SAFE").name == "SAFE"


def test_preset_from_env_override():
    with mock.patch.dict(os.environ, {"CYBERTYPING_PRESET": "BLITZ"}):
        assert preset_from_env(default="SAFE").name == "BLITZ"


def test_preset_from_env_rejects_unknown():
    with mock.patch.dict(os.environ, {"CYBERTYPING_PRESET": "TURBO"}):
        with pytest.raises(ValueError):
            preset_from_env()


def _mk_key(label, x, y, plane_z=0.05):
    """Hand-build a PreparedKey without the IK step."""
    q_pre = np.zeros(5)
    q_press = np.zeros(5)
    q_press[0] = 1.0   # something different from q_pre
    return PreparedKey(
        label=label, robot_xy=(x, y),
        plane_z=plane_z, press_z=plane_z + 0.003,
        q_pre=q_pre, q_press=q_press,
        q_lifts={
            int(round(PRESET_FAST.inter_key_lift_adjacent_m * 1000)): q_pre.copy(),
            int(round(PRESET_FAST.inter_key_lift_near_m * 1000)): q_pre.copy(),
            int(round(PRESET_FAST.inter_key_lift_far_m * 1000)): q_pre.copy(),
            int(round(PRESET_FAST.approach_height_m * 1000)): q_pre.copy(),
        },
    )


def test_lift_selection_adjacent():
    """Two keys 1.5 cm apart should get the adjacent lift."""
    a = _mk_key("a", 0.20, 0.05)
    b = _mk_key("b", 0.215, 0.05)
    lift = _inter_key_lift_m(a, b, PRESET_FAST)
    assert lift == PRESET_FAST.inter_key_lift_adjacent_m


def test_lift_selection_near():
    """Two keys 5 cm apart -> near lift."""
    a = _mk_key("a", 0.20, 0.05)
    b = _mk_key("b", 0.25, 0.05)
    lift = _inter_key_lift_m(a, b, PRESET_FAST)
    assert lift == PRESET_FAST.inter_key_lift_near_m


def test_lift_selection_far():
    """Two keys 12 cm apart -> far lift (full lift over the keyboard)."""
    a = _mk_key("a", 0.20, 0.05)
    b = _mk_key("b", 0.32, 0.05)
    lift = _inter_key_lift_m(a, b, PRESET_FAST)
    assert lift == PRESET_FAST.inter_key_lift_far_m


def test_trajectory_planning_requires_core():
    """plan_sentence_trajectory uses core.slew_segment; verify it produces
    something when given a hand-built PreparedKeymap.

    If ikpy isn't installed (and core fails to import), skip rather than fail.
    """
    try:
        from cybertyping import core  # noqa: F401
    except ImportError:
        pytest.skip("ikpy / lerobot not installed; cannot import core")

    from cybertyping.tasks.fast_typer import plan_sentence_trajectory
    prepared = PreparedKeymap(
        keys={
            "a": _mk_key("a", 0.20, 0.05),
            "b": _mk_key("b", 0.215, 0.05),
            "c": _mk_key("c", 0.30, 0.05),
        },
        press_offset_m=0.003,
        preset=PRESET_FAST,
        q_scan=np.zeros(5),
    )
    traj = plan_sentence_trajectory(prepared, ["a", "b", "c"])
    assert isinstance(traj, list)
    assert len(traj) > 0
    # every element is a 5-vector
    for q in traj:
        assert isinstance(q, np.ndarray)
        assert q.shape == (5,)


def test_estimate_sentence_time_proportional_to_length():
    try:
        from cybertyping import core  # noqa: F401
    except ImportError:
        pytest.skip("ikpy / lerobot not installed; cannot import core")

    from cybertyping.tasks.fast_typer import estimate_sentence_time_s
    prepared = PreparedKeymap(
        keys={
            ch: _mk_key(ch, 0.20 + i * 0.02, 0.05)
            for i, ch in enumerate("abcdefg")
        },
        press_offset_m=0.003,
        preset=PRESET_FAST,
        q_scan=np.zeros(5),
    )
    t1 = estimate_sentence_time_s(prepared, ["a"])
    t4 = estimate_sentence_time_s(prepared, ["a", "b", "c", "d"])
    assert t1 >= 0.0
    assert t4 > t1
