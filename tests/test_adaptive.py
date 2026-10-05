"""Tests for the ADAPTIVE preset + per-key state learning."""

import numpy as np
import pytest

from cybertyping.tasks.fast_typer import (
    PreparedKey, PreparedKeymap,
    PRESET_FAST, PRESET_BLITZ, PRESET_SAFE, PRESET_ADAPTIVE, PRESETS,
    LEARNED_OFFSET_MIN_M, LEARNED_OFFSET_MAX_M,
    update_key_after_press, _resolve_per_key,
    serialize_adaptive_state, restore_adaptive_state,
)


def _mk_key(label, x=0.20, y=0.05):
    return PreparedKey(
        label=label, robot_xy=(x, y), plane_z=0.05, press_z=0.053,
        q_pre=np.zeros(5), q_press=np.array([1.0, 0, 0, 0, 0]),
        q_lifts={}, res_pre_m=0.0, res_press_m=0.0,
    )


# ---------------------------------------------------------------------------
# Confidence + bucketing
# ---------------------------------------------------------------------------

def test_confidence_neutral_when_no_history():
    k = _mk_key("a")
    assert k.confidence == 0.5
    assert not k.is_proven
    assert not k.is_problematic


def test_confidence_after_successes():
    k = _mk_key("a")
    for _ in range(3):
        update_key_after_press(k, registered=True)
    assert k.success_count == 3
    assert k.miss_count == 0
    assert k.confidence == 1.0
    assert k.is_proven
    assert not k.is_problematic


def test_is_problematic_on_repeated_misses():
    k = _mk_key("a")
    update_key_after_press(k, registered=False)
    update_key_after_press(k, registered=False)
    assert k.is_problematic
    assert not k.is_proven


# ---------------------------------------------------------------------------
# Retry depth learning
# ---------------------------------------------------------------------------

def test_retry_success_persists_learned_offset():
    k = _mk_key("a")
    # First press: miss
    update_key_after_press(k, registered=False, was_retry=False)
    # Retry at +2mm: success
    update_key_after_press(k, registered=True, was_retry=True,
                           retry_extra_depth_m=0.002)
    assert k.learned_press_offset_m == pytest.approx(0.002, abs=1e-9)
    # We DON'T count the retry success as a first-try success.
    assert k.miss_count == 1
    assert k.success_count == 0


def test_retry_miss_bumps_offset_by_half_depth():
    k = _mk_key("a")
    update_key_after_press(k, registered=False, was_retry=False)
    update_key_after_press(k, registered=False, was_retry=True,
                           retry_extra_depth_m=0.002)
    assert k.learned_press_offset_m == pytest.approx(0.001, abs=1e-9)


def test_learned_offset_clamped_to_max():
    k = _mk_key("a")
    update_key_after_press(k, registered=False)
    # Try to learn a way-too-deep offset
    update_key_after_press(k, registered=True, was_retry=True,
                           retry_extra_depth_m=0.020)
    assert k.learned_press_offset_m <= LEARNED_OFFSET_MAX_M + 1e-9


# ---------------------------------------------------------------------------
# Preset registration
# ---------------------------------------------------------------------------

def test_adaptive_preset_registered():
    assert "ADAPTIVE" in PRESETS
    assert PRESET_ADAPTIVE.adaptive is True
    # NOTE: verify+retry intentionally disabled across all presets because
    # the eval keyboard is not plugged into our laptop -> pynput sees no
    # events -> every key reports as missed -> phantom retries burn the
    # time budget. See cybertyping/tasks/fast_typer.py.
    assert PRESET_ADAPTIVE.use_keystroke_verify is False
    assert PRESET_ADAPTIVE.allow_retry is False


def test_non_adaptive_presets_have_adaptive_false():
    for name in ("SAFE", "FAST", "BLITZ"):
        assert PRESETS[name].adaptive is False


# ---------------------------------------------------------------------------
# Per-key effective parameter resolution
# ---------------------------------------------------------------------------

def test_resolve_per_key_non_adaptive_returns_defaults():
    k = _mk_key("a")
    k.success_count = 10  # proven
    eff = _resolve_per_key(PRESET_FAST, k)
    # FAST is non-adaptive: per-key proven status shouldn't change the params
    assert eff.max_step_deg_per_tick == PRESET_FAST.max_step_deg_per_tick
    assert eff.lift_scale == 1.0


def test_resolve_per_key_adaptive_proven_uses_blitz_style():
    k = _mk_key("a")
    for _ in range(3):
        update_key_after_press(k, registered=True)
    eff = _resolve_per_key(PRESET_ADAPTIVE, k)
    assert eff.max_step_deg_per_tick == PRESET_BLITZ.max_step_deg_per_tick
    assert eff.press_dwell_ticks == 1
    assert eff.lift_scale < 1.0


def test_resolve_per_key_adaptive_problematic_uses_safe_style():
    k = _mk_key("a")
    update_key_after_press(k, registered=False)
    update_key_after_press(k, registered=False)
    eff = _resolve_per_key(PRESET_ADAPTIVE, k)
    assert eff.max_step_deg_per_tick == PRESET_SAFE.max_step_deg_per_tick
    assert eff.press_dwell_ticks >= 3
    # Should include an extra 1mm safety margin on top of any learned offset
    assert eff.press_offset_m >= 0.001 - 1e-9


def test_resolve_per_key_adaptive_neutral_uses_preset_defaults():
    k = _mk_key("a")  # no history -> neutral
    eff = _resolve_per_key(PRESET_ADAPTIVE, k)
    assert eff.max_step_deg_per_tick == PRESET_ADAPTIVE.max_step_deg_per_tick


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_serialize_round_trip():
    keys = {
        "a": _mk_key("a"),
        "b": _mk_key("b"),
    }
    # Mutate state
    keys["a"].success_count = 5
    keys["a"].learned_press_offset_m = 0.0015
    keys["b"].miss_count = 2
    keys["b"].last_correction_xy_m = (0.001, -0.0005)

    prepared = PreparedKeymap(
        keys=keys, press_offset_m=0.003, preset=PRESET_ADAPTIVE,
    )
    state = serialize_adaptive_state(prepared)
    assert state["preset_name"] == "ADAPTIVE"
    assert state["press_offset_m"] == pytest.approx(0.003)
    assert state["per_key"]["a"]["success_count"] == 5

    # Build a fresh PreparedKeymap and restore
    fresh = PreparedKeymap(
        keys={"a": _mk_key("a"), "b": _mk_key("b"), "c": _mk_key("c")},
        press_offset_m=0.003, preset=PRESET_ADAPTIVE,
    )
    n = restore_adaptive_state(fresh, state)
    assert n == 2  # only a and b were in the saved state
    assert fresh.keys["a"].success_count == 5
    assert fresh.keys["a"].learned_press_offset_m == pytest.approx(0.0015)
    assert fresh.keys["b"].miss_count == 2
    # c was never in saved state -> defaults preserved
    assert fresh.keys["c"].success_count == 0
