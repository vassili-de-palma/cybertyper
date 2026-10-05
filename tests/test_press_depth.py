"""Tests for press_depth solver (hardware-free; uses the self-test path)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cybertyping.calibration.press_depth import (
    PressDepthResult, load_press_depth, _self_test,
)


def test_self_test_passes():
    """The synthetic self-test in press_depth.py should run cleanly."""
    _self_test()


def test_result_save_and_load(tmp_path):
    res = PressDepthResult(
        offset_m=0.0032,
        per_key={"f": 0.0030, "j": 0.0031, "space": 0.0035},
        rms_residual_mm=0.22,
        n_keys_used=3,
    )
    out = tmp_path / "press_depth.json"
    res.save(out)
    assert out.exists()
    loaded = load_press_depth(out)
    assert loaded == pytest.approx(0.0032, rel=1e-6)


def test_load_press_depth_missing_file_returns_none(tmp_path):
    out = tmp_path / "nonexistent.json"
    assert load_press_depth(out) is None


def test_result_to_json_schema():
    res = PressDepthResult(
        offset_m=0.0033, per_key={"a": 0.0033},
        rms_residual_mm=0.0, n_keys_used=1,
    )
    d = res.to_json()
    assert "offset_m" in d and isinstance(d["offset_m"], float)
    assert "per_key_offset_m" in d
    assert "rms_residual_mm" in d
    assert "n_keys_used" in d
    assert "captured_at" in d
