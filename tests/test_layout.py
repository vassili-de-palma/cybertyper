"""Canonical layout sanity checks."""

import numpy as np
import pytest

from cybertyping.perception.keymap.layout import load_layout


def test_loads():
    L = load_layout()
    # All 26 letters present
    assert set(L.letters()) == set("abcdefghijklmnopqrstuvwxyz")
    # Specials we rely on
    assert "space" in L.keys
    assert "enter" in L.keys
    assert "q" in L.keys
    assert "p" in L.keys


def test_stagger_geometry():
    """ANSI stagger constants we hard-coded must hold."""
    L = load_layout()
    # QWERTY row Y == 1.5, ASDF == 2.5, ZXCV == 3.5
    assert L.keys["q"].y == pytest.approx(1.5)
    assert L.keys["a"].y == pytest.approx(2.5)
    assert L.keys["z"].y == pytest.approx(3.5)
    # Unit spacing in x within a row
    assert L.keys["w"].x - L.keys["q"].x == pytest.approx(1.0)
    assert L.keys["s"].x - L.keys["a"].x == pytest.approx(1.0)
    # ASDF row offset +0.25u from QWERTY row
    assert L.keys["a"].x - L.keys["q"].x == pytest.approx(0.25)
    # ZXCV row offset +0.5u from ASDF row
    assert L.keys["z"].x - L.keys["a"].x == pytest.approx(0.5)


def test_centers_matrix_shape():
    L = load_layout()
    pts = L.centers(["q", "p", "z", "m"])
    assert pts.shape == (4, 2)
    assert np.all(np.isfinite(pts))


def test_recommended_anchors():
    L = load_layout()
    anc = L.recommended_anchors()
    assert len(anc) >= 4
    for k in anc:
        assert k in L.keys, f"recommended anchor {k!r} missing from layout"
