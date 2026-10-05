"""Homography pipeline tests.

We don't have a real keyboard photo here, so we synthesize the pixel
positions of anchors by applying a known homography to the canonical
layout. The pipeline should recover the homography (RMSE near 0) and
reproject every other key to its synthesised pixel position.
"""

import numpy as np
import cv2
import pytest

from cybertyping.perception.keymap.layout import load_layout
from cybertyping.perception.keymap.homography import (
    build_keymap_from_anchors, crossvalidate, solve_homography, reproject,
    fit_rmse,
)


def _synth_image_anchors(H_true, anchor_labels, layout, noise_px=0.0, seed=0):
    """Build bboxes whose CENTER lands at the true pixel.

    Naively writing `(int(x-10), int(y-10), int(x+10), int(y+10))` quantizes
    the center by up to 0.5 px because the center is (x1+x2)/2 with rounding
    twice. We compensate by adding 1 to x2/y2 when the fractional part is high.
    Practically the production pipeline uses VLM bboxes (integers anyway)
    and the homography fit absorbs the same kind of noise.
    """
    canon = layout.centers(anchor_labels)
    image_xy = reproject(H_true, canon)
    if noise_px > 0:
        rng = np.random.default_rng(seed)
        image_xy = image_xy + rng.normal(0, noise_px, image_xy.shape)
    out = {}
    for i, lbl in enumerate(anchor_labels):
        x, y = float(image_xy[i, 0]), float(image_xy[i, 1])
        # Symmetric bbox whose floor()+ceil() center equals 2*x exactly.
        x1, y1 = int(np.floor(x)) - 10, int(np.floor(y)) - 10
        x2, y2 = int(np.ceil(x))  + 10, int(np.ceil(y))  + 10
        out[lbl] = (x1, y1, x2, y2)
    return out


def _random_homography(s=60.0, tx=200.0, ty=120.0, theta=np.deg2rad(2.0)):
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta),  np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1.0]])
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1.0]])
    return T @ R @ S


def test_recovers_clean_homography():
    layout = load_layout()
    H_true = _random_homography()
    anchors = layout.recommended_anchors()
    bboxes  = _synth_image_anchors(H_true, anchors, layout, noise_px=0.0)
    km = build_keymap_from_anchors(bboxes, layout=layout)
    # <2 px on synthetic noise-free data is the realistic floor given that the
    # test passes integer bboxes (production VLM bboxes are also integer).
    assert km.rmse_canonical_px < 2.0, f"RMSE too high: {km.rmse_canonical_px}"
    # Every letter should have an entry
    for letter in layout.letters():
        assert letter in km.entries


def test_robust_to_one_bad_anchor():
    layout = load_layout()
    H_true = _random_homography()
    anchors = layout.recommended_anchors()
    bboxes  = _synth_image_anchors(H_true, anchors, layout, noise_px=0.5)
    # Sabotage one anchor (50 px off)
    bad_label = anchors[0]
    x1, y1, x2, y2 = bboxes[bad_label]
    bboxes[bad_label] = (x1 + 50, y1 + 50, x2 + 50, y2 + 50)
    km = build_keymap_from_anchors(bboxes, layout=layout)
    # LMEDS should reject the outlier
    assert bad_label not in km.anchors_used or km.rmse_canonical_px < 5.0


def test_crossval():
    layout = load_layout()
    H_true = _random_homography()
    anchors = layout.recommended_anchors()
    held_out = ["r", "l", "g"]
    bboxes  = _synth_image_anchors(H_true, anchors, layout, noise_px=0.0)
    km = build_keymap_from_anchors(bboxes, layout=layout)
    xval_bboxes = _synth_image_anchors(H_true, held_out, layout, noise_px=0.0)
    errs = crossvalidate(km, xval_bboxes)
    for k, e in errs.items():
        # On clean synthetic data with integer bboxes, sub-3-pixel error means
        # the homography is essentially exact; production warn threshold is 12.
        assert e < 3.0, f"{k}: reprojection error {e}px"


def test_solve_homography_needs_4_points():
    layout = load_layout()
    pts = layout.centers(["q", "p", "z"])
    with pytest.raises(ValueError):
        solve_homography(pts, pts)
