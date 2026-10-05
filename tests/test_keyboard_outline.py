"""Synthetic test: draw a dark rectangle on a light background, recover its
corners."""

import cv2
import numpy as np
import pytest

from cybertyping.perception.detect.outline_rect import (
    detect_keyboard_outline, homography_from_outline,
)


def test_finds_axis_aligned_rectangle():
    img = np.full((600, 1000, 3), 200, dtype=np.uint8)  # light gray bg
    # Black "keyboard" rectangle
    cv2.rectangle(img, (200, 150), (800, 450), (15, 15, 15), thickness=-1)

    out = detect_keyboard_outline(img)
    assert out is not None
    corners = out.corners
    assert corners.shape == (4, 2)
    # Ordered TL, TR, BR, BL. Check approximate positions (within ~3px tolerance).
    tl, tr, br, bl = corners
    assert np.allclose(tl, (200, 150), atol=3)
    assert np.allclose(tr, (800, 150), atol=3)
    assert np.allclose(br, (800, 450), atol=3)
    assert np.allclose(bl, (200, 450), atol=3)
    assert out.score > 0.9


def test_finds_rotated_rectangle():
    img = np.full((600, 1000, 3), 200, dtype=np.uint8)
    # Rotated rectangle (skewed via rotation matrix on its 4 corners)
    cx, cy = 500, 300
    w, h = 600, 300
    theta = np.deg2rad(8.0)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    base = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    pts = (base @ R.T + np.array([cx, cy])).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(img, [pts], (15, 15, 15))

    out = detect_keyboard_outline(img)
    assert out is not None
    assert out.score > 0.9
    # Centroid of detected corners should match cx, cy
    centroid = out.corners.mean(axis=0)
    assert abs(centroid[0] - cx) < 4
    assert abs(centroid[1] - cy) < 4


def test_returns_none_when_no_dark_region():
    img = np.full((600, 1000, 3), 200, dtype=np.uint8)
    out = detect_keyboard_outline(img)
    # All-light image: Otsu may produce a near-empty mask. Either None or a
    # very low-score outline is acceptable. We require it not to confidently
    # claim a result.
    if out is not None:
        assert out.score < 0.95 or out.area_px < 100
