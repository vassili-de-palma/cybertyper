"""Eval 1 dot-detector test on synthetic images."""

import cv2
import numpy as np

from cybertyping.perception.detect.color_blobs import detect_dots, EVAL1_DOT_ORDER


def _synth_image_with_dots():
    img = np.full((600, 800, 3), 230, dtype=np.uint8)  # paper white
    placements = {
        "red":   (120, 300),
        "green": (320, 150),
        "blue":  (520, 300),
        "cyan":  (320, 450),
    }
    bgrs = {
        "red":   (0,   0,   220),
        "green": (0,   220, 0),
        "blue":  (220, 0,   0),
        "cyan":  (220, 220, 140),
    }
    for color, center in placements.items():
        cv2.circle(img, center, 13, bgrs[color], thickness=-1)
    return img, placements


def test_detects_all_four():
    img, expected = _synth_image_with_dots()
    dots = detect_dots(img)
    for color in EVAL1_DOT_ORDER:
        assert color in dots, f"{color} not detected"
        ex, ey = expected[color]
        px, py = dots[color].pixel_xy
        err = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
        assert err < 3.0, f"{color}: error {err:.2f}px too high"
