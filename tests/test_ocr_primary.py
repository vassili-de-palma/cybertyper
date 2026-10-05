"""Tests for the OCR-primary detection path.

Uses MockOCRBackend.read_full to fake full-image OCR output, then exercises
detect_anchors_via_ocr end-to-end.
"""

import numpy as np
import pytest

from cybertyping.perception.keymap.layout import load_layout
from cybertyping.perception.keymap.homography import build_keymap_from_anchors
from cybertyping.perception.detect.ocr_backends import MockOCRBackend
from cybertyping.perception.detect.ocr_anchors import (
    detect_anchors_via_ocr, _single_char, annotate_ocr_detections,
)


def _synth_truth(scale=60.0, ox=200.0, oy=120.0):
    """Build full-image (label, bbox) truth for every letter + space + enter."""
    layout = load_layout()
    out = []
    for lbl in layout.letters() + ["space", "enter"]:
        cx, cy = layout.center(lbl)
        ix, iy = cx * scale + ox, cy * scale + oy
        out.append((lbl, (int(ix - 12), int(iy - 12),
                          int(ix + 12), int(iy + 12))))
    return out


def _dummy_image(w=1200, h=600):
    return np.full((h, w, 3), 200, dtype=np.uint8)


def test_single_char_extraction():
    assert _single_char("q") == "q"
    assert _single_char("Q") == "q"
    assert _single_char("qq") == "q"
    assert _single_char("q.") == "q"
    assert _single_char("qr") is None      # ambiguous mixed letters
    assert _single_char("") is None
    assert _single_char("12") is None


def test_detect_anchors_via_ocr_with_mock():
    truth = _synth_truth()
    img = _dummy_image()
    backend = MockOCRBackend(truth)
    dets = detect_anchors_via_ocr(img, ocr=backend)
    # Every letter should have been detected
    layout = load_layout()
    for letter in layout.letters():
        assert letter in dets, f"missing {letter}"


def test_ocr_anchors_drive_homography():
    """The detections produced by OCR-primary should be fittable by the
    same build_keymap_from_anchors function used by the VLM path."""
    truth = _synth_truth()
    img = _dummy_image()
    backend = MockOCRBackend(truth)
    dets = detect_anchors_via_ocr(img, ocr=backend)
    km = build_keymap_from_anchors(dets)
    # Synthetic, noise-free truth -> recovery should be tight.
    assert km.rmse_canonical_px < 3.0


def test_annotate_does_not_crash():
    """Smoke test for visualisation."""
    truth = _synth_truth()
    img = _dummy_image()
    backend = MockOCRBackend(truth)
    dets = detect_anchors_via_ocr(img, ocr=backend)
    annotated = annotate_ocr_detections(img, dets)
    assert annotated.shape == img.shape
