"""Tests for the multi-frame OCR scan (detection union across N frames)."""

import numpy as np
import pytest

from cybertyping.perception.keymap.layout import load_layout
from cybertyping.perception.detect.ocr_backends import MockOCRBackend
from cybertyping.perception.detect.ocr_anchors import (
    detect_anchors_via_ocr_multiframe,
)


def _synth_truth(missing=()):
    """Build a truth list where some letters are intentionally missing
    (to simulate occlusion / OCR misses in a single frame)."""
    layout = load_layout()
    out = []
    for lbl in layout.letters() + ["space", "enter"]:
        if lbl in missing:
            continue
        cx, cy = layout.center(lbl)
        ix, iy = cx * 60 + 200, cy * 60 + 120
        out.append((lbl, (int(ix - 12), int(iy - 12),
                          int(ix + 12), int(iy + 12))))
    return out


def _dummy_image(w=1200, h=600, fill=200):
    return np.full((h, w, 3), fill, dtype=np.uint8)


def test_multiframe_union_recovers_keys_missing_in_some_frames():
    """Each frame's OCR sees a different subset; the union should cover
    everything either frame saw."""
    truths = [
        _synth_truth(missing={"q", "z", "m"}),    # frame 1 misses these
        _synth_truth(missing={"p", "a", "j"}),    # frame 2 misses different ones
        _synth_truth(missing={"r", "l"}),         # frame 3 misses yet another set
    ]
    images = [_dummy_image() for _ in truths]
    frame_idx = [0]

    def grab_frame():
        i = frame_idx[0]
        frame_idx[0] = (i + 1) % len(images)
        return images[i]

    # We need different MockOCRBackends per frame to simulate per-frame
    # variability. We'll do this by wrapping the call.
    ocrs = [MockOCRBackend(t) for t in truths]
    backend_idx = [0]

    class _RotatingOCR:
        def read_full(self, image):
            i = backend_idx[0]
            backend_idx[0] = (i + 1) % len(ocrs)
            return ocrs[i].read_full(image)
        def read_crop(self, *a, **k):
            return ""

    union, frames = detect_anchors_via_ocr_multiframe(
        grab_frame=grab_frame, n_frames=3, inter_frame_delay_s=0.0,
        ocr=_RotatingOCR(),
    )
    layout = load_layout()
    # No single frame's OCR returned all letters, but the union should
    # include keys that *any* frame saw.
    for letter in layout.letters():
        # At least 2 of the 3 truths include each letter, so union should
        # cover all 26.
        assert letter in union, f"{letter} missing from union"


def test_multiframe_with_single_frame_equals_single_pass():
    """n_frames=1 should produce the same result as detect_anchors_via_ocr."""
    truth = _synth_truth()
    img = _dummy_image()
    ocr = MockOCRBackend(truth)

    union, frames = detect_anchors_via_ocr_multiframe(
        grab_frame=lambda: img, n_frames=1, inter_frame_delay_s=0.0,
        ocr=ocr,
    )
    layout = load_layout()
    for letter in layout.letters():
        assert letter in union
    assert len(frames) == 1
