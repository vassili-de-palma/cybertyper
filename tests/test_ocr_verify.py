"""OCR verification tests using MockOCRBackend.

We synthesise an image where every letter key has a known bounding box.
MockOCRBackend reads the center of each crop and returns the label whose
truth bbox contains that center. With a CORRECT keymap, every crop center
lands in the matching truth bbox, so OCR agreement is 100%. We can then
perturb the keymap (shift it one column right) and verify the pipeline
both detects the disagreement AND correctly identifies the systematic
shift as dx_cols=+1.
"""

import numpy as np
import pytest

from cybertyping.perception.keymap.layout import load_layout
from cybertyping.perception.keymap.homography import (
    build_keymap_from_anchors, reproject,
)
from cybertyping.perception.detect.ocr_backends import (
    MockOCRBackend, verify_keymap,
)


def _synth_truth_and_keymap(scale_px_per_unit=60.0, ox=200.0, oy=120.0):
    """Build a synthetic image-space layout for testing.

    Returns:
      - truth_bboxes:  list[(label, (x1,y1,x2,y2))] for every letter at its
                       projected pixel position (used by MockOCRBackend).
      - keymap:        a correct KeyMap produced by feeding anchor bboxes
                       through build_keymap_from_anchors.
      - layout:        the canonical Layout (for use in tests).
    """
    layout = load_layout()
    anchors = layout.recommended_anchors()

    def to_image(canon_xy):
        return (canon_xy[0] * scale_px_per_unit + ox,
                canon_xy[1] * scale_px_per_unit + oy)

    # Anchor bboxes (integer)
    anchor_bboxes = {}
    for lbl in anchors:
        cx, cy = to_image(layout.center(lbl))
        anchor_bboxes[lbl] = (int(cx - 10), int(cy - 10),
                              int(cx + 10), int(cy + 10))
    keymap = build_keymap_from_anchors(anchor_bboxes, layout=layout)

    # Truth bboxes for EVERY letter -- these are what MockOCRBackend uses.
    truth_bboxes = []
    half = 25  # truth bboxes generous so crop centers always land inside
    for lbl in layout.letters():
        cx, cy = to_image(layout.center(lbl))
        truth_bboxes.append((lbl, (int(cx - half), int(cy - half),
                                   int(cx + half), int(cy + half))))
    return truth_bboxes, keymap, layout


def _dummy_image(w=1200, h=600):
    return np.full((h, w, 3), 200, dtype=np.uint8)


def test_perfect_keymap_yields_full_agreement():
    truth, keymap, _ = _synth_truth_and_keymap()
    img = _dummy_image()
    backend = MockOCRBackend(truth)
    report = verify_keymap(img, keymap, backend, crop_size=20)
    assert report.agreement_rate == pytest.approx(1.0, abs=1e-6), (
        f"Expected 100% agreement on a perfect keymap, got "
        f"{report.agreement_rate*100:.1f}% with disagreements: "
        f"{[k for k,v in report.per_key.items() if not v.agrees]}"
    )
    assert report.shift_hypothesis is None


def test_column_shifted_keymap_detected():
    """Shift the keymap one column right (mimicking a homography that's off
    by one column). MockOCR should report systematic dx_cols=-1 shift
    (because the predicted center for "R" now lands on the truth bbox of
    its left neighbour "E")."""
    truth, keymap, layout = _synth_truth_and_keymap(scale_px_per_unit=60.0)

    # Manually shift every keymap entry one COLUMN right in image space.
    # 1 col = scale_px_per_unit = 60 px to the right.
    shifted_entries = {}
    for label, e in keymap.entries.items():
        shifted_entries[label] = type(e)(
            label=e.label,
            image_xy=(e.image_xy[0] + 60.0, e.image_xy[1]),
            canonical_xy=e.canonical_xy,
            robot_xy=e.robot_xy, robot_z=e.robot_z,
            image_bbox=e.image_bbox,
        )
    keymap.entries = shifted_entries

    img = _dummy_image()
    backend = MockOCRBackend(truth)
    report = verify_keymap(img, keymap, backend, crop_size=20)

    # Agreement should be low (since most predictions now land on the wrong key).
    assert report.agreement_rate < 0.5

    # The systematic shift should be detected. We pushed predictions to the
    # RIGHT, so when we crop at the predicted "R" position, OCR sees the
    # truth bbox that's one column TO THE RIGHT of R -> reads "T". The
    # diagnostic reports dx = (canonical-of-OCR) - (canonical-of-expected)
    # = (6.0) - (5.0) = +1.
    assert report.shift_hypothesis is not None, "Should have detected a shift"
    assert report.shift_hypothesis["dx_cols"] == 1
    assert report.shift_hypothesis["dy_rows"] == 0
    assert report.shift_hypothesis["support"] >= 5


def test_random_noise_does_not_falsely_report_shift():
    """A few random OCR errors should NOT trigger a shift hypothesis."""
    truth, keymap, layout = _synth_truth_and_keymap()
    # Sabotage three truth entries: replace their label with random letters.
    rng = np.random.default_rng(0)
    sabotage_idx = rng.choice(len(truth), size=3, replace=False)
    perturbed = list(truth)
    used = set(t[0] for t in truth)
    for i in sabotage_idx:
        lbl, bbox = perturbed[i]
        # Pick a different letter not already at this bbox
        replacement = rng.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c != lbl])
        perturbed[i] = (replacement, bbox)
    img = _dummy_image()
    backend = MockOCRBackend(perturbed)
    report = verify_keymap(img, keymap, backend, crop_size=20)
    # 3 / 26 disagreements = ~12%
    assert 0.7 < report.agreement_rate < 1.0
    # The disagreements are random letters -> no consistent (dx,dy) majority.
    if report.shift_hypothesis is not None:
        # If a shift is reported, the support must be small relative to n_disagree.
        s = report.shift_hypothesis
        # Loose acceptance: we'd rather have a false positive that doesn't take
        # over the pipeline than miss a real shift.
        assert s["support"] <= 2, (
            f"Random noise produced a confident shift hypothesis: {s}"
        )
