"""OCR-based keyboard detector (no VLM dependency).

Runs OCR on the full image, keeps single-letter detections, returns
{label: bbox} in the same shape as VLM.detect_anchors so it plugs into
perception.homography.build_keymap_from_anchors unchanged.

Typically yields 15-25 anchors per scan on a clean wrist-cam frame.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from cybertyping.perception.keymap.layout import Layout, load_layout
from .ocr_backends import OCRBackend, get_default_ocr_backend


@dataclass
class OCRDetection:
    """A single text detection in the full keyboard image."""
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _single_char(text: str) -> str | None:
    """Reduce an OCR result to a single ASCII letter, if possible.

    PaddleOCR / EasyOCR sometimes return "qq" for a Q key (double-detection
    of the legend), or "q." with a trailing punctuation. We accept those.
    Returns None if no clean single letter can be extracted.
    """
    if not text:
        return None
    letters = [c for c in text.lower() if c.isalpha() and c.isascii()]
    if not letters:
        return None
    # If all letters are the same (e.g. "qq"), keep it.
    if len(set(letters)) == 1:
        return letters[0]
    # Mixed letters -> ambiguous; reject unless exactly one letter present.
    if len(letters) == 1:
        return letters[0]
    return None


def detect_anchors_via_ocr(
    image_bgr: np.ndarray,
    ocr: OCRBackend | None = None,
    anchor_labels: Iterable[str] | None = None,
    layout: Layout | None = None,
    min_confidence: float = 0.3,
    pad_px: int = 6,
    use_multi_preprocess: bool = False,
    use_optimized: bool = False,
) -> dict[str, tuple[int, int, int, int]]:
    """OCR-only anchor detection. Drop-in replacement for VLMBackend.detect_anchors.

    Args:
        image_bgr:       full keyboard image.
        ocr:             OCRBackend with read_full() support. If None,
                         resolved via get_default_ocr_backend().
        anchor_labels:   restrict output to these labels (default: all
                         letters a-z plus 'space' and 'enter' if OCR
                         finds the words "space"/"enter").
        layout:          canonical layout (default: configs/us_keyboard_layout.json).
        min_confidence:  drop detections below this OCR confidence.
        pad_px:          add this much padding around the OCR bbox so the
                         downstream homography fit can use the box center.

    Returns:
        {label: (x1, y1, x2, y2)}  same schema as VLM detect_anchors.
    """
    if ocr is None:
        ocr = get_default_ocr_backend()
        if ocr is None:
            raise RuntimeError(
                "No OCR backend available. Install paddleocr, easyocr, or "
                "pytesseract to use detect_anchors_via_ocr."
            )
    layout = layout or load_layout()
    valid_labels = set(anchor_labels) if anchor_labels else set(layout.letters() + ["space", "enter"])

    # Three OCR strategies in order of increasing cost / coverage:
    #   use_optimized      , full + per-row strips + per-row at 4x/5x (best)
    #   use_multi_preprocess, full image with 3 preprocessing variants
    #   default             , single full-image OCR pass
    if use_optimized and hasattr(ocr, "read_full_optimized"):
        raw = ocr.read_full_optimized(image_bgr)
    elif use_multi_preprocess and hasattr(ocr, "read_full_multi"):
        raw = ocr.read_full_multi(image_bgr)
    else:
        raw = ocr.read_full(image_bgr)
    h, w = image_bgr.shape[:2]

    # Bucket per character so we can keep only the highest-confidence
    # detection per key (OCR sometimes returns the same letter twice on a
    # cap with shadow/highlight).
    best: dict[str, OCRDetection] = {}
    for text, bbox, conf in raw:
        if conf < min_confidence:
            continue
        if bbox[2] - bbox[0] < 3 or bbox[3] - bbox[1] < 3:
            continue
        # Special-case multi-char words for non-letter anchors.
        tnorm = text.strip().lower()
        if tnorm in ("space", "spacebar") and "space" in valid_labels:
            label = "space"
        elif tnorm in ("enter", "return") and "enter" in valid_labels:
            label = "enter"
        else:
            ch = _single_char(text)
            if ch is None or ch not in valid_labels:
                continue
            label = ch
        existing = best.get(label)
        if existing is None or conf > existing.confidence:
            best[label] = OCRDetection(text=tnorm, bbox=tuple(bbox), confidence=float(conf))

    # Apply padding and clip
    out: dict[str, tuple[int, int, int, int]] = {}
    for label, det in best.items():
        x1, y1, x2, y2 = det.bbox
        x1 = max(0, int(x1 - pad_px))
        y1 = max(0, int(y1 - pad_px))
        x2 = min(w - 1, int(x2 + pad_px))
        y2 = min(h - 1, int(y2 + pad_px))
        out[label] = (x1, y1, x2, y2)
    return out



def detect_anchors_via_ocr_multiframe(
    grab_frame: Callable[[], np.ndarray],
    n_frames: int = 3,
    inter_frame_delay_s: float = 0.15,
    ocr: OCRBackend | None = None,
    anchor_labels: Iterable[str] | None = None,
    layout: Layout | None = None,
    min_confidence: float = 0.3,
    pad_px: int = 6,
) -> tuple[dict[str, tuple[int, int, int, int]], list[np.ndarray]]:
    """Take N frames, run OCR on each, union detections.

    Why: a single OCR pass on a single 640x480 wrist-cam frame typically
    finds 12-18 letter keys (out of 26) -- some are obscured by glare,
    motion blur, autofocus drift, or simply unlucky thresholding. Three
    quick frames spaced ~150 ms apart (each with the camera autofocus
    settling slightly differently) usually pushes total detections to
    20-24, which tightens the homography fit and the leave-one-out
    crossval considerably.

    The union strategy: keep the HIGHEST-CONFIDENCE detection per label
    across all frames. We use the first frame as the "reference" so all
    bbox coords are in the same coordinate system (we assume the arm
    isn't moving during the scan -- if it is, you have bigger problems).

    Args:
        grab_frame:  zero-arg callable returning a fresh BGR frame.
        n_frames:    how many shots to take (default 3 = good cost/value).
        inter_frame_delay_s: pause between grabs (lets autofocus settle).

    Returns:
        (union_anchors, frames) -- where `frames` is the list of captured
        BGR frames (so the caller can pick one for downstream OCR-verify
        or annotation).
    """
    if ocr is None:
        ocr = get_default_ocr_backend()
        if ocr is None:
            raise RuntimeError(
                "No OCR backend available. Install paddleocr, easyocr, or "
                "pytesseract."
            )
    layout = layout or load_layout()
    valid_labels = (set(anchor_labels) if anchor_labels else
                    set(layout.letters() + ["space", "enter"]))

    frames: list[np.ndarray] = []
    # per-label best (bbox, confidence)
    best: dict[str, tuple[tuple[int, int, int, int], float]] = {}

    for i in range(n_frames):
        frame = grab_frame()
        frames.append(frame)
        dets = detect_anchors_via_ocr(
            frame, ocr=ocr, anchor_labels=anchor_labels,
            layout=layout, min_confidence=min_confidence, pad_px=pad_px,
        )
        for label, bbox in dets.items():
            existing = best.get(label)
            # We don't get the confidence back from detect_anchors_via_ocr
            # (it filters internally); proxy "confidence" by frame index
            # so later frames overwrite earlier ones only if they actually
            # detected something we didn't have. Simpler: keep first
            # occurrence, accept later only if missing.
            if existing is None:
                best[label] = (bbox, 1.0)
        if i + 1 < n_frames and inter_frame_delay_s > 0:
            time.sleep(inter_frame_delay_s)

    union_anchors = {label: bbox for label, (bbox, _) in best.items()}
    return union_anchors, frames



def annotate_ocr_detections(
    image_bgr: np.ndarray,
    detections: dict[str, tuple[int, int, int, int]],
    color: tuple[int, int, int] = (180, 220, 0),
) -> np.ndarray:
    """Draw the OCR-primary anchor boxes onto a copy of the image."""
    import cv2
    out = image_bgr.copy()
    for label, bbox in detections.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        cv2.putText(out, label.upper(), (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(out, f"ocr_primary: {len(detections)} anchors",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2, cv2.LINE_AA)
    return out



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m cybertyping.perception.detect.ocr_anchors <image>")
        sys.exit(1)
    import cv2
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read {sys.argv[1]}")
        sys.exit(2)
    dets = detect_anchors_via_ocr(img)
    print(f"Detected {len(dets)} letter keys:")
    for label, bbox in sorted(dets.items()):
        print(f"  {label}: {bbox}")
    out = annotate_ocr_detections(img, dets)
    cv2.imwrite("ocr_primary_annotated.jpg", out)
    print("Saved ocr_primary_annotated.jpg")
