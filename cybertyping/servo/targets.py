"""Tip detection and pluggable servo targets.

A *target* answers one question every servo tick: "which pixel should the
tip be driven to right now?" Two implementations exist:

* :class:`ColorDotTarget`: the largest blob of a given colour (Eval 1).
* :class:`KeyTarget`: a keyboard key. Letter anchors are detected in the
  frame (by OCR or by a vision-language model), a homography maps the
  canonical US layout onto the image, and the target key's centre is read
  off that map. Detection is expensive, so it is refreshed every N ticks
  and the last result is cached in between.

Both expose the same small interface used by :mod:`cybertyping.servo.control`::

    target.label            short name for overlays and logs
    target.update(frame)    -> (x, y) pixel or None
    target.draw(frame)      overlay debug info on the frame (in place)
"""

from __future__ import annotations

import os
from typing import Callable, Protocol

import cv2
import numpy as np

from cybertyping.perception.detect.dots import detect_dot
from cybertyping.servo.camera import draw_cross

Pixel = tuple[float, float]
AnchorDetector = Callable[[np.ndarray], dict[str, tuple[int, int, int, int]]]

# Default colour of the marker on the gripper tip. Must not collide with any
# target colour: use yellow for the coloured dots, red for the keyboard.
DEFAULT_TIP_COLOR = "red"

# Reject tip detections that jump more than this between consecutive frames
# (anti-flicker: a glare patch briefly matching the tip colour).
TIP_MAX_JUMP_PX = 150.0

# Pixel distance beyond which an OCR detection of the target key itself is
# considered a mis-read and the homography prediction is used instead.
KEY_MAX_GROSS_PX = 55.0


# ---------------------------------------------------------------------------
# Tip
# ---------------------------------------------------------------------------

def find_tip(frame: np.ndarray | None, color: str,
             prior_xy: Pixel | None = None,
             max_jump_px: float = TIP_MAX_JUMP_PX) -> Pixel | None:
    """Centroid of the largest ``color`` blob, or None.

    ``prior_xy`` enables the frame-to-frame jump rejection.
    """
    if frame is None:
        return None
    pt = detect_dot(frame, color)
    if pt is None:
        return None
    if prior_xy is not None and np.hypot(pt[0] - prior_xy[0], pt[1] - prior_xy[1]) > max_jump_px:
        return None
    return pt


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

class Target(Protocol):
    label: str

    def update(self, frame: np.ndarray | None) -> Pixel | None: ...

    def draw(self, frame: np.ndarray) -> None: ...


class ColorDotTarget:
    """A coloured dot. Detected per frame; the last position persists if the
    dot is momentarily lost (e.g. occluded by the gripper)."""

    def __init__(self, color: str):
        self.color = color
        self.label = color
        self.last_xy: Pixel | None = None

    def update(self, frame: np.ndarray | None) -> Pixel | None:
        pt = find_tip(frame, self.color)   # same HSV detector, no jump filter
        if pt is not None:
            self.last_xy = pt
        return self.last_xy

    def draw(self, frame: np.ndarray) -> None:
        if self.last_xy is not None:
            draw_cross(frame, self.last_xy, (255, 0, 0), 20, 2)


class KeyTarget:
    """A keyboard key located through anchors + homography.

    Parameters
    ----------
    label:          key to press ('a'-'z', 'space', 'enter').
    anchor_detector: callable frame -> {label: bbox}. See
                    :func:`ocr_anchor_detector` / :func:`vlm_anchor_detector`.
    layout:         canonical layout (default: configs/us_keyboard_layout.json).
    refresh_every_ticks: run the detector every N ``update`` calls.
    """

    def __init__(self, label: str, anchor_detector: AnchorDetector,
                 layout=None, refresh_every_ticks: int = 15):
        from cybertyping.perception.keymap.layout import load_layout
        self.label = label
        self.detect_anchors = anchor_detector
        self.layout = layout or load_layout()
        self.refresh_every_ticks = max(1, int(refresh_every_ticks))
        self.target_xy: Pixel | None = None
        self.anchors: dict = {}
        self.keymap = None
        self.status = "pending"
        self._tick = 0
        self._since_refresh = 0

    # -- detection -----------------------------------------------------------

    def _locate(self, frame: np.ndarray) -> None:
        from cybertyping.perception.keymap.homography import build_keymap_from_anchors
        try:
            anchors = self.detect_anchors(frame)
        except Exception as e:
            print(f"[key-target] detector error: {e}")
            self.status = "exception"
            return
        if anchors:
            self.anchors = anchors
        if len(anchors) < 4:
            self.status = "few_anchors"
            return
        try:
            km = build_keymap_from_anchors(anchors, layout=self.layout)
        except Exception as e:
            print(f"[key-target] homography error: {e}")
            self.status = "exception"
            return
        self.keymap = km
        if self.label not in km.entries:
            self.status = "missing_key"
            return
        # Prefer the direct detection of the target key when it agrees with
        # the homography; otherwise trust the homography (OCR mis-read).
        hom_xy = km.entries[self.label].image_xy
        if self.label in anchors:
            x1, y1, x2, y2 = anchors[self.label]
            det_xy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            dist = float(np.hypot(det_xy[0] - hom_xy[0], det_xy[1] - hom_xy[1]))
            self.target_xy = det_xy if dist <= KEY_MAX_GROSS_PX else hom_xy
        else:
            self.target_xy = hom_xy
        self.status = "ok"

    def update(self, frame: np.ndarray | None) -> Pixel | None:
        due = self.target_xy is None or (self._tick % self.refresh_every_ticks == 0)
        self._tick += 1
        if due and frame is not None:
            self._locate(frame)
            self._since_refresh = 0
        else:
            self._since_refresh += 1
        return self.target_xy

    # -- overlay -------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        next_refresh = max(0, self.refresh_every_ticks - self._since_refresh)
        status = (f"anchors={len(self.anchors)} status={self.status} "
                  f"next_refresh={next_refresh}t target={self.label.upper()}")
        cv2.rectangle(frame, (0, h - 26), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, status, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        yellow, cyan, magenta = (0, 255, 255), (255, 255, 0), (255, 0, 255)
        for label, (x1, y1, x2, y2) in self.anchors.items():
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), yellow, 1)
            cv2.putText(frame, label.upper(), (int(x1), max(0, int(y1) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, yellow, 1, cv2.LINE_AA)
        if self.keymap is not None:
            for entry in self.keymap.entries.values():
                draw_cross(frame, entry.image_xy, cyan, 8, 1)
        if self.target_xy is not None:
            draw_cross(frame, self.target_xy, magenta, 28, 3)
            tx, ty = self.target_xy
            cv2.putText(frame, self.label.upper(), (int(tx) + 10, int(ty) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, magenta, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Anchor detectors
# ---------------------------------------------------------------------------

def ocr_anchor_detector(ocr=None, layout=None, min_confidence: float = 0.3,
                        multi_preprocess: bool | None = None) -> AnchorDetector:
    """Anchor detector backed by OCR (Tesseract by default).

    ``multi_preprocess`` runs OCR on three contrast variants and unions the
    results: more anchors, roughly one extra second per call. Defaults to
    the ``OCR_MULTI`` env var.
    """
    from cybertyping.perception.detect.ocr_anchors import detect_anchors_via_ocr
    from cybertyping.perception.detect.ocr_backends import TesseractBackend
    from cybertyping.perception.keymap.layout import load_layout

    ocr = ocr or TesseractBackend()
    layout = layout or load_layout()
    labels = layout.letters() + ["space", "enter"]
    if multi_preprocess is None:
        multi_preprocess = bool(int(os.environ.get("OCR_MULTI", "0")))

    def detect(frame: np.ndarray) -> dict:
        return detect_anchors_via_ocr(
            frame, ocr=ocr, anchor_labels=labels,
            min_confidence=min_confidence,
            use_multi_preprocess=multi_preprocess,
        )
    return detect


def vlm_anchor_detector(backend=None, layout=None,
                        anchor_labels: list[str] | None = None) -> AnchorDetector:
    """Anchor detector backed by a vision-language model.

    Uses :func:`cybertyping.perception.detect.vlm_anchors.get_default_backend`
    (Claude or Gemini via API key, else a local Molmo/Qwen) unless a backend
    is given. Each call is a network round trip of a few seconds, so pair it
    with a large ``refresh_every_ticks`` on :class:`KeyTarget`.
    """
    from cybertyping.perception.detect.vlm_anchors import get_default_backend
    from cybertyping.perception.keymap.layout import load_layout

    backend = backend or get_default_backend()
    layout = layout or load_layout()
    labels = anchor_labels or layout.recommended_anchors()

    def detect(frame: np.ndarray) -> dict:
        return backend.detect_anchors(frame, labels)
    return detect


def normalize_key(ch: str) -> str | None:
    """Map a typed character to a layout label, or None if unmappable."""
    c = ch.lower()
    if c == " ":
        return "space"
    if c == "\n":
        return "enter"
    if len(c) == 1 and "a" <= c <= "z":
        return c
    if c in ("space", "enter"):
        return c
    return None


def tokenize_sentence(sentence: str) -> list[str]:
    """Sentence -> list of key labels; characters outside a-z/space are dropped."""
    out: list[str] = []
    for ch in sentence:
        label = normalize_key(ch)
        if label is not None:
            out.append(label)
    return out
