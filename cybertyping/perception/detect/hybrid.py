"""VLM-coarse + OCR-refine anchor detection.

Use any coarse detector to identify which key is which, then OCR a
window around each detection to refine the centroid. Fixes the common
VLM failure mode of right-label-wrong-pixel without losing the label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import cv2
import numpy as np

from cybertyping.perception.detect.ocr_backends import OCRBackend, get_default_ocr_backend


class _CoarseDetector(Protocol):
    """Anything that produces a {label: bbox} dict from an image."""
    def detect_anchors(
        self,
        image_bgr: np.ndarray,
        anchor_labels: Iterable[str],
    ) -> dict[str, tuple[int, int, int, int]]: ...


@dataclass
class RefinementStats:
    label: str
    vlm_bbox: tuple[int, int, int, int]
    refined_bbox: tuple[int, int, int, int] | None
    ocr_text: str
    delta_px: float                  # how far the center moved


class HybridRefinedDetector:
    """Wrap a coarse detector with an OCR refinement step.

    Usage:
        from cybertyping.perception.detect.vlm_anchors import QwenBackend
        from cybertyping.perception.detect.ocr_backends import PaddleOCRBackend
        from cybertyping.perception.detect.hybrid import HybridRefinedDetector
        coarse = QwenBackend()
        ocr    = PaddleOCRBackend()
        hybrid = HybridRefinedDetector(coarse, ocr)
        anchors = hybrid.detect_anchors(img, ["q","p","z","m","space"])
    """

    def __init__(
        self,
        coarse: _CoarseDetector,
        ocr: OCRBackend | None = None,
        search_window_factor: float = 3.0,
        min_window_px: int = 80,
        max_window_px: int = 240,
        keep_vlm_if_disagree: bool = True,
    ):
        self._coarse = coarse
        self._ocr = ocr if ocr is not None else get_default_ocr_backend()
        if self._ocr is None:
            raise RuntimeError(
                "HybridRefinedDetector requires an OCR backend. Install "
                "paddleocr, easyocr, or pytesseract."
            )
        self.search_window_factor = float(search_window_factor)
        self.min_window_px = int(min_window_px)
        self.max_window_px = int(max_window_px)
        self.keep_vlm_if_disagree = bool(keep_vlm_if_disagree)
        # last detect call's per-label refinement diagnostics
        self.last_stats: list[RefinementStats] = []

    def _search_window(self, bbox: tuple[int, int, int, int],
                       frame_w: int, frame_h: int
                       ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        win = int(round(max(w, h) * self.search_window_factor))
        win = max(self.min_window_px, min(self.max_window_px, win))
        half = win // 2
        wx1 = int(max(0, round(cx - half)))
        wy1 = int(max(0, round(cy - half)))
        wx2 = int(min(frame_w - 1, round(cx + half)))
        wy2 = int(min(frame_h - 1, round(cy + half)))
        return (wx1, wy1, wx2, wy2)

    def _refine_one(
        self,
        image_bgr: np.ndarray,
        label: str,
        vlm_bbox: tuple[int, int, int, int],
    ) -> RefinementStats:
        h, w = image_bgr.shape[:2]
        wx1, wy1, wx2, wy2 = self._search_window(vlm_bbox, w, h)
        crop = image_bgr[wy1:wy2, wx1:wx2]
        if crop.size == 0:
            return RefinementStats(label, vlm_bbox, None, "", 0.0)

        # Pick the path that exists on the OCR backend.
        if hasattr(self._ocr, "read_full"):
            try:
                detections = self._ocr.read_full(crop)
            except NotImplementedError:
                detections = self._ocr_fallback_crop(crop, label)
        else:
            detections = self._ocr_fallback_crop(crop, label)

        # Pick the detection whose text matches the expected label.
        expected = label.lower()
        matched = []
        for text, bbox, conf in detections:
            text = (text or "").strip().lower()
            if not text:
                continue
            # Letter labels: accept exact one-char match.
            if len(expected) == 1 and expected.isalpha():
                if any(c == expected for c in text):
                    matched.append((text, bbox, conf))
            # Special labels (space, enter): accept the word.
            elif text == expected:
                matched.append((text, bbox, conf))
        if not matched:
            if self.keep_vlm_if_disagree:
                return RefinementStats(label, vlm_bbox, vlm_bbox, "", 0.0)
            return RefinementStats(label, vlm_bbox, None, "", 0.0)

        # Pick the highest-confidence match (or the largest bbox in the
        # presence of confidence ties).
        matched.sort(key=lambda m: (m[2], (m[1][2] - m[1][0]) * (m[1][3] - m[1][1])),
                     reverse=True)
        best_text, best_bbox, _ = matched[0]
        bx1, by1, bx2, by2 = best_bbox
        # Translate the crop-local bbox back into full-image coordinates.
        gx1 = wx1 + bx1
        gy1 = wy1 + by1
        gx2 = wx1 + bx2
        gy2 = wy1 + by2
        # delta = how far the center moved versus the VLM's
        cx_old = (vlm_bbox[0] + vlm_bbox[2]) / 2.0
        cy_old = (vlm_bbox[1] + vlm_bbox[3]) / 2.0
        cx_new = (gx1 + gx2) / 2.0
        cy_new = (gy1 + gy2) / 2.0
        delta = float(np.hypot(cx_new - cx_old, cy_new - cy_old))
        return RefinementStats(
            label=label, vlm_bbox=vlm_bbox,
            refined_bbox=(gx1, gy1, gx2, gy2),
            ocr_text=best_text, delta_px=delta,
        )

    def _ocr_fallback_crop(self, crop: np.ndarray, label: str
                           ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """When read_full is unavailable, run read_crop on the full window
        and synthesize a single detection at the window center."""
        text = self._ocr.read_crop(crop)
        if not text:
            return []
        h, w = crop.shape[:2]
        bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
        return [(text, bbox, 0.5)]

    def detect_anchors(
        self,
        image_bgr: np.ndarray,
        anchor_labels: Iterable[str],
    ) -> dict[str, tuple[int, int, int, int]]:
        labels = list(anchor_labels)
        coarse = self._coarse.detect_anchors(image_bgr, labels)
        out: dict[str, tuple[int, int, int, int]] = {}
        self.last_stats = []
        for label, vlm_bbox in coarse.items():
            stats = self._refine_one(image_bgr, label, vlm_bbox)
            self.last_stats.append(stats)
            if stats.refined_bbox is not None:
                out[label] = stats.refined_bbox
        return out
