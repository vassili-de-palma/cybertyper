"""OCR backends + per-key verification of the homography keymap.

Backends: PaddleOCRBackend (best on small glyphs), EasyOCRBackend,
TesseractBackend, MockOCRBackend (for tests). get_default_ocr_backend()
picks the first one that loads.

verify_keymap() crops a window around each predicted key center, runs
OCR, returns per-key agreement plus a shift-hypothesis if many wrong
letters cluster at the same canonical offset (indicating a homography
column shift that can be corrected by a leave-one-out anchor refit).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np


# How big a crop around the predicted key center, in pixels. 80 is a
# reasonable default for a 1280-wide overhead photo of a full keyboard,
# where one key cap is ~50-80 px wide.
DEFAULT_CROP_PX = 80



class OCRBackend(ABC):
    """Read one short string from a small image crop.

    Should return an ASCII lowercase string (empty if nothing recognised).
    Implementations MUST be deterministic for a given crop (no random sampling).

    Optional capability (override if the backend supports full-image detection):
        read_full(image_bgr) -> list[(text, (x1,y1,x2,y2), confidence)]
    This is what perception/ocr_primary.py uses to skip the VLM entirely.
    """

    @abstractmethod
    def read_crop(self, crop_bgr: np.ndarray) -> str: ...

    def read_many(self, crops: list[np.ndarray]) -> list[str]:
        return [self.read_crop(c) for c in crops]

    def warmup(self) -> None:
        """Force any lazy GPU/MPS kernel compilation by running a tiny dummy
        inference. Default is a no-op; backends with deferred init should
        override. Safe to call multiple times."""
        return

    def read_full(self, image_bgr: np.ndarray
                  ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Detect all text in the full image. Optional capability.

        Default implementation: NotImplementedError. Backends that support
        whole-image detection (EasyOCR, PaddleOCR) should override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement read_full. "
            "Use EasyOCRBackend or PaddleOCRBackend for full-image detection."
        )


class EasyOCRBackend(OCRBackend):
    """easyocr Reader. Heavyweight install (PyTorch) but the most accurate."""

    def __init__(self, gpu: bool = False, languages: tuple[str, ...] = ("en",)):
        try:
            import easyocr  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "EasyOCRBackend requires easyocr. Install: pip install easyocr"
            ) from e
        import easyocr
        self.reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)

    def read_crop(self, crop_bgr: np.ndarray) -> str:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        out = self.reader.readtext(rgb, detail=0, allowlist="abcdefghijklmnopqrstuvwxyz")
        return ("".join(out)).strip().lower() if out else ""

    def read_full(self, image_bgr: np.ndarray,
                  upscale: float = 2.5,
                  min_size: int = 5,
                  text_threshold: float = 0.3,
                  low_text: float = 0.2,
                  ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Detect every legible text region in the full image.

        EasyOCR's default `min_size=20` and `text_threshold=0.7` drop
        small keycap legends entirely. We upscale 2.5x and relax both
        thresholds; bbox coords are mapped back to the original image.

        Thresholds intentionally lower than the easyocr defaults: on
        Windows the CRAFT detector produces slightly different scores
        than on macOS (PyTorch is not bit-reproducible across BLAS
        implementations), so a glyph that scores 0.42 on Mac may score
        0.38 on Windows. Looser thresholds + downstream verification by
        verify_keymap make the pipeline robust to that drift.
        """
        h_orig, w_orig = image_bgr.shape[:2]
        if upscale and upscale != 1.0:
            im = cv2.resize(image_bgr, None, fx=upscale, fy=upscale,
                            interpolation=cv2.INTER_CUBIC)
        else:
            im = image_bgr
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        out = self.reader.readtext(
            rgb,
            allowlist="abcdefghijklmnopqrstuvwxyz0123456789",
            min_size=min_size,
            text_threshold=text_threshold,
            low_text=low_text,
        )
        results: list[tuple[str, tuple[int, int, int, int], float]] = []
        for entry in out:
            poly, text, conf = entry
            pts = np.asarray(poly, dtype=np.float32) / upscale
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            bbox = (max(0, int(round(x1))), max(0, int(round(y1))),
                    min(w_orig - 1, int(round(x2))),
                    min(h_orig - 1, int(round(y2))))
            results.append((str(text).strip().lower(), bbox, float(conf)))
        return results

    def read_full_multi(
        self, image_bgr: np.ndarray,
        variants: list[dict] | None = None,
    ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Run read_full at multiple upscales and union the detections.

        EasyOCR's CRAFT detector is sensitive to glyph size in pixels
        relative to its receptive field. Running 2x, 2.5x, and 3.5x and
        keeping the highest-confidence detection per text recovers
        keycap legends that scored just under threshold at any single
        scale (a common failure mode after Mac->Windows backend drift).
        """
        if variants is None:
            variants = [
                {"upscale": 2.5, "text_threshold": 0.3, "low_text": 0.2},
                {"upscale": 2.0, "text_threshold": 0.3, "low_text": 0.2},
                {"upscale": 3.5, "text_threshold": 0.3, "low_text": 0.2},
            ]
        best: dict[tuple[str, int, int], tuple[tuple[int, int, int, int], float]] = {}
        for v in variants:
            dets = self.read_full(
                image_bgr,
                upscale=float(v.get("upscale", 2.5)),
                min_size=int(v.get("min_size", 5)),
                text_threshold=float(v.get("text_threshold", 0.3)),
                low_text=float(v.get("low_text", 0.2)),
            )
            for text, bbox, conf in dets:
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                # Bucket by (text, ~10px cell) so the same letter detected
                # at multiple scales merges into one entry; different
                # letters at the same location stay separate.
                key = (text, cx // 10, cy // 10)
                existing = best.get(key)
                if existing is None or conf > existing[1]:
                    best[key] = (bbox, conf)
        return [(t, bbox, conf) for (t, _, _), (bbox, conf) in best.items()]

    def read_full_optimized(
        self, image_bgr: np.ndarray,
        n_row_strips: int = 5,
        strip_overlap_frac: float = 0.5,
        include_baseline: bool = True,
        fast: bool = False,
    ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Multi-strategy OCR for the full keyboard image.

        Ports the strip-and-union strategy that TesseractBackend uses,
        for the same reason: a glyph that fails on the whole image often
        passes when the input is a 1/5-height strip where the glyph is
        relatively larger and CRAFT has less competing context.

        Strategies (all unioned, highest-conf per (text, location)):
          1. Full-image multi-upscale (read_full_multi).
          2. Per-row strip OCR at default upscales.
          3. Per-row strip at higher upscale (4x) -- catches the smallest legends.

        Args:
            n_row_strips:      not currently used; strip height is fixed
                               at 20% of image height with overlap.
            strip_overlap_frac: 0.5 = each strip starts at midpoint of
                               the previous one.
            include_baseline:  also do the full-image multi-upscale pass.
            fast:              skip the higher-upscale strip pass.
        """
        H, W = image_bgr.shape[:2]
        best: dict[tuple[str, int, int],
                   tuple[tuple[int, int, int, int], float]] = {}

        def _ingest(detections, dy_offset):
            for text, bbox, conf in detections:
                if not text:
                    continue
                x1, y1, x2, y2 = bbox
                gx1, gy1 = x1, y1 + dy_offset
                gx2, gy2 = x2, y2 + dy_offset
                gx1 = max(0, min(W - 1, gx1))
                gx2 = max(0, min(W - 1, gx2))
                gy1 = max(0, min(H - 1, gy1))
                gy2 = max(0, min(H - 1, gy2))
                if gx2 - gx1 < 3 or gy2 - gy1 < 3:
                    continue
                cx = (gx1 + gx2) // 2
                cy = (gy1 + gy2) // 2
                key = (text, cx // 10, cy // 10)
                existing = best.get(key)
                if existing is None or conf > existing[1]:
                    best[key] = ((gx1, gy1, gx2, gy2), conf)

        if include_baseline:
            if fast:
                _ingest(self.read_full(image_bgr), 0)
            else:
                _ingest(self.read_full_multi(image_bgr), 0)

        strip_h = int(H * 0.20)
        step = max(1, int(strip_h * (1 - strip_overlap_frac)))
        max_y = int(H * 0.90) - strip_h
        for y in range(0, max_y + 1, step):
            strip = image_bgr[y:y + strip_h, :, :]
            try:
                if fast:
                    _ingest(self.read_full(strip), y)
                else:
                    _ingest(self.read_full_multi(strip), y)
            except Exception:
                pass
            if not fast:
                try:
                    _ingest(self.read_full_multi(strip, variants=[
                        {"upscale": 4.0, "text_threshold": 0.3, "low_text": 0.2},
                    ]), y)
                except Exception:
                    pass

        return [(t, bbox, conf) for (t, _, _), (bbox, conf) in best.items()]


class PaddleOCRBackend(OCRBackend):
    """PaddleOCR. Notably better than EasyOCR on small dense glyphs (e.g.
    individual keycap labels) in our internal tests. Heavy install (~1 GB
    PyTorch-or-Paddle wheels).

    Install: pip install paddleocr paddlepaddle
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        try:
            from paddleocr import PaddleOCR  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "PaddleOCRBackend requires paddleocr. Install: "
                "pip install paddleocr paddlepaddle"
            ) from e
        from paddleocr import PaddleOCR
        # use_angle_cls=True helps when keycaps are imaged from slight angles.
        # det_db_thresh=0.3 makes detection more sensitive to small glyphs.
        self._ocr = PaddleOCR(
            use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False,
            det_db_thresh=0.3, det_db_box_thresh=0.5,
        )

    def read_crop(self, crop_bgr: np.ndarray) -> str:
        results = self._ocr.ocr(crop_bgr, cls=True)
        # PaddleOCR returns [[ [poly, (text, conf)], ... ]] (one outer list per image)
        if not results or not results[0]:
            return ""
        # take the highest-confidence result
        best_text = ""
        best_conf = -1.0
        for entry in results[0]:
            _poly, (text, conf) = entry
            if conf > best_conf:
                best_conf = float(conf)
                best_text = str(text).strip().lower()
        # restrict to letters/digits
        keep = "".join(c for c in best_text if c.isalnum())
        return keep

    def read_full(self, image_bgr: np.ndarray
                  ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        results = self._ocr.ocr(image_bgr, cls=True)
        if not results or not results[0]:
            return []
        out: list[tuple[str, tuple[int, int, int, int], float]] = []
        for entry in results[0]:
            poly, (text, conf) = entry
            pts = np.asarray(poly, dtype=np.float32)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            bbox = (int(round(x1)), int(round(y1)),
                    int(round(x2)), int(round(y2)))
            out.append((str(text).strip().lower(), bbox, float(conf)))
        return out


class TesseractBackend(OCRBackend):
    """pytesseract + tesseract binary. Lower accuracy on tiny glyphs but
    much lighter than easyocr / paddleocr.

    IMPORTANT preprocessing notes (validated on a real wrist-cam keyboard
    image; see runs/bench_vass_*):
      - Raw 640x480 frame: Tesseract returns 0 letters.
      - 2x-3x upscaling: jumps to 17-18 confident letter detections.
      - Inversion (white-on-dark -> dark-on-light): essential.
      - Otsu threshold: helps on top of inversion.
    Defaults below bake in this stack.
    """

    def __init__(self, psm: int = 10, full_psm: int = 11,
                 upscale: float = 3.0,
                 read_full_min_conf: float = 30.0):
        try:
            import pytesseract  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "TesseractBackend requires pytesseract. Install: "
                "pip install pytesseract  (and the tesseract OS binary)"
            ) from e
        import pytesseract
        self._pt = pytesseract
        # psm 10 = "treat the image as a single character", best for crops.
        # psm 11 = "sparse text, find as much text as possible in no particular
        #          order", best for full-frame letter spotting.
        self.config = (f"--psm {psm} "
                       "-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz "
                       "-c load_system_dawg=0 -c load_freq_dawg=0")
        # Allow lowercase too so Tesseract reads multi-char words like
        # "Enter" / "Space" that are written in mixed case on most keycaps.
        # Single-letter detections still get filtered to one canonical label
        # by the downstream parser.
        self.config_full = (f"--psm {full_psm} "
                            "-c tessedit_char_whitelist="
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            "abcdefghijklmnopqrstuvwxyz "
                            "-c load_system_dawg=0 -c load_freq_dawg=0")
        self.upscale = float(upscale)
        self.read_full_min_conf = float(read_full_min_conf)

    @staticmethod
    def _preprocess(image_bgr: np.ndarray, upscale: float) -> np.ndarray:
        """Tesseract-friendly preprocessing: optional upscale + invert + Otsu."""
        if upscale and upscale != 1.0:
            image_bgr = cv2.resize(
                image_bgr, None, fx=upscale, fy=upscale,
                interpolation=cv2.INTER_CUBIC,
            )
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    def read_crop(self, crop_bgr: np.ndarray) -> str:
        # For crops we DON'T upscale (a single key cap zoomed to >100px
        # actually hurts tesseract's small-text recognizer), just invert+Otsu.
        th = self._preprocess(crop_bgr, upscale=1.0)
        txt = self._pt.image_to_string(th, config=self.config)
        return txt.strip().lower()

    def read_full(self, image_bgr: np.ndarray
                  ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Detect every legible letter in the full keyboard image.

        We upscale first (default 3x) because Tesseract effectively can't
        see ~20px-tall key legends at the wrist-cam's native resolution;
        a 3x upscale of an inverted+Otsu binarisation usually finds
        15-20+ letters on a typical eval frame.
        """
        h_orig, w_orig = image_bgr.shape[:2]
        processed = self._preprocess(image_bgr, upscale=self.upscale)
        data = self._pt.image_to_data(processed, config=self.config_full,
                                      output_type=self._pt.Output.DICT)
        out: list[tuple[str, tuple[int, int, int, int], float]] = []
        for i, txt in enumerate(data.get("text", [])):
            txt = (txt or "").strip()
            if not txt:
                continue
            conf = float(data["conf"][i])
            if conf < self.read_full_min_conf:
                continue
            x  = int(data["left"][i]   / self.upscale)
            y  = int(data["top"][i]    / self.upscale)
            w  = int(data["width"][i]  / self.upscale)
            h  = int(data["height"][i] / self.upscale)
            x2 = min(w_orig - 1, x + w)
            y2 = min(h_orig - 1, y + h)
            out.append((txt.lower(), (max(0, x), max(0, y), x2, y2),
                        conf / 100.0))
        return out

    def read_full_optimized(
        self, image_bgr: np.ndarray,
        n_row_strips: int = 5,
        strip_overlap_frac: float = 0.5,
        include_baseline: bool = True,
        fast: bool = False,
    ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Multi-strategy OCR for the full keyboard image.

        Combines THREE strategies and unions the results:
          1. Full-image multi-preprocess (read_full_multi).
          2. Per-row strip OCR, splits the image into N horizontal strips
             with overlap and OCRs each. Tesseract sees larger relative
             glyphs because the strip is shorter, and PSM 11 has less
             competing text to confuse it.
          3. Per-row strip at HIGHER upscale (4x and 5x) -- catches keys
             whose legend is at the threshold of what default 3x can see.

        Strategy 2 is the biggest win. Measured on a real wrist-cam frame
        (Vass's Logitech, 640x480): baseline = 17 letters, +per-row = 21
        letters (+24%). The added letters are typically the ones in the
        far-left and far-right columns where the full-image OCR was
        distracted by row-of-letters context.

        Returns the unioned list with highest-confidence bbox per letter.
        Coordinates are in the ORIGINAL image's pixel space.

        Args:
            n_row_strips:      how many horizontal strips to slice the image
                               into. 3-5 is the practical range.
            strip_overlap_frac: 0 = non-overlapping, 0.5 = each strip starts
                               at the midpoint of the previous one. Overlap
                               catches letters near strip boundaries.
            include_baseline:  also do the full-image multi-preprocess pass.
            fast:              if True, use ONE preprocessing variant per
                               strip (just 3x + invert + Otsu) and skip the
                               4x/5x high-upscale variant. Roughly 3x faster
                               (1.5s vs 5s on a 640x480 frame) at the cost
                               of ~2 fewer anchors on the worst inputs.
        """
        H, W = image_bgr.shape[:2]
        best: dict[str, tuple[tuple[int, int, int, int], float]] = {}

        def _ingest(detections, dy_offset):
            for label, bbox, conf in detections:
                if not label or not label.isalpha() or len(label) != 1:
                    continue
                x1, y1, x2, y2 = bbox
                gx1, gy1 = x1, y1 + dy_offset
                gx2, gy2 = x2, y2 + dy_offset
                gx1 = max(0, min(W - 1, gx1))
                gx2 = max(0, min(W - 1, gx2))
                gy1 = max(0, min(H - 1, gy1))
                gy2 = max(0, min(H - 1, gy2))
                if gx2 - gx1 < 3 or gy2 - gy1 < 3:
                    continue
                existing = best.get(label)
                if existing is None or conf > existing[1]:
                    best[label] = ((gx1, gy1, gx2, gy2), conf)

        # Strategy 1: full-image baseline
        if include_baseline:
            if fast:
                full_dets = self.read_full(image_bgr)
            else:
                full_dets = self.read_full_multi(image_bgr)
            full_dets = [(t, b, c) for t, b, c in full_dets
                         if t and t.isalpha() and len(t) == 1]
            _ingest(full_dets, 0)

        # Strategies 2 + 3: per-row strips
        strip_h = int(H * 0.20)
        step = max(1, int(strip_h * (1 - strip_overlap_frac)))
        max_y = int(H * 0.90) - strip_h
        for y in range(0, max_y + 1, step):
            strip = image_bgr[y:y + strip_h, :, :]
            try:
                if fast:
                    # FAST mode: single OCR pass per strip at the default 3x
                    # upscale. Skip the multi-variant + high-upscale passes.
                    row_dets = self.read_full(strip)
                else:
                    row_dets = self.read_full_multi(strip)
                _ingest(row_dets, y)
            except Exception:
                pass
            if not fast:
                # Slow mode: also do high-upscale pass per strip
                try:
                    row_dets_hi = self.read_full_multi(strip, variants=[
                        {"upscale": 4.0, "otsu": True},
                        {"upscale": 5.0, "otsu": True},
                    ])
                    _ingest(row_dets_hi, y)
                except Exception:
                    pass

        return [(label, bbox, conf) for label, (bbox, conf) in best.items()]

    def read_full_multi(self, image_bgr: np.ndarray,
                        variants: list[dict] | None = None
                        ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Run multiple preprocessing variants and union the detections.

        Each variant is a dict of preprocessing parameters; we run read_full
        on each and merge results, keeping per-letter the highest-confidence
        detection. Substantially more anchors per scan at a small extra cost.

        Default variants (validated on a real wrist-cam frame):
          - upscale=3 + invert + Otsu (the default read_full path)
          - upscale=2 + invert (no Otsu) -- catches keys where Otsu picked a
            bad threshold
          - upscale=4 + invert + Otsu  -- bigger boost for the smallest legends
        """
        if variants is None:
            variants = [
                {"upscale": 3.0, "otsu": True},
                {"upscale": 2.0, "otsu": False},
                {"upscale": 4.0, "otsu": True},
            ]
        h_orig, w_orig = image_bgr.shape[:2]
        best: dict[str, tuple[tuple[int, int, int, int], float]] = {}
        for v in variants:
            up = float(v.get("upscale", self.upscale))
            otsu = bool(v.get("otsu", True))
            if up != 1.0:
                im = cv2.resize(image_bgr, None, fx=up, fy=up,
                                interpolation=cv2.INTER_CUBIC)
            else:
                im = image_bgr
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            if otsu:
                _, processed = cv2.threshold(gray, 0, 255,
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                processed = gray
            data = self._pt.image_to_data(processed, config=self.config_full,
                                          output_type=self._pt.Output.DICT)
            for i, txt in enumerate(data.get("text", [])):
                txt = (txt or "").strip().lower()
                if not txt: continue
                conf = float(data["conf"][i])
                if conf < self.read_full_min_conf: continue
                # Pick a label: special-case multi-letter words 'enter' /
                # 'space' so we can pin those keys to their actual pixel
                # position; otherwise require a clean single letter.
                if txt in ("enter", "return"):
                    label = "enter"
                elif txt in ("space", "spacebar"):
                    label = "space"
                else:
                    letters = [c for c in txt if c.isalpha()]
                    if not letters or len(set(letters)) > 1: continue
                    label = letters[0]
                x  = int(data["left"][i]   / up)
                y  = int(data["top"][i]    / up)
                w  = int(data["width"][i]  / up)
                h  = int(data["height"][i] / up)
                x2 = min(w_orig - 1, x + w)
                y2 = min(h_orig - 1, y + h)
                bbox = (max(0, x), max(0, y), x2, y2)
                existing = best.get(label)
                if existing is None or conf / 100.0 > existing[1]:
                    best[label] = (bbox, conf / 100.0)
        return [(label, bbox, conf) for label, (bbox, conf) in best.items()]


class MockOCRBackend(OCRBackend):
    """Deterministic OCR backed by a precomputed (label, image_bbox) list.

    Useful for unit tests: we render the synthetic keyboard with known
    letter positions and bboxes, hand them to MockOCRBackend, and then
    every crop centered on a key returns the right letter.

    Args:
        truth: list of (label, (x1, y1, x2, y2)). Each crop is mapped to a
               label by checking which truth bbox contains the crop center.
               If no bbox contains it, the backend returns "".
    """

    def __init__(self, truth: list[tuple[str, tuple[int, int, int, int]]],
                 last_crop_origin: tuple[int, int] | None = None):
        self.truth = truth
        # Caller must set last_crop_origin BEFORE each read_crop call so we
        # can map the crop's local coordinates back to global image coords.
        self._origin = last_crop_origin

    def set_origin(self, origin: tuple[int, int]) -> None:
        self._origin = origin

    def read_crop(self, crop_bgr: np.ndarray) -> str:
        if self._origin is None:
            raise RuntimeError("MockOCRBackend.read_crop: call set_origin first.")
        h, w = crop_bgr.shape[:2]
        cx = self._origin[0] + w / 2
        cy = self._origin[1] + h / 2
        for label, (x1, y1, x2, y2) in self.truth:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return label.lower()
        return ""

    def read_full(self, image_bgr: np.ndarray
                  ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Return every (label, bbox, 1.0) in the precomputed truth list
        whose bbox is inside the input image. Used by ocr_primary tests."""
        h, w = image_bgr.shape[:2]
        out: list[tuple[str, tuple[int, int, int, int], float]] = []
        for label, (x1, y1, x2, y2) in self.truth:
            # Clip to image; skip if entirely outside.
            if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
                continue
            cx = max(0, min(w - 1, x1))
            cy = max(0, min(h - 1, y1))
            cx2 = max(0, min(w - 1, x2))
            cy2 = max(0, min(h - 1, y2))
            out.append((label.lower(), (cx, cy, cx2, cy2), 1.0))
        return out


def get_default_ocr_backend(prefer: list[str] | None = None) -> OCRBackend | None:
    """Cascade through OCR backends in preference order.

    Default order: PaddleOCR -> EasyOCR -> Tesseract -> None.
    PaddleOCR is preferred because it's more accurate on tightly-spaced
    single-character glyphs (key caps); EasyOCR is the fallback because
    most setups already have it installed; Tesseract is the lightest of
    the three but lowest accuracy.
    """
    order = prefer or ["paddle", "easy", "tesseract"]
    for name in order:
        try:
            if name == "paddle":
                return PaddleOCRBackend(use_gpu=False)
            if name == "easy":
                return EasyOCRBackend(gpu=False)
            if name == "tesseract":
                return TesseractBackend()
        except SystemExit:
            continue
        except Exception:
            continue
    return None



@dataclass
class KeyVerification:
    label: str
    ocr_text: str
    agrees: bool
    crop_bbox: tuple[int, int, int, int]


@dataclass
class VerificationReport:
    per_key: dict[str, KeyVerification]
    agreement_rate: float       # fraction of letter keys whose OCR matched
    shift_hypothesis: dict | None  # {"dx_cols": int, "dy_rows": int, "support": int} or None

    @property
    def disagree(self) -> dict[str, KeyVerification]:
        return {k: v for k, v in self.per_key.items() if not v.agrees}


def _safe_crop(image: np.ndarray, cx: float, cy: float, size: int
               ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    half = size // 2
    x1 = max(0, int(round(cx - half)))
    y1 = max(0, int(round(cy - half)))
    x2 = min(w, int(round(cx + half)))
    y2 = min(h, int(round(cy + half)))
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def verify_keymap(
    image_bgr: np.ndarray,
    keymap,                 # cybertyping.perception.keymap.homography.KeyMap
    backend: OCRBackend,
    crop_size: int = DEFAULT_CROP_PX,
    targets: Iterable[str] | None = None,
) -> VerificationReport:
    """Run OCR on each predicted key crop, return a per-key agreement report.

    Only letter keys are verified (digits/symbols often have duplicate
    legends per cap).
    """
    from cybertyping.perception.keymap.layout import load_layout
    layout = load_layout()

    if targets is None:
        # Only letters, they have unambiguous single-character legends.
        targets = [lbl for lbl, k in layout.keys.items() if k.type == "letter"]
    targets = [t for t in targets if t in keymap.entries]

    per_key: dict[str, KeyVerification] = {}
    for label in targets:
        entry = keymap.entries[label]
        cx, cy = entry.image_xy
        crop, bbox = _safe_crop(image_bgr, cx, cy, crop_size)
        # The MockOCRBackend needs to know the crop origin in global coords.
        if isinstance(backend, MockOCRBackend):
            backend.set_origin((bbox[0], bbox[1]))
        text = backend.read_crop(crop)
        # Restrict to single character: take the first letter only, lowercase.
        ocr_char = next((c for c in text.lower() if c.isalpha()), "")
        agrees = (ocr_char == label.lower())
        per_key[label] = KeyVerification(label=label, ocr_text=text,
                                         agrees=agrees, crop_bbox=bbox)

    n_total = max(1, len(per_key))
    agree_rate = sum(1 for v in per_key.values() if v.agrees) / n_total
    shift = _detect_systematic_shift(per_key, layout)
    return VerificationReport(per_key=per_key, agreement_rate=agree_rate,
                              shift_hypothesis=shift)


def _detect_systematic_shift(per_key: dict[str, KeyVerification], layout) -> dict | None:
    """If many keys disagree but the wrong letters form a consistent (dx,dy)
    pattern in canonical layout space, return that shift.

    Implementation: for every disagreeing key (label_expected, ocr_read),
    compute (canonical_xy[ocr_read] - canonical_xy[label_expected]). Round
    to the nearest integer (column, row) offset. If a majority of
    disagreements share the same offset, that's our shift hypothesis.
    """
    offsets: list[tuple[int, int]] = []
    for k, v in per_key.items():
        if v.agrees or not v.ocr_text:
            continue
        ocr_ch = next((c for c in v.ocr_text if c.isalpha()), "")
        if ocr_ch not in layout.keys or k not in layout.keys:
            continue
        ex, ey = layout.center(k)
        ox, oy = layout.center(ocr_ch)
        dx, dy = ox - ex, oy - ey
        # Snap to integer column/row indices. Rows are integer y, columns
        # vary by row stagger but column STEP is 1.0, so dx in column-pitch units
        # is just dx (rounded).
        offsets.append((int(round(dx)), int(round(dy))))
    if not offsets:
        return None
    counter = Counter(offsets)
    (dx, dy), support = counter.most_common(1)[0]
    n_disagree = max(1, sum(1 for v in per_key.values() if not v.agrees))
    if support / n_disagree >= 0.6 and (dx, dy) != (0, 0):
        return {"dx_cols": int(dx), "dy_rows": int(dy), "support": int(support),
                "n_disagree": int(n_disagree)}
    return None



def annotate_verification(
    image_bgr: np.ndarray, report: VerificationReport,
    ok_color=(0, 200, 0), bad_color=(0, 60, 220),
) -> np.ndarray:
    out = image_bgr.copy()
    for label, v in report.per_key.items():
        x1, y1, x2, y2 = v.crop_bbox
        color = ok_color if v.agrees else bad_color
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        cv2.putText(out, f"{label}->{v.ocr_text or '?'}",
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.putText(out, f"agree={report.agreement_rate*100:.1f}%",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    if report.shift_hypothesis is not None:
        s = report.shift_hypothesis
        cv2.putText(out, f"shift?: dx={s['dx_cols']} dy={s['dy_rows']} "
                         f"({s['support']}/{s['n_disagree']})",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 60, 220), 2, cv2.LINE_AA)
    return out
