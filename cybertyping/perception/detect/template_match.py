"""ORB feature matching against a reference keyboard image.

Seed with either a hand-labelled photo (TemplateMatcher.from_image) or
a synthetic render of the canonical layout (.from_canonical_layout).
detect_anchors() returns {label: bbox} in the same shape as VLM detectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from cybertyping.perception.keymap.layout import Layout, load_layout



def render_canonical_keyboard(
    layout: Layout | None = None,
    scale_px: int = 80,
    margin_px: int = 40,
    bg_color: tuple[int, int, int] = (180, 180, 180),     # gray table
    key_color: tuple[int, int, int] = (40, 40, 40),       # dark keys
    text_color: tuple[int, int, int] = (235, 235, 235),   # bright legends
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """Render a synthetic top-down view of the canonical US keyboard.

    Returns (image_bgr, key_bboxes_in_image).
    """
    layout = layout or load_layout()

    # Compute the layout extent in canonical units to size the canvas.
    max_x = 0.0
    max_y = 0.0
    for k in layout.keys.values():
        max_x = max(max_x, k.x + k.width / 2.0)
        max_y = max(max_y, k.y + k.height / 2.0)

    canvas_w = int(round(max_x * scale_px)) + 2 * margin_px
    canvas_h = int(round(max_y * scale_px)) + 2 * margin_px

    img = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
    bboxes: dict[str, tuple[int, int, int, int]] = {}

    for label, k in layout.keys.items():
        kw = max(0.85, k.width - 0.10)
        kh = max(0.85, k.height - 0.10)
        cx_px = int(round(k.x * scale_px)) + margin_px
        cy_px = int(round(k.y * scale_px)) + margin_px
        hw = int(round(kw * scale_px / 2))
        hh = int(round(kh * scale_px / 2))
        x1, y1 = cx_px - hw, cy_px - hh
        x2, y2 = cx_px + hw, cy_px + hh
        cv2.rectangle(img, (x1, y1), (x2, y2), key_color, thickness=-1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), thickness=1)
        # Draw the label (uppercase for visibility)
        legend = label.upper() if len(label) == 1 else label[:4].upper()
        font_scale = 0.7 if len(legend) == 1 else 0.35
        (tw, th), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tx = cx_px - tw // 2
        ty = cy_px + th // 2
        cv2.putText(img, legend, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, 1, cv2.LINE_AA)
        bboxes[label] = (x1, y1, x2, y2)

    return img, bboxes



@dataclass
class TemplateMatchResult:
    homography_ref_to_input: np.ndarray | None
    n_keypoints_ref: int
    n_keypoints_input: int
    n_matches: int
    n_inliers: int
    inlier_ratio: float


class TemplateMatcher:
    """Find the keyboard in an input image by matching against a reference.

    Uses ORB features and FLANN matching, with RANSAC for the homography
    fit. ORB is rotation-invariant and fast; for very rotated inputs you
    may want to bump `nfeatures` higher.
    """

    def __init__(
        self,
        reference_bgr: np.ndarray,
        key_bboxes_ref: dict[str, tuple[int, int, int, int]],
        nfeatures: int = 4000,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        ratio_test: float = 0.75,
        min_inliers: int = 10,
    ):
        self.reference = reference_bgr
        self.key_bboxes_ref = dict(key_bboxes_ref)
        self.ratio_test = float(ratio_test)
        self.min_inliers = int(min_inliers)

        self._orb = cv2.ORB_create(
            nfeatures=nfeatures, scaleFactor=scale_factor, nlevels=n_levels,
            edgeThreshold=15, fastThreshold=10,
        )
        ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        self._ref_kp, self._ref_desc = self._orb.detectAndCompute(ref_gray, None)
        if self._ref_desc is None:
            raise ValueError("ORB found no keypoints in the reference image.")

        # Brute-force Hamming matcher with cross-check disabled (we need
        # 2-NN matches for the Lowe ratio test).
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    @classmethod
    def from_canonical_layout(cls, scale_px: int = 80, **kwargs) -> "TemplateMatcher":
        """Build a matcher seeded from the synthetic canonical layout."""
        ref_img, bboxes = render_canonical_keyboard(scale_px=scale_px)
        return cls(reference_bgr=ref_img, key_bboxes_ref=bboxes, **kwargs)

    @classmethod
    def from_image(
        cls,
        image_path: str | Path,
        key_bboxes_ref: dict[str, tuple[int, int, int, int]],
        **kwargs,
    ) -> "TemplateMatcher":
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read reference image: {image_path}")
        return cls(reference_bgr=img, key_bboxes_ref=key_bboxes_ref, **kwargs)

    def _detect_input(self, image_bgr: np.ndarray):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return self._orb.detectAndCompute(gray, None)

    def fit(self, image_bgr: np.ndarray) -> TemplateMatchResult:
        """Compute homography from REFERENCE pixel space to INPUT pixel space."""
        in_kp, in_desc = self._detect_input(image_bgr)
        if in_desc is None or len(in_kp) < 4:
            return TemplateMatchResult(None, len(self._ref_kp), 0, 0, 0, 0.0)
        # 2-NN matches per descriptor.
        raw = self._bf.knnMatch(self._ref_desc, in_desc, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good.append(m)
        if len(good) < self.min_inliers:
            return TemplateMatchResult(None, len(self._ref_kp), len(in_kp),
                                       len(good), 0, 0.0)
        src = np.float32([self._ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([in_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return TemplateMatchResult(None, len(self._ref_kp), len(in_kp),
                                       len(good), 0, 0.0)
        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.min_inliers:
            return TemplateMatchResult(None, len(self._ref_kp), len(in_kp),
                                       len(good), inliers, inliers / max(1, len(good)))
        return TemplateMatchResult(
            homography_ref_to_input=H,
            n_keypoints_ref=len(self._ref_kp),
            n_keypoints_input=len(in_kp),
            n_matches=len(good),
            n_inliers=inliers,
            inlier_ratio=inliers / max(1, len(good)),
        )

    def detect_anchors(
        self,
        image_bgr: np.ndarray,
        anchor_labels: Iterable[str] | None = None,
    ) -> dict[str, tuple[int, int, int, int]]:
        """Drop-in replacement for VLMBackend.detect_anchors.

        Returns the projected key bounding boxes in INPUT pixel space.
        """
        result = self.fit(image_bgr)
        if result.homography_ref_to_input is None:
            return {}
        labels = list(anchor_labels) if anchor_labels else list(self.key_bboxes_ref.keys())
        out: dict[str, tuple[int, int, int, int]] = {}
        H = result.homography_ref_to_input
        for label in labels:
            if label not in self.key_bboxes_ref:
                continue
            x1, y1, x2, y2 = self.key_bboxes_ref[label]
            corners_ref = np.float32([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ]).reshape(-1, 1, 2)
            corners_in = cv2.perspectiveTransform(corners_ref, H).reshape(-1, 2)
            xs = corners_in[:, 0]
            ys = corners_in[:, 1]
            bx1, by1 = int(round(xs.min())), int(round(ys.min()))
            bx2, by2 = int(round(xs.max())), int(round(ys.max()))
            h, w = image_bgr.shape[:2]
            bx1 = max(0, min(w - 1, bx1))
            bx2 = max(0, min(w - 1, bx2))
            by1 = max(0, min(h - 1, by1))
            by2 = max(0, min(h - 1, by2))
            if bx2 - bx1 < 2 or by2 - by1 < 2:
                continue
            out[label] = (bx1, by1, bx2, by2)
        return out



if __name__ == "__main__":
    # Render canonical layout, slightly perturb it, recover the homography,
    # and verify Q ends up close to its expected reprojection.
    ref, bboxes = render_canonical_keyboard(scale_px=60)
    h, w = ref.shape[:2]
    # Build a "rotated, translated, scaled" version of the reference as
    # the "input" image.
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 0.92)
    M[0, 2] += 20
    M[1, 2] -= 10
    fake_input = cv2.warpAffine(ref, M, (w + 40, h + 40),
                                borderValue=(180, 180, 180))
    tm = TemplateMatcher(ref, bboxes)
    fit = tm.fit(fake_input)
    print(f"matches={fit.n_matches} inliers={fit.n_inliers} ratio={fit.inlier_ratio:.2f}")
    detected = tm.detect_anchors(fake_input, ["q", "p", "z", "m", "space"])
    for label, bbox in detected.items():
        print(f"  {label}: {bbox}")
