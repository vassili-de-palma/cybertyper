"""Keyboard outline detection, pure-geometry fallback when VLM fails.

Use cases:
  (A) Sanity check the VLM anchor positions: every anchor pixel should
      lie inside the detected keyboard rectangle.
  (B) Complete fallback when the VLM returns < 4 anchors. We can still
      build a rough homography from the rectangle corners alone if we
      know the canonical bounding box of the layout (the corners of the
      letter region).

The eval setup makes this easy: black keyboard on light-gray table
(#B8ADA9). The keyboard is the darkest connected region in the frame.

Algorithm:
    1. Grayscale + Gaussian blur.
    2. Otsu's threshold (inverted: keyboard pixels = white, table = black).
    3. Morphological close to fill key gaps.
    4. Largest external contour.
    5. minAreaRect -> 4 corner points + angle.
    6. Order corners as TL, TR, BR, BL.

If the keyboard isn't the largest dark region (e.g. the arm is in shadow,
the operator's hand is visible), pass `min_area_frac=0.2` or higher to
filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class KeyboardOutline:
    corners: np.ndarray             # (4, 2) ordered TL, TR, BR, BL
    angle_deg: float                # rotation of the rect in image
    area_px: float
    score: float                    # contour area / bounding rect area (rectangularity)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 unsorted (x,y) corners as TL, TR, BR, BL.

    Use the standard sum/diff trick: TL has smallest sum, BR has largest;
    TR has smallest diff (x-y), BL has largest.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.stack([tl, tr, br, bl])


def detect_keyboard_outline(
    image_bgr: np.ndarray,
    min_area_frac: float = 0.05,
    blur_ksize: int = 5,
) -> KeyboardOutline | None:
    """Find the largest dark rectangular region. Returns None if no plausible
    region is found.

    Args:
        min_area_frac: minimum contour area as a fraction of the full image.
                       Filters out small dark blobs (shadows, debris).
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    # Otsu, inverted so the keyboard (dark) becomes white.
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    image_area = float(h * w)
    best: KeyboardOutline | None = None
    best_area = -1.0
    for c in contours:
        a = float(cv2.contourArea(c))
        if a < min_area_frac * image_area:
            continue
        rect = cv2.minAreaRect(c)            # ((cx,cy),(w,h),angle)
        rw, rh = rect[1]
        if min(rw, rh) < 1:
            continue
        rect_area = rw * rh
        rectangularity = a / rect_area
        if rectangularity < 0.6:
            continue                          # too non-rectangular
        if a > best_area:
            best_area = a
            corners = _order_corners(cv2.boxPoints(rect))
            best = KeyboardOutline(
                corners=corners,
                angle_deg=float(rect[2]),
                area_px=a,
                score=float(rectangularity),
            )
    return best



def homography_from_outline(
    outline: KeyboardOutline,
    canonical_tl: tuple[float, float] = (1.5, 1.5),
    canonical_tr: tuple[float, float] = (11.0, 1.5),
    canonical_br: tuple[float, float] = (8.75, 3.5),
    canonical_bl: tuple[float, float] = (2.75, 3.5),
) -> np.ndarray:
    """Build a rough homography assuming the detected rectangle corners
    correspond to the LETTER REGION corners of the canonical layout
    (Q top-left, P top-right, M bottom-right, Z bottom-left by default).

    This is a coarse fallback ONLY. Real keyboard rectangles include the
    function row, modifier rows, etc., so the rectangle corners overshoot
    the letter region by ~1 row + ~1.5 column on every side. The caller
    can compensate via the canonical_* arguments to match what their
    keyboard photo actually looks like.

    Returns 3x3 H_canon_to_image.
    """
    src = np.array([canonical_tl, canonical_tr, canonical_br, canonical_bl],
                   dtype=np.float64)
    dst = outline.corners.astype(np.float64)
    H = cv2.getPerspectiveTransform(src.astype(np.float32),
                                    dst.astype(np.float32))
    return H


def annotate_outline(image_bgr: np.ndarray,
                     outline: KeyboardOutline | None,
                     color=(0, 200, 255)) -> np.ndarray:
    out = image_bgr.copy()
    if outline is None:
        cv2.putText(out, "No keyboard outline found",
                    (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)
        return out
    pts = outline.corners.astype(int).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
    for i, (x, y) in enumerate(outline.corners):
        cv2.putText(out, ["TL", "TR", "BR", "BL"][i], (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(out, f"score={outline.score:.2f} area={int(outline.area_px)}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2, cv2.LINE_AA)
    return out
