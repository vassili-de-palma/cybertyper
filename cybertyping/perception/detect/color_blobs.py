"""Eval 1 colored-dots detector.

The brief specifies FOUR colored circles, ~1.5 cm diameter, on an A4 sheet:
    red    #FF0000
    green  #00FF00
    blue   #0000FF
    cyan   #7FFFFF

Goal: return their centroids in the image (and, downstream, in robot XY).
Pure classical CV, no VLM. Has to be the most reliable points-on-table
hit the robot can make.

Pipeline:
    1. Find the A4 paper as the largest bright low-saturation region.
       Crop to it. Everything outside the paper (gray table, shadows,
       glare on the floor) is now ineligible for false positives.
    2. Downsample the crop ~2x for fast connected-components localization.
    3. For each color, mask via BGR channel arithmetic (e.g. blue dot ==
       "B - max(R,G) >= 60"). This is per-pixel *relative*, so a
       white-balance shift that lifts every channel doesn't change which
       channel dominates -- the test stays valid. Much more robust than
       absolute HSV bands.
    4. Largest blob per color in the downsampled mask -> approximate bbox.
    5. Re-mask the FULL-RESOLUTION pixels inside that bbox, take the
       moments centroid on the actual connected component. Sub-pixel
       accuracy, full-res precision.

The classic HSV path remains available by passing a `bands` dict to
`detect_dots` -- only used when an operator wants to tune live for an
unusual lighting condition. The BGR path is the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Legacy HSV bands (kept for the optional `bands=` override).
# OpenCV HSV ranges (H: 0..179, S/V: 0..255).
#
# Reference colors and their HSV (OpenCV scale):
#   #FF0000  red     -> H=0   (wraps), S=255, V=255
#   #00FF00  green   -> H=60,         S=255, V=255
#   #0000FF  blue    -> H=120,        S=255, V=255
#   #7FFFFF  cyan    -> H=90,         S=128, V=255   (only HALF saturation)
# ---------------------------------------------------------------------------
DEFAULT_BANDS = {
    "red":   [((0,   80, 80), (10,  255, 255)),
              ((170, 80, 80), (179, 255, 255))],     # wraps around 0
    "green": [((38,  60, 60), (85,  255, 255))],
    "blue":  [((100, 130, 100), (130, 255, 255))],
    "cyan":  [((85,  70, 200), (95,  255, 255))],
}

# Eval rubric order:
EVAL1_DOT_ORDER = ["red", "green", "blue", "cyan"]

MIN_BLOB_AREA_PX = 60     # ~7px radius. A4@~30cm tends to give 100-300 px per dot.
MAX_BLOB_AREA_PX = 8000   # filter out giant glare regions

# Paper detection: every channel bright, low chroma.
# Floor=200 leaves a clear gap above the spec'd gray table (#B8ADA9, mean
# channel ~175) so the table doesn't masquerade as paper. White A4 in
# normal lighting reads 230-250 per channel; even mildly off-white print
# paper clears 200.
PAPER_BRIGHTNESS_FLOOR = 200
PAPER_CHROMA_CEIL = 30
# Reject "paper" candidates that cover less than this fraction of the frame.
PAPER_MIN_AREA_FRAC = 0.04


@dataclass
class Dot:
    color: str
    pixel_xy: tuple[float, float]
    area_px: float
    bbox: tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# BGR channel-arithmetic masks (the primary, lighting-invariant path).
# ---------------------------------------------------------------------------

def _bgr_masks_all(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """All four color masks in one pass over the BGR image.

    Each mask is a uint8 array (0 / 255). Tests are per-pixel relative,
    so a white-balance drift that lifts every channel by the same amount
    doesn't break detection -- only the relative dominance matters.
    """
    b, g, r = cv2.split(image_bgr)
    b = b.astype(np.int16); g = g.astype(np.int16); r = r.astype(np.int16)
    max_gb = np.maximum(g, b)
    max_rb = np.maximum(r, b)
    max_rg = np.maximum(r, g)
    min_gb = np.minimum(g, b)

    # red #FF0000: R dominates G and B by a wide margin, R is bright.
    red = ((r - max_gb) >= 60) & (r >= 100)
    # green #00FF00: G dominates.
    green = ((g - max_rb) >= 60) & (g >= 100)
    # blue #0000FF: B dominates. Previously dim blue dots slipped past the
    # HSV V>=60 floor; this test fires as long as B is meaningfully larger
    # than R and G, even on a poorly lit / dark-blue print.
    blue = ((b - max_rg) >= 60) & (b >= 100)
    # pale cyan #7FFFFF: G and B both high, R noticeably lower. Distinct
    # from pure cyan (would require R near 0), pure white (R~=G~=B), and
    # blue/green dots (one of G,B would be near 0).
    cyan = ((min_gb - r) >= 50) & (g >= 180) & (b >= 180) & (r >= 60)

    return {
        "red":   red.astype(np.uint8)   * 255,
        "green": green.astype(np.uint8) * 255,
        "blue":  blue.astype(np.uint8)  * 255,
        "cyan":  cyan.astype(np.uint8)  * 255,
    }


def _bgr_mask_one(image_bgr: np.ndarray, color: str) -> np.ndarray:
    """Same as `_bgr_masks_all` but returns just one mask. Used for the
    full-resolution refinement pass where we only need one color at a time.
    """
    return _bgr_masks_all(image_bgr)[color]


# ---------------------------------------------------------------------------
# Paper ROI: largest bright + low-chroma region.
# ---------------------------------------------------------------------------

def _find_paper_roi(image_bgr: np.ndarray, pad_px: int = 16,
                    ) -> tuple[int, int, int, int] | None:
    """Bounding box of the largest paper-like region, padded.

    Paper criterion: every BGR channel >= PAPER_BRIGHTNESS_FLOOR AND
    chroma (max - min channel) <= PAPER_CHROMA_CEIL. This is invariant to
    a uniform white-balance shift and rejects the gray table (chroma low
    but channels typically < 180) and saturated dot pixels (chroma high).

    Returns (x0, y0, x1, y1) in full-frame pixel coords, or None if no
    component is large enough.
    """
    b, g, r = cv2.split(image_bgr)
    bright = (b >= PAPER_BRIGHTNESS_FLOOR) & \
             (g >= PAPER_BRIGHTNESS_FLOOR) & \
             (r >= PAPER_BRIGHTNESS_FLOOR)
    bi = b.astype(np.int16); gi = g.astype(np.int16); ri = r.astype(np.int16)
    chroma = np.maximum(np.maximum(bi, gi), ri) - \
             np.minimum(np.minimum(bi, gi), ri)
    paper = (bright & (chroma <= PAPER_CHROMA_CEIL)).astype(np.uint8) * 255

    # Fill in the dots-on-paper holes so the paper shows up as one solid
    # blob (otherwise four ~1.5cm dots cut the paper into a swiss cheese).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, k, iterations=3)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN,  k, iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=8)
    if n <= 1:
        return None
    H, W = image_bgr.shape[:2]
    min_area = PAPER_MIN_AREA_FRAC * H * W
    best_i, best_area = -1, 0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        if a > best_area:
            best_i, best_area = i, a
    if best_i < 0:
        return None
    x = int(stats[best_i, cv2.CC_STAT_LEFT])
    y = int(stats[best_i, cv2.CC_STAT_TOP])
    w = int(stats[best_i, cv2.CC_STAT_WIDTH])
    h = int(stats[best_i, cv2.CC_STAT_HEIGHT])
    x0 = max(0, x - pad_px)
    y0 = max(0, y - pad_px)
    x1 = min(W, x + w + pad_px)
    y1 = min(H, y + h + pad_px)
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Legacy HSV mask helper (only used when caller passes `bands=`).
# ---------------------------------------------------------------------------

def _mask_color(hsv: np.ndarray, bands: list[tuple]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in bands:
        mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                            np.array(hi, dtype=np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def detect_dots(
    image_bgr: np.ndarray,
    bands: dict[str, list[tuple]] | None = None,
    *,
    use_paper_roi: bool = True,
    downscale: int = 2,
) -> dict[str, Dot]:
    """Find one dot per color. Missing colors are simply omitted.

    Pipeline (default args):
      1. Crop to the A4 sheet (skip the gray table entirely).
      2. Downsample by `downscale` (2x => 4x fewer pixels on the mask).
      3. BGR channel-arithmetic masks per color.
      4. Largest blob per color on the downsampled mask -> coarse bbox.
      5. Re-mask the full-resolution pixels in that bbox; moments centroid
         on the actual connected component -> sub-pixel pixel_xy.

    Reliability safety: if the paper-ROI pass returns fewer than 4 dots,
    retry on the full frame (in case paper detection clipped a dot near
    the sheet edge or got split by a shadow). Only the missing colors
    are searched for in the fallback.

    Arguments:
      bands         If given, the legacy HSV path is used instead of BGR
                    arithmetic. Useful for live-tuning under weird lighting.
                    `None` (default) is faster and more reliable.
      use_paper_roi If False, skip the paper-finding step (scan the whole
                    frame). Disable only when you're sure the whole frame
                    is paper, e.g. unit tests with a synthetic background.
      downscale     Integer >= 1. Localization runs at 1/downscale on each
                    axis; centroid refinement always runs at full res.
    """
    primary = _detect_dots_once(image_bgr, bands,
                                use_paper_roi=use_paper_roi,
                                downscale=downscale)
    if use_paper_roi and len(primary) < 4:
        # Retry full-frame for the colors we missed (cheap second pass).
        fallback = _detect_dots_once(image_bgr, bands,
                                     use_paper_roi=False,
                                     downscale=downscale)
        for color, dot in fallback.items():
            primary.setdefault(color, dot)
    return primary


def _detect_dots_once(
    image_bgr: np.ndarray,
    bands: dict[str, list[tuple]] | None,
    *,
    use_paper_roi: bool,
    downscale: int,
) -> dict[str, Dot]:
    H, W = image_bgr.shape[:2]

    # ----- 1. Paper ROI ----------------------------------------------------
    roi = _find_paper_roi(image_bgr) if use_paper_roi else None
    if roi is None:
        x0, y0 = 0, 0
        crop = image_bgr
    else:
        x0, y0, x1, y1 = roi
        crop = image_bgr[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]

    # ----- 2. Downsample for fast localization -----------------------------
    ds = max(1, int(downscale))
    # Don't downsample tiny crops -- we'd lose the dots.
    if ds > 1 and min(ch, cw) < 240:
        ds = 1
    if ds > 1:
        small = cv2.resize(crop, (cw // ds, ch // ds),
                           interpolation=cv2.INTER_AREA)
    else:
        small = crop

    # ----- 3. Color masks --------------------------------------------------
    if bands is None:
        small_masks = _bgr_masks_all(small)
    else:
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        small_masks = {c: _mask_color(hsv, bs) for c, bs in bands.items()}

    # Clean speckle on the downsampled masks (cheap because they're small).
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for c in list(small_masks):
        m = small_masks[c]
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k3, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3, iterations=2)
        small_masks[c] = m

    min_area_small = max(4, MIN_BLOB_AREA_PX // (ds * ds))
    max_area_small = max(min_area_small + 1, MAX_BLOB_AREA_PX // (ds * ds))

    out: dict[str, Dot] = {}

    for color, mask_small in small_masks.items():
        # ----- 4. Largest blob on the downsampled mask ---------------------
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask_small,
                                                          connectivity=8)
        best_i, best_area = -1, 0
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < min_area_small or a > max_area_small:
                continue
            if a > best_area:
                best_i, best_area = i, a
        if best_i < 0:
            continue

        # bbox in `small` coords -> scale to `crop` coords, pad, clamp.
        sx = int(stats[best_i, cv2.CC_STAT_LEFT])
        sy = int(stats[best_i, cv2.CC_STAT_TOP])
        sw = int(stats[best_i, cv2.CC_STAT_WIDTH])
        sh = int(stats[best_i, cv2.CC_STAT_HEIGHT])
        pad = max(2, ds * 2)
        cx_lo = max(0, sx * ds - pad)
        cy_lo = max(0, sy * ds - pad)
        cx_hi = min(cw, (sx + sw) * ds + pad)
        cy_hi = min(ch, (sy + sh) * ds + pad)

        # ----- 5. Refine at full resolution --------------------------------
        region = crop[cy_lo:cy_hi, cx_lo:cx_hi]
        if bands is None:
            region_mask = _bgr_mask_one(region, color)
        else:
            region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            region_mask = _mask_color(region_hsv, bands[color])
        if int(region_mask.sum()) == 0:
            continue

        rn, rlabels, rstats, _ = cv2.connectedComponentsWithStats(
            region_mask, connectivity=8)
        if rn <= 1:
            continue
        # Largest component in the region (excluding background).
        areas = rstats[1:, cv2.CC_STAT_AREA]
        ri = int(np.argmax(areas)) + 1
        real_area = int(rstats[ri, cv2.CC_STAT_AREA])
        if real_area < MIN_BLOB_AREA_PX or real_area > MAX_BLOB_AREA_PX:
            continue

        single = (rlabels == ri).astype(np.uint8)
        M = cv2.moments(single, binaryImage=True)
        if M["m00"] == 0:
            continue
        local_cx = M["m10"] / M["m00"]
        local_cy = M["m01"] / M["m00"]

        # Translate back to full-frame coords (region -> crop -> frame).
        full_cx = local_cx + cx_lo + x0
        full_cy = local_cy + cy_lo + y0

        rx = int(rstats[ri, cv2.CC_STAT_LEFT])
        ry = int(rstats[ri, cv2.CC_STAT_TOP])
        rw = int(rstats[ri, cv2.CC_STAT_WIDTH])
        rh = int(rstats[ri, cv2.CC_STAT_HEIGHT])
        bbox = (rx + cx_lo + x0,
                ry + cy_lo + y0,
                rx + rw + cx_lo + x0,
                ry + rh + cy_lo + y0)

        out[color] = Dot(color=color, pixel_xy=(full_cx, full_cy),
                         area_px=float(real_area), bbox=bbox)

    return out


def annotate_dots(image_bgr: np.ndarray, dots: dict[str, Dot]) -> np.ndarray:
    out = image_bgr.copy()
    palette = {"red": (0, 0, 255), "green": (0, 255, 0),
               "blue": (255, 0, 0), "cyan": (255, 255, 0)}
    for color, d in dots.items():
        c = palette.get(color, (255, 255, 255))
        cx, cy = int(round(d.pixel_xy[0])), int(round(d.pixel_xy[1]))
        cv2.drawMarker(out, (cx, cy), c, cv2.MARKER_CROSS, 18, 2)
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 1)
        cv2.putText(out, color, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out



if __name__ == "__main__":
    img = np.full((480, 640, 3), 230, dtype=np.uint8)  # paper-white background
    placements = {
        "red":   ((100, 240), (0,   0,   255)),
        "green": ((250, 100), (0,   255, 0)),
        "blue":  ((400, 240), (255, 0,   0)),
        "cyan":  ((250, 380), (255, 255, 127)),
    }
    for color, (center, bgr) in placements.items():
        cv2.circle(img, center, 12, bgr, thickness=-1)

    dots = detect_dots(img)
    for color, expected in placements.items():
        if color not in dots:
            print(f"[MISS] {color}")
            continue
        ex, ey = expected[0]
        px, py = dots[color].pixel_xy
        err = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
        print(f"{color}: detected ({px:.1f}, {py:.1f})  expected ({ex}, {ey})  err={err:.2f}px")
