"""
Detect 4 colored dots (red, green, blue, cyan) on paper using ellipse fitting
on contours, then label each ellipse by its mean color in Lab space.

Ellipse fitting (vs Hough circles) handles off-axis shots where the dots
appear as ellipses, not perfect circles. The pipeline:
    1. Grayscale + CLAHE (local contrast boost, helps dim/uneven lighting).
    2. Adaptive threshold (dots are darker than local paper background).
    3. External contours.
    4. cv2.fitEllipse on each contour with >= 5 points.
    5. Filter by area, aspect ratio, and fill ratio (contour / ellipse).
    6. Sample mean Lab inside each surviving ellipse, greedy-assign to the
       4 reference colors.
"""
import cv2
import numpy as np

# Reference colors in Lab space (D65). One reference per label.
REFERENCES_LAB = {
    'red':   np.array([53.24,  80.09,  67.20], dtype=np.float32),
    'green': np.array([87.73, -86.18,  83.18], dtype=np.float32),
    'blue':  np.array([32.30,  79.20, -107.86], dtype=np.float32),
    'cyan':  np.array([91.11, -48.09, -14.13], dtype=np.float32),
}

# Reference hues (OpenCV HSV convention: H in [0, 180]). Red wraps so we
# check both ends. Used by detect_dots(color_match="hsv") for cameras whose
# Lab output drifts (washed-out colors all cluster near cyan in Lab space).
REFERENCES_HUE = {
    'red':   [0.0, 180.0],
    'green': [60.0],
    'cyan':  [90.0],
    'blue':  [120.0],
}
HSV_MIN_SAT = 60      # below this, the pixel is too gray to label by hue
HSV_MIN_VAL = 40      # below this, too dark to label by hue
HSV_MAX_HUE_DIST = 18.0  # max |h - ref| in degrees (out of 180) to accept

LABEL_COLORS_BGR = {           # for drawing
    'red':   (0,   0,   255),
    'green': (0,   200, 0),
    'blue':  (255, 0,   0),
    'cyan':  (255, 255, 0),
}


def detect_dots(img, min_axis=15, max_axis=120,
                max_aspect_ratio=2.0, min_fill_ratio=0.70,
                max_lab_dist=130.0,
                color_match: str = "hsv"):
    """Return {label: (cx, cy, minor_axis, major_axis, angle_deg, dist)}.

    cx, cy        ellipse centroid (float).
    minor/major   full-length axes in pixels (cv2.fitEllipse convention).
    angle_deg     rotation of the major axis, 0-180.
    dist          color-distance from this ellipse's mean color to its label
                  reference (smaller = better color match).

    color_match:
        "hsv" (default) — match by HSV hue. Robust to brightness/saturation
            drift (washed-out colors still hit the right hue bucket).
        "lab" — original Lab L2 distance matching. Strict: a sub-saturated
            blue dot can land closer to the cyan Lab reference than to blue.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE: local histogram equalization. Saves the under-exposed photos
    # where the dots' grayscale value is otherwise too close to the paper.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    gray_eq = clahe.apply(gray)

    # Adaptive threshold rather than Otsu: paper brightness varies across
    # the frame (shadows, vignette), so a global threshold misses dim dots.
    # blockSize must be odd; ~5% of the image dimension is a robust default.
    block = max(15, (min(h, w) // 20) | 1)
    bw = cv2.adaptiveThreshold(
        gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=block, C=10,
    )

    # Small open kills speckle; small close fills interior pinholes from
    # the threshold step (e.g. dot specular highlights).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Lab once for the whole image (only used by color_match="lab").
    if color_match == "lab":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] *= 100.0 / 255.0
        lab[..., 1] -= 128.0
        lab[..., 2] -= 128.0
    else:
        lab = None
    # HSV once for the whole image (only used by color_match="hsv").
    if color_match == "hsv":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        hsv = None

    candidates = []  # (label, dist, cx, cy, ma, MA, angle)
    for cnt in contours:
        if len(cnt) < 5:          # fitEllipse requires >= 5 points.
            continue
        area = cv2.contourArea(cnt)
        if area < 50 or area > 80_000:
            continue

        # cv2.fitEllipse documents the return as (width, height) of the
        # rotated bounding box. The first component is usually the minor
        # axis in practice, but this isn't contractual -- on near-circular
        # blobs the order can flip. Normalize so MA is always the major and
        # `angle` is the rotation of MA (otherwise the perpendicular axis).
        (cx, cy), (axis1, axis2), angle = cv2.fitEllipse(cnt)
        if axis1 > axis2:
            ma, MA = axis2, axis1
            angle = (angle + 90.0) % 180.0
        else:
            ma, MA = axis1, axis2

        if MA < 2 * min_axis or MA > 2 * max_axis:
            continue
        if ma <= 0 or (MA / ma) > max_aspect_ratio:
            continue

        # Fill ratio: a true filled dot's contour area should be close to the
        # fitted ellipse's area. Crescents, glare arcs, paper edges fail this.
        ellipse_area = np.pi * (ma / 2.0) * (MA / 2.0)
        if ellipse_area <= 0 or (area / ellipse_area) < min_fill_ratio:
            continue

        # Sample mean color inside a shrunken ellipse to avoid the edge ring
        # (anti-aliased boundary pixels would bias the color toward paper).
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, ((cx, cy), (ma * 0.6, MA * 0.6), angle), 255, -1)
        if int(mask.sum()) == 0:
            continue

        if color_match == "hsv":
            mean_h, mean_s, mean_v = cv2.mean(hsv, mask=mask)[:3]
            # Reject gray/dark patches — hue is meaningless without saturation
            if mean_s < HSV_MIN_SAT or mean_v < HSV_MIN_VAL:
                continue
            # Hue is on a 180-modulus circle; min(diff, 180-diff) handles wrap.
            best_label = None
            best_dist = float("inf")
            for label, refs in REFERENCES_HUE.items():
                for ref_h in refs:
                    d = abs(mean_h - ref_h)
                    d = min(d, 180.0 - d)
                    if d < best_dist:
                        best_dist = d
                        best_label = label
            if best_dist > HSV_MAX_HUE_DIST:
                continue   # too far from every reference hue — likely noise
            candidates.append((best_label, best_dist, cx, cy, ma, MA, angle))
        else:  # "lab"
            mean_lab = np.array(cv2.mean(lab, mask=mask)[:3], dtype=np.float32)
            # Pick the SINGLE best-matching reference per ellipse (avoids the
            # bug where a sub-saturated blue dot lands closer to the cyan Lab
            # reference and steals the cyan slot, blocking the blue slot).
            best_label, best_dist = min(
                ((label, float(np.linalg.norm(mean_lab - ref)))
                 for label, ref in REFERENCES_LAB.items()),
                key=lambda kv: kv[1],
            )
            candidates.append((best_label, best_dist, cx, cy, ma, MA, angle))

    # Greedy assignment by ascending color distance: each label and each
    # ellipse used at most once. For HSV mode the per-candidate threshold has
    # already rejected bad matches, so the break here is only meaningful in
    # Lab mode.
    candidates.sort(key=lambda c: c[1])
    dist_cutoff = max_lab_dist if color_match == "lab" else float("inf")
    assigned = {}
    used = set()
    for label, dist, cx, cy, ma, MA, angle in candidates:
        if dist > dist_cutoff:
            break
        if label in assigned:
            continue
        key = (round(cx), round(cy))
        if key in used:
            continue
        assigned[label] = (cx, cy, ma, MA, angle, dist)
        used.add(key)
        if len(assigned) == len(REFERENCES_LAB):
            break

    return assigned


# HSV ranges per color label. Tuned for a typical webcam; tweak if your
# camera's white balance shifts a color outside these bands. Red wraps the
# hue axis, so it has two sub-ranges.
HSV_RANGES = {
    'blue':   [(np.array([105, 60, 50]),  np.array([130, 255, 255]))],
    'green':  [(np.array([40,  60, 50]),  np.array([80,  255, 255]))],
    'cyan':   [(np.array([85,  60, 50]),  np.array([100, 255, 255]))],
    'yellow': [(np.array([20,  80, 80]),  np.array([35,  255, 255]))],
    'red':    [(np.array([0,  100, 80]),  np.array([10,  255, 255])),
               (np.array([160,100, 80]),  np.array([179, 255, 255]))],
}


def detect_dot(img, color: str = "blue", min_area: int = 200):
    """Find the largest blob of `color` in the frame. Returns (cx, cy) or None.

    Pure HSV thresholding -> morphology -> largest contour above min_area.
    Robust to brightness/saturation drift (the failure mode that broke the
    Lab-based detect_dots). One color at a time; call twice for tip + target.

    color: one of HSV_RANGES keys ('blue', 'green', 'cyan', 'red').
    min_area: minimum contour area (px²) to accept.
    """
    if color not in HSV_RANGES:
        raise ValueError(
            f"Unknown color {color!r}; pick from {list(HSV_RANGES)}"
        )
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in HSV_RANGES[color]:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return None
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    return (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]))


def detect_dot_color(img, color, **detect_kwargs):
    """Return (cx, cy) for the dot of the requested color label, or None.

    Thin wrapper over detect_dots(). For querying multiple colors per frame
    prefer calling detect_dots() once and reading the returned dict — this
    helper re-runs the full pipeline on every call.

    color must be a key of REFERENCES_LAB ('red', 'green', 'blue', 'cyan').
    detect_kwargs are forwarded to detect_dots (min_axis, max_axis, etc.).
    """
    if color not in REFERENCES_LAB:
        raise ValueError(
            f"Unknown color {color!r}; must be one of {list(REFERENCES_LAB)}"
        )
    detections = detect_dots(img, **detect_kwargs)
    if color not in detections:
        return None
    cx, cy, *_ = detections[color]
    return (float(cx), float(cy))


def annotate(img, detections):
    """Draw the fitted ellipse around each detected dot, plus its major and
    minor axes through the centroid. The ellipse outline is enlarged by 1.4x
    so it sits outside the dot edge -- otherwise the outline blends into
    the dot itself and looks like the dot edge.
    """
    out = img.copy()
    for label, (cx, cy, ma, MA, angle, dist) in detections.items():
        color = LABEL_COLORS_BGR[label]
        center = (int(round(cx)), int(round(cy)))

        # Outline enlarged so it's visible around (not on) the dot.
        cv2.ellipse(out, (center, (ma * 1.4, MA * 1.4), angle), color, 3)

        # Major + minor axes, drawn as line segments through the centroid.
        # After detect_dots' normalization, `angle` is the rotation of the
        # major axis MA, measured from vertical in image coords (y-down).
        # So the unit vector along MA is (sin a, -cos a); minor is perpendicular.
        a_rad = np.deg2rad(angle)
        dx_M, dy_M = np.sin(a_rad), -np.cos(a_rad)
        dx_m, dy_m = np.cos(a_rad),  np.sin(a_rad)
        p1 = (int(cx + dx_M * MA / 2), int(cy + dy_M * MA / 2))
        p2 = (int(cx - dx_M * MA / 2), int(cy - dy_M * MA / 2))
        q1 = (int(cx + dx_m * ma / 2), int(cy + dy_m * ma / 2))
        q2 = (int(cx - dx_m * ma / 2), int(cy - dy_m * ma / 2))
        cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
        cv2.line(out, q1, q2, color, 2, cv2.LINE_AA)

        cv2.drawMarker(out, center, color, cv2.MARKER_CROSS, 14, 2)
        # Use max(ma, MA) so the label sits clear of the dot regardless of
        # which way the ellipse points (defensive even though MA >= ma here).
        label_offset = int(max(ma, MA) * 0.75) + 6
        cv2.putText(out,
                    f'{label} {ma:.0f}x{MA:.0f} @ {angle:.0f}deg',
                    (center[0] + label_offset, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out



def make_test_image():
    """Synthetic paper-with-dots image. Used by the __main__ smoke test."""
    img = np.full((500, 700, 3), 245, dtype=np.uint8)
    grad = np.linspace(0, -15, 700, dtype=np.float32)
    img = np.clip(img.astype(np.float32) + grad[None, :, None], 0, 255).astype(np.uint8)
    dots = [
        ((150, 130), (0,   0,   220)),  # red
        ((550, 130), (0,   180, 0)),    # green
        ((150, 370), (220, 0,   0)),    # blue
        ((550, 370), (220, 220, 0)),    # cyan
    ]
    for (cx, cy), bgr in dots:
        cv2.circle(img, (cx, cy), 35, bgr, -1, lineType=cv2.LINE_AA)
    return img


if __name__ == '__main__':
    img = make_test_image()
    cv2.imwrite('/tmp/input.png', img)

    detections = detect_dots(img)
    for label, (cx, cy, ma, MA, angle, dist) in detections.items():
        print(f'{label:6s} -> center=({cx:.1f},{cy:.1f})  '
              f'axes=({ma:.1f},{MA:.1f})  angle={angle:.1f}  dist={dist:.1f}')

    out = annotate(img, detections)
    cv2.imwrite('/tmp/output.png', out)
    print('\nSaved: /tmp/output.png')
