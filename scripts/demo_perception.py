#!/usr/bin/env python
"""End-to-end perception demo for one keyboard image.

Runs OCR detection, fits the homography to the canonical layout, computes
leave-one-out crossval, and writes an annotated diagnostic image showing
which keys were directly observed vs reprojected by the homography.

    python scripts/demo_perception.py keyboard-reference-frame.jpeg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cybertyping.perception.keymap.layout import load_layout
from cybertyping.perception.detect.ocr_backends import TesseractBackend
from cybertyping.perception.detect.ocr_anchors import detect_anchors_via_ocr
from cybertyping.perception.keymap.homography import build_keymap_from_anchors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="keyboard-reference-frame.jpeg")
    ap.add_argument("--out", default="runs/demo_perception.jpg")
    ap.add_argument("--multi", action="store_true",
                    help="Use multi-preprocess OCR (more anchors, +1s)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read image: {args.image}")
        return 2

    layout = load_layout()
    ocr = TesseractBackend()
    h, w = img.shape[:2]
    print(f"Input image: {args.image}  ({w}x{h})")

    t0 = time.perf_counter()
    anchors = detect_anchors_via_ocr(
        img, ocr=ocr,
        anchor_labels=layout.letters() + ["space", "enter"],
        min_confidence=0.3,
        use_multi_preprocess=args.multi,
    )
    t_ocr = (time.perf_counter() - t0) * 1000

    if len(anchors) < 4:
        print(f"Only found {len(anchors)} anchors; cannot fit homography")
        return 3

    t = time.perf_counter()
    km = build_keymap_from_anchors(anchors, layout=layout)
    t_fit = (time.perf_counter() - t) * 1000

    # OCR-first with a gross-error sanity check.
    #
    # Default: trust the OCR-detected position for every key OCR saw.
    # Empirical positions are more reliable than canonical assumptions
    # (especially for wide keys with asymmetric labels like Enter/Space).
    #
    # Fall back to homography reprojection ONLY when:
    #   - OCR didn't detect the key (key is missing), or
    #   - OCR detected the key at a position MORE than MAX_GROSS_PX from
    #     the homography expectation (indicating a likely mis-read where
    #     OCR put the letter at the wrong key location).
    #
    # MAX_GROSS_PX is intentionally lenient (~1.5 keycap widths) so we
    # accept asymmetric labels (Enter printed at the top of its key)
    # while still rejecting genuine mis-reads (Q showing up where M is).
    MAX_GROSS_PX = 55.0
    overridden = []
    rejected = []
    for label, bbox in anchors.items():
        if label not in km.entries:
            continue
        ocr_xy = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        hom_xy = km.entries[label].image_xy
        dist = float(np.hypot(ocr_xy[0] - hom_xy[0], ocr_xy[1] - hom_xy[1]))
        if dist > MAX_GROSS_PX:
            rejected.append((label, dist))
            continue
        km.entries[label].image_xy = ocr_xy
        overridden.append(label)
    reprojected = [l for l in km.entries
                   if l not in overridden
                   and l not in (r[0] for r in rejected)]

    # Leave-one-out crossval
    errs = []
    for drop in km.anchors_used:
        reduced = {k: v for k, v in anchors.items() if k != drop}
        try:
            km2 = build_keymap_from_anchors(reduced, layout=layout)
            pred = np.array(km2.entries[drop].image_xy)
            tb = anchors[drop]
            true_xy = np.array(((tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0))
            errs.append((drop, float(np.linalg.norm(pred - true_xy))))
        except Exception:
            pass

    print()
    print(f"OCR detection:      {t_ocr:6.0f} ms  ({len(anchors)} letters found)")
    print(f"Homography fit:     {t_fit:6.0f} ms  (RMSE {km.rmse_canonical_px:.2f} px)")
    if errs:
        worst = max(e for _, e in errs)
        mean = float(np.mean([e for _, e in errs]))
        print(f"Leave-one-out:      worst {worst:.2f} px, mean {mean:.2f} px")
    print(f"Full keymap:        {len(km.entries)} keys reprojected "
          f"(a-z + space + enter)")
    print(f"OCR-direct:         {len(overridden)} keys ({sorted(overridden)})")
    print(f"Reprojected:        {len(reprojected)} keys ({sorted(reprojected)})")
    if rejected:
        print(f"OCR rejected as gross errors (>{int(55)}px from expected):")
        for label, dist in sorted(rejected, key=lambda kv: -kv[1]):
            print(f"  {label}: OCR pos {dist:.0f}px from homography expectation")

    # Annotated diagnostic image. Colour-code:
    #   GREEN square  -> press target came from OCR detection
    #   ORANGE cross  -> press target came from homography reprojection
    #   RED cross     -> OCR detected but rejected as gross error
    ocr_set = set(overridden)
    reject_set = {r[0] for r in rejected}
    overlay = img.copy()
    for label, entry in km.entries.items():
        cx, cy = int(round(entry.image_xy[0])), int(round(entry.image_xy[1]))
        if label in ocr_set:
            color = (0, 220, 0)
            cv2.drawMarker(overlay, (cx, cy), color, cv2.MARKER_SQUARE, 12, 2)
        elif label in reject_set:
            color = (0, 0, 255)
            cv2.drawMarker(overlay, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)
        else:
            color = (0, 165, 255)
            cv2.drawMarker(overlay, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(overlay, label.upper(), (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    header = (f"{len(ocr_set)} OCR-direct (green) + "
              f"{len(reprojected)} reprojected (orange)"
              + (f" + {len(reject_set)} rejected (red)" if reject_set else ""))
    cv2.putText(overlay, header, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if errs:
        sub = (f"H_RMSE={km.rmse_canonical_px:.2f}px  "
               f"L1O worst={worst:.2f}px  mean={mean:.2f}px")
        cv2.putText(overlay, sub, (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)
    print(f"\nAnnotated overlay -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
