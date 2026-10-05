#!/usr/bin/env python
"""Live diagnostic for detect_dots(): see which step fails for the blue dot.

Shows three windows side by side:
  - "live"   : camera feed with annotated detections (colored ellipses)
  - "binary" : adaptive-threshold mask (what detect_dots sees as foreground)
  - "candidates" : every contour that passed the size/aspect/fill filters,
                   with its mean Lab color and per-label distances printed
                   to stdout. Lets you tune max_lab_dist.

USAGE:
    $env:CAMERA_INDEX="0"; python scripts/probe_dots.py
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

if platform.system() == "Windows":
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cybertyping.perception.detect.dots import (  # noqa: E402
    REFERENCES_LAB,
    LABEL_COLORS_BGR,
    detect_dots,
    annotate,
)


def _backend():
    s = platform.system()
    if s == "Windows":
        return cv2.CAP_DSHOW
    if s == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def _binary_mask(img):
    """Reproduce detect_dots' threshold step so we can visualize it."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    gray_eq = clahe.apply(gray)
    block = max(15, (min(h, w) // 20) | 1)
    bw = cv2.adaptiveThreshold(
        gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=block, C=10,
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
    return bw


def _candidate_diagnostics(img, max_lab_dist_show=400.0):
    """Reproduce detect_dots' candidate loop and return per-contour reports."""
    h, w = img.shape[:2]
    bw = _binary_mask(img)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0

    reports = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area < 50 or area > 80_000:
            continue
        (cx, cy), (a1, a2), angle = cv2.fitEllipse(cnt)
        if a1 > a2:
            ma, MA = a2, a1
            angle = (angle + 90.0) % 180.0
        else:
            ma, MA = a1, a2
        if MA < 30 or MA > 240:
            continue
        if ma <= 0 or (MA / ma) > 2.0:
            continue
        ell_area = np.pi * (ma / 2.0) * (MA / 2.0)
        if ell_area <= 0 or (area / ell_area) < 0.7:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, ((cx, cy), (ma * 0.6, MA * 0.6), angle), 255, -1)
        if int(mask.sum()) == 0:
            continue
        mean_lab = np.array(cv2.mean(lab, mask=mask)[:3], dtype=np.float32)
        dists = {
            label: float(np.linalg.norm(mean_lab - ref))
            for label, ref in REFERENCES_LAB.items()
        }
        best = min(dists.items(), key=lambda kv: kv[1])
        reports.append({
            "cx": cx, "cy": cy, "ma": ma, "MA": MA,
            "mean_lab": mean_lab, "dists": dists, "best": best,
        })
    return bw, reports


def main() -> int:
    idx = int(os.environ.get("CAMERA_INDEX", "1"))
    cap = cv2.VideoCapture(idx, _backend())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"[probe] cannot open camera index {idx}")
        return 1

    print(f"[probe] camera index {idx} open. q/ESC to quit.")
    print("[probe] press SPACE to print all candidates for the current frame.")

    last_print_n = -1
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Run the real detector
        detections = detect_dots(frame)
        annotated = annotate(frame, detections)

        bw, reports = _candidate_diagnostics(frame)

        # Draw every candidate (even ones rejected by Lab dist) in white
        for r in reports:
            cv2.ellipse(annotated,
                        ((r["cx"], r["cy"]), (r["ma"] * 1.4, r["MA"] * 1.4), 0),
                        (255, 255, 255), 1)
            cv2.putText(annotated,
                        f"{r['best'][0][:1]}{r['best'][1]:.0f}",
                        (int(r["cx"]) + 6, int(r["cy"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(annotated, f"detected: {list(detections.keys())}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, f"candidates: {len(reports)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("live", annotated)
        cv2.imshow("binary", bw)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            print(f"\n=== {len(reports)} candidates this frame ===")
            for i, r in enumerate(reports):
                print(f"  [{i}] center=({r['cx']:.0f},{r['cy']:.0f}) "
                      f"size={r['ma']:.0f}x{r['MA']:.0f} "
                      f"mean_lab=({r['mean_lab'][0]:+.1f},"
                      f"{r['mean_lab'][1]:+.1f},{r['mean_lab'][2]:+.1f}) "
                      f"best_match={r['best'][0]} @ dist={r['best'][1]:.1f}")
                for lab_name, d in r["dists"].items():
                    flag = " <-- THRESH" if d <= 130 else ""
                    print(f"        {lab_name:6s} dist={d:6.1f}{flag}")
            print(f"detected by detect_dots: {detections}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
