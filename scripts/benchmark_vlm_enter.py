#!/usr/bin/env python
"""Compare VLM detect_point vs detect_bbox for the Enter key.

For each available VLM backend (set ANTHROPIC_API_KEY / GEMINI_API_KEY or
have molmo/qwen weights local), runs both query modes against the input
image, times each, and reports the resulting (x, y) center alongside the
homography-projected Enter position for reference.

    python scripts/benchmark_vlm_enter.py keyboard-reference-frame.jpeg
    python scripts/benchmark_vlm_enter.py kb.jpg --label enter --label space
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


def _try_load(name: str):
    """Lazy backend construction. Returns None if it can't be made."""
    try:
        if name == "claude":
            from cybertyping.perception.detect.vlm_anchors import ClaudeBackend
            return ClaudeBackend()
        if name == "gemini":
            from cybertyping.perception.detect.vlm_anchors import GeminiBackend
            return GeminiBackend()
        if name == "molmo":
            from cybertyping.perception.detect.vlm_anchors import MolmoBackend
            return MolmoBackend()
        if name == "qwen":
            from cybertyping.perception.detect.vlm_anchors import QwenBackend
            return QwenBackend()
    except Exception as e:
        print(f"  [{name}] unavailable: {type(e).__name__}: {e}")
        return None
    return None


def _homography_reference(image_bgr: np.ndarray, label: str
                          ) -> tuple[float, float] | None:
    """OCR-primary + homography, returns the expected (x, y) for `label`."""
    layout = load_layout()
    try:
        ocr = TesseractBackend()
        anchors = detect_anchors_via_ocr(
            image_bgr, ocr=ocr,
            anchor_labels=layout.letters() + ["space", "enter"],
            use_multi_preprocess=True,
        )
        km = build_keymap_from_anchors(anchors, layout=layout)
        if label in km.entries:
            return km.entries[label].image_xy
    except Exception:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--label", action="append", default=None,
                    help="Key label(s) to query. Default: enter, space")
    ap.add_argument("--backends", type=str,
                    default="claude,gemini,molmo,qwen",
                    help="Comma-separated list of backends to try.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="How many times to call each backend (averages timing).")
    ap.add_argument("--out", type=str, default="runs/vlm_enter_compare.jpg")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read {args.image}")
        return 2
    h, w = img.shape[:2]
    labels = args.label or ["enter", "space"]
    print(f"Image: {args.image} ({w}x{h})")
    print(f"Labels: {labels}")

    # Reference from OCR + homography
    refs: dict[str, tuple[float, float] | None] = {
        lbl: _homography_reference(img, lbl) for lbl in labels
    }
    for lbl, p in refs.items():
        if p:
            print(f"  homography ref for {lbl!r}: ({p[0]:.0f}, {p[1]:.0f})")
        else:
            print(f"  homography ref for {lbl!r}: (not available)")

    backend_names = [n.strip() for n in args.backends.split(",") if n.strip()]
    results = []   # rows of (backend, label, mode, x, y, dt_ms, dist_to_ref)

    for bname in backend_names:
        print(f"\n[load] {bname}")
        backend = _try_load(bname)
        if backend is None:
            continue
        for lbl in labels:
            ref = refs.get(lbl)
            # Point mode
            for r in range(args.repeats):
                t = time.perf_counter()
                try:
                    pt = backend.detect_point(img, lbl)
                except Exception as e:
                    pt = None
                    print(f"  [{bname}] point '{lbl}' raised: {e}")
                dt_ms = (time.perf_counter() - t) * 1000
                if pt is not None:
                    dist = float(np.hypot(pt[0]-ref[0], pt[1]-ref[1])) if ref else float("nan")
                    print(f"  [{bname}] point {lbl}: ({pt[0]:5.0f},{pt[1]:5.0f})  "
                          f"{dt_ms:6.0f}ms  dist_to_ref={dist:5.0f}px")
                    results.append((bname, lbl, "point", pt[0], pt[1], dt_ms, dist))
                else:
                    print(f"  [{bname}] point {lbl}: FAIL                {dt_ms:6.0f}ms")
                    results.append((bname, lbl, "point", None, None, dt_ms, None))
            # Bbox mode
            for r in range(args.repeats):
                t = time.perf_counter()
                try:
                    bb = backend.detect_bbox(img, lbl)
                except Exception as e:
                    bb = None
                    print(f"  [{bname}] bbox '{lbl}' raised: {e}")
                dt_ms = (time.perf_counter() - t) * 1000
                if bb is not None:
                    cx, cy = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
                    dist = float(np.hypot(cx-ref[0], cy-ref[1])) if ref else float("nan")
                    bw = bb[2]-bb[0]; bh = bb[3]-bb[1]
                    print(f"  [{bname}] bbox  {lbl}: ({cx:5.0f},{cy:5.0f})  "
                          f"{dt_ms:6.0f}ms  dist_to_ref={dist:5.0f}px  size=({bw}x{bh})")
                    results.append((bname, lbl, "bbox", cx, cy, dt_ms, dist))
                else:
                    print(f"  [{bname}] bbox  {lbl}: FAIL                {dt_ms:6.0f}ms")
                    results.append((bname, lbl, "bbox", None, None, dt_ms, None))

    # Summary table
    print("\n" + "=" * 78)
    print(f"  {'backend':10s}  {'label':8s}  {'mode':6s}  {'x':>6s} {'y':>6s}  {'t(ms)':>7s}  {'dist_to_ref':>11s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*6} {'-'*6}  {'-'*7}  {'-'*11}")
    for bname, lbl, mode, x, y, dt, dist in results:
        x_str = f"{x:6.0f}" if x is not None else "  FAIL"
        y_str = f"{y:6.0f}" if y is not None else "  FAIL"
        d_str = f"{dist:11.1f}" if dist is not None and not np.isnan(dist) else "         na"
        print(f"  {bname:10s}  {lbl:8s}  {mode:6s}  {x_str} {y_str}  {dt:7.0f}  {d_str}")

    # Render an annotated comparison image
    overlay = img.copy()
    # Reference points: green crosses
    for lbl, ref in refs.items():
        if ref is None: continue
        x, y = int(round(ref[0])), int(round(ref[1]))
        cv2.drawMarker(overlay, (x, y), (0, 220, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(overlay, f"ref:{lbl}", (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1, cv2.LINE_AA)
    # Backend results
    color_map = {
        "claude": (255,128,0), "gemini": (0,200,255),
        "molmo": (255,0,255),  "qwen": (180,180,180),
    }
    for bname, lbl, mode, x, y, dt, dist in results:
        if x is None: continue
        col = color_map.get(bname, (255,255,255))
        marker = cv2.MARKER_TRIANGLE_UP if mode == "point" else cv2.MARKER_SQUARE
        px, py = int(round(x)), int(round(y))
        cv2.drawMarker(overlay, (px, py), col, marker, 14, 2)
        cv2.putText(overlay, f"{bname[:3]}/{mode[0]}", (px+5, py+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    cv2.putText(overlay, "green=homography reference. shapes: ^point square=bbox",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"\nAnnotated comparison -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
