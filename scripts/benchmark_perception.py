#!/usr/bin/env python
"""Side-by-side benchmark of every perception backend on one keyboard image.

USAGE:
    # Real image:
    python scripts/benchmark_perception.py --image data/vass_keyboard.jpg

    # No image, generate a synthetic one and benchmark against it. Useful
    # for offline regression tests of the benchmark plumbing itself.
    python scripts/benchmark_perception.py --synthetic

    # Choose which backends to run (default: every one that is installed):
    python scripts/benchmark_perception.py --image kb.jpg \
        --backends gemini,claude,molmo,qwen,ocr_primary,template,hybrid

OUTPUT:
    runs/bench_<timestamp>/
        report.html         -- the master report (open in a browser)
        report.json        , machine-readable summary
        <backend>_anchors.jpg , per-backend annotated image
        <backend>_keymap.jpg  , per-backend post-homography full keymap

WHAT'S MEASURED:
    For each backend that succeeds:
      n_anchors           , how many requested anchors were returned
      anchor_label_acc    , if --ground-truth provided, fraction of
                              anchors whose label is correct
      anchor_pixel_rmse   , if --ground-truth provided, RMS pixel error
                              of the returned anchor centers vs truth
      homography_rmse_px  , in-sample fit residual of the homography
      crossval_rmse_px     -- held-out reprojection error (if truth provided)
      ocr_agreement_pct   , per-key OCR agreement after homography
      latency_s           , wall-clock duration of detect_anchors
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cybertyping.perception.keymap.layout import load_layout
from cybertyping.perception.keymap.homography import (
    KeyMap, build_keymap_from_anchors, crossvalidate, fit_rmse, reproject,
    annotate, solve_homography,
)
from cybertyping.perception.detect.ocr_backends import (
    OCRBackend, get_default_ocr_backend, verify_keymap,
)
from cybertyping.perception.detect.vlm_anchors import bbox_center



def _load_backend(name: str):
    """Lazily try to construct each backend; return None if unavailable.

    Each backend that can't be constructed (missing API key, missing
    library, model not downloaded) just gets quietly skipped from the
    benchmark instead of blowing up.
    """
    try:
        if name == "gemini":
            from cybertyping.perception.detect.vlm_anchors import GeminiBackend
            return GeminiBackend()
        if name == "claude":
            from cybertyping.perception.detect.vlm_anchors import ClaudeBackend
            return ClaudeBackend()
        if name == "molmo":
            from cybertyping.perception.detect.vlm_anchors import MolmoBackend
            return MolmoBackend()
        if name == "qwen":
            from cybertyping.perception.detect.vlm_anchors import QwenBackend
            return QwenBackend()
        if name == "ocr_primary":
            from cybertyping.perception.detect.ocr_anchors import detect_anchors_via_ocr
            ocr = get_default_ocr_backend()
            if ocr is None:
                return None
            class _OCRPrim:
                def __init__(self, ocr): self._ocr = ocr
                def detect_anchors(self, image, labels):
                    return detect_anchors_via_ocr(image, ocr=self._ocr,
                                                  anchor_labels=labels)
            return _OCRPrim(ocr)
        if name == "template":
            from cybertyping.perception.detect.template_match import TemplateMatcher
            return TemplateMatcher.from_canonical_layout(scale_px=80)
        if name == "hybrid":
            # Hybrid wraps another backend; default to (gemini or molmo or qwen)
            # + paddle/easy OCR.
            from cybertyping.perception.detect.hybrid import HybridRefinedDetector
            for c_name in ("gemini", "molmo", "qwen", "claude"):
                coarse = _load_backend(c_name)
                if coarse is not None:
                    return HybridRefinedDetector(coarse,
                                                 ocr=get_default_ocr_backend())
            return None
    except Exception:
        return None
    return None


ALL_BACKENDS = ["gemini", "claude", "molmo", "qwen",
                "ocr_primary", "template", "hybrid"]



def synthesise_image_and_truth(
    scale_px: int = 75,
    perturb_rot_deg: float = 4.0,
    perturb_scale: float = 0.95,
    perturb_tx: float = 30.0,
    perturb_ty: float = -15.0,
    noise_sigma: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """Render the canonical layout, apply a small affine perturbation, and
    return the perturbed image plus the ground-truth key bboxes IN THE
    PERTURBED IMAGE."""
    from cybertyping.perception.detect.template_match import render_canonical_keyboard
    ref, bboxes_ref = render_canonical_keyboard(scale_px=scale_px)
    h, w = ref.shape[:2]
    rng = np.random.default_rng(seed)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), perturb_rot_deg, perturb_scale)
    M[0, 2] += perturb_tx
    M[1, 2] += perturb_ty

    new_w = w + int(abs(perturb_tx) * 2)
    new_h = h + int(abs(perturb_ty) * 2)
    warped = cv2.warpAffine(ref, M, (new_w, new_h),
                            borderValue=(180, 180, 180))
    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, warped.shape).astype(np.int16)
        warped = np.clip(warped.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Project reference bboxes through M to get ground truth.
    bboxes_truth: dict[str, tuple[int, int, int, int]] = {}
    for label, (x1, y1, x2, y2) in bboxes_ref.items():
        corners = np.array([
            [x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1],
        ], dtype=np.float32)
        proj = (M @ corners.T).T
        bx1, by1 = int(round(proj[:, 0].min())), int(round(proj[:, 1].min()))
        bx2, by2 = int(round(proj[:, 0].max())), int(round(proj[:, 1].max()))
        bboxes_truth[label] = (max(0, bx1), max(0, by1),
                               min(new_w - 1, bx2), min(new_h - 1, by2))
    return warped, bboxes_truth



@dataclass
class BackendResult:
    name: str
    n_requested: int
    n_anchors: int
    anchor_label_acc: float | None
    anchor_pixel_rmse: float | None
    homography_rmse_px: float | None
    crossval_rmse_px: float | None
    ocr_agreement_pct: float | None
    latency_s: float
    error: str | None = None
    # Filenames of saved annotated images (relative to the bench output dir).
    annot_anchors: str | None = None
    annot_keymap: str | None = None


def _bbox_center(b: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _evaluate_one(
    name: str,
    backend,
    image: np.ndarray,
    anchor_labels: list[str],
    crossval_labels: list[str],
    truth: dict[str, tuple[int, int, int, int]] | None,
    ocr: OCRBackend | None,
    out_dir: Path,
) -> BackendResult:
    n_requested = len(anchor_labels)
    t0 = time.perf_counter()
    try:
        det = backend.detect_anchors(image, anchor_labels)
    except Exception as e:
        dt = time.perf_counter() - t0
        return BackendResult(
            name=name, n_requested=n_requested, n_anchors=0,
            anchor_label_acc=None, anchor_pixel_rmse=None,
            homography_rmse_px=None, crossval_rmse_px=None,
            ocr_agreement_pct=None, latency_s=dt,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
        )
    dt = time.perf_counter() - t0
    n_anchors = len(det)

    # Anchor accuracy vs truth
    label_acc = None
    pix_rmse = None
    if truth is not None and det:
        correct = sum(1 for k in det if k in truth)
        label_acc = correct / max(1, len(det))
        errs = []
        for k, bbox in det.items():
            if k not in truth:
                continue
            cx_p, cy_p = _bbox_center(bbox)
            cx_t, cy_t = _bbox_center(truth[k])
            errs.append((cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2)
        pix_rmse = float(np.sqrt(np.mean(errs))) if errs else None

    # Homography fit
    hom_rmse = None
    crossval_rmse = None
    keymap = None
    if n_anchors >= 4:
        try:
            keymap = build_keymap_from_anchors(det)
            hom_rmse = keymap.rmse_canonical_px
        except Exception as e:
            return BackendResult(
                name=name, n_requested=n_requested, n_anchors=n_anchors,
                anchor_label_acc=label_acc, anchor_pixel_rmse=pix_rmse,
                homography_rmse_px=None, crossval_rmse_px=None,
                ocr_agreement_pct=None, latency_s=dt,
                error=f"homography fit failed: {e}",
            )

    # Crossval if we have truth
    if keymap is not None and truth is not None:
        held_out = {k: truth[k] for k in crossval_labels if k in truth and k not in det}
        if held_out:
            errors = crossvalidate(keymap, held_out)
            if errors:
                crossval_rmse = float(np.sqrt(np.mean([v ** 2 for v in errors.values()])))

    # OCR agreement (skip if no OCR backend, or no keymap)
    ocr_agree = None
    if keymap is not None and ocr is not None:
        try:
            rep = verify_keymap(image, keymap, ocr)
            ocr_agree = rep.agreement_rate * 100.0
        except Exception:
            ocr_agree = None

    # Save annotated overlays
    annot_anchors_path = None
    annot_keymap_path = None
    if det:
        annot = image.copy()
        for label, bbox in det.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 200, 0), 1)
            cv2.putText(annot, label.upper(), (x1, max(8, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)
        title = (f"{name}  n={n_anchors}/{n_requested}  "
                 f"px_rmse={pix_rmse if pix_rmse is None else f'{pix_rmse:.1f}'}  "
                 f"H_rmse={hom_rmse if hom_rmse is None else f'{hom_rmse:.1f}'}  "
                 f"t={dt*1000:.0f}ms")
        cv2.putText(annot, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        annot_anchors_path = f"{name}_anchors.jpg"
        cv2.imwrite(str(out_dir / annot_anchors_path), annot)
    if keymap is not None:
        keymap_annot = annotate(image, keymap)
        annot_keymap_path = f"{name}_keymap.jpg"
        cv2.imwrite(str(out_dir / annot_keymap_path), keymap_annot)

    return BackendResult(
        name=name, n_requested=n_requested, n_anchors=n_anchors,
        anchor_label_acc=label_acc, anchor_pixel_rmse=pix_rmse,
        homography_rmse_px=hom_rmse, crossval_rmse_px=crossval_rmse,
        ocr_agreement_pct=ocr_agree, latency_s=dt,
        annot_anchors=annot_anchors_path, annot_keymap=annot_keymap_path,
    )



def _format_cell(val) -> str:
    if val is None:
        return "<td style='color:#999'>n/a</td>"
    if isinstance(val, float):
        return f"<td>{val:.2f}</td>"
    if isinstance(val, int):
        return f"<td>{val}</td>"
    return f"<td>{val}</td>"


def render_html_report(results: list[BackendResult],
                       image_path: str, out_dir: Path) -> str:
    rows = []
    for r in results:
        if r.error:
            err_short = r.error.splitlines()[0][:120]
            rows.append(
                f"<tr><td><b>{r.name}</b></td>"
                f"<td colspan=8 style='color:#c0392b'>ERROR: {err_short}</td></tr>"
            )
            continue
        rows.append(
            "<tr>"
            f"<td><b>{r.name}</b></td>"
            f"{_format_cell(r.n_anchors)}"
            f"{_format_cell(r.anchor_label_acc)}"
            f"{_format_cell(r.anchor_pixel_rmse)}"
            f"{_format_cell(r.homography_rmse_px)}"
            f"{_format_cell(r.crossval_rmse_px)}"
            f"{_format_cell(r.ocr_agreement_pct)}"
            f"{_format_cell(r.latency_s)}"
            f"<td>"
            + (f"<a href='{r.annot_anchors}'>anchors</a>"
               if r.annot_anchors else "")
            + (f" | <a href='{r.annot_keymap}'>keymap</a>"
               if r.annot_keymap else "")
            + "</td></tr>"
        )

    table = (
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse;font-family:sans-serif'>"
        "<thead><tr>"
        "<th>backend</th>"
        "<th>n_anchors</th>"
        "<th>label_acc</th>"
        "<th>anchor_px_rmse</th>"
        "<th>homo_rmse_px</th>"
        "<th>xval_rmse_px</th>"
        "<th>ocr_agree_%</th>"
        "<th>latency_s</th>"
        "<th>overlays</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows) +
        "</tbody></table>"
    )
    html = (
        "<!doctype html><meta charset='utf-8'>"
        "<title>Perception benchmark</title>"
        "<style>"
        "body { font-family: sans-serif; max-width: 1100px; margin: 32px auto; }"
        "h1 { margin-bottom: 4px; }"
        "small { color: #666; }"
        "</style>"
        "<h1>Perception backend benchmark</h1>"
        f"<small>image: <code>{image_path}</code></small>"
        f"<p>{table}</p>"
        "<h3>Reading the table</h3>"
        "<ul>"
        "<li><b>anchor_px_rmse</b> = RMS distance between the backend's anchor centers and ground-truth anchor centers (lower is better; only computed when ground truth is known)</li>"
        "<li><b>homo_rmse_px</b> = in-sample residual of the planar homography fit to the returned anchors</li>"
        "<li><b>xval_rmse_px</b> = held-out reprojection error of the homography on keys the backend did NOT detect</li>"
        "<li><b>ocr_agree_%</b> = fraction of letter keys whose post-homography crop OCRs to the expected letter</li>"
        "</ul>"
    )
    out_html = out_dir / "report.html"
    out_html.write_text(html)
    return str(out_html)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None,
                    help="Keyboard image to benchmark.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use a synthetic image with known ground truth.")
    ap.add_argument("--backends", type=str, default=None,
                    help=f"Comma-separated subset of {ALL_BACKENDS}. "
                         "Default: every one that loads.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output directory. Default: runs/bench_<timestamp>/")
    ap.add_argument("--anchors", type=str, default="q,p,z,m,space,enter",
                    help="Comma-separated anchor labels to request.")
    ap.add_argument("--crossval", type=str, default="r,l,g,n,b,e",
                    help="Comma-separated held-out labels for crossval.")
    args = ap.parse_args()

    if args.image and args.synthetic:
        print("Pick --image OR --synthetic, not both.", file=sys.stderr)
        return 2
    if not args.image and not args.synthetic:
        print("Need --image PATH or --synthetic.", file=sys.stderr)
        return 2

    if args.synthetic:
        print("[setup] Synthesising image + ground truth")
        image, truth = synthesise_image_and_truth()
        image_label = "<synthetic>"
    else:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Could not read {args.image}", file=sys.stderr)
            return 2
        truth = None
        image_label = args.image

    anchor_labels = [s.strip() for s in args.anchors.split(",") if s.strip()]
    crossval_labels = [s.strip() for s in args.crossval.split(",") if s.strip()]

    backends_to_run = (
        [s.strip() for s in args.backends.split(",")]
        if args.backends else ALL_BACKENDS
    )

    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "runs" / f"bench_{int(time.time())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save the input image for reference.
    cv2.imwrite(str(out_dir / "input.jpg"), image)

    print(f"[bench] image: {image_label}")
    print(f"[bench] out:   {out_dir}")
    print(f"[bench] anchors: {anchor_labels}")
    print(f"[bench] backends: {backends_to_run}")

    ocr = get_default_ocr_backend()
    if ocr is None:
        print("[bench] No OCR backend available; agreement metric will be n/a")

    results: list[BackendResult] = []
    for name in backends_to_run:
        print(f"\n[bench] -> {name}")
        backend = _load_backend(name)
        if backend is None:
            print(f"  [skip] {name} unavailable (missing key/library/model)")
            continue
        res = _evaluate_one(
            name, backend, image,
            anchor_labels, crossval_labels, truth, ocr, out_dir,
        )
        print(f"  n_anchors={res.n_anchors}/{res.n_requested}  "
              f"px_rmse={res.anchor_pixel_rmse}  "
              f"H_rmse={res.homography_rmse_px}  "
              f"ocr={res.ocr_agreement_pct}  "
              f"t={res.latency_s*1000:.0f}ms")
        if res.error:
            print(f"  ERROR: {res.error.splitlines()[0]}")
        results.append(res)

    # JSON report
    report = {
        "image": image_label,
        "anchors": anchor_labels,
        "crossval": crossval_labels,
        "results": [asdict(r) for r in results],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # HTML report
    html_path = render_html_report(results, image_label, out_dir)
    print(f"\n[done] Report: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
