"""Tests for the scripts/benchmark_perception.py plumbing.

We exercise the synthetic image generator and the per-backend evaluator
with a deterministic StubBackend, so the test runs without network /
hardware / API keys.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Importing the benchmark module from its scripts/ path. We must register
# it in sys.modules BEFORE exec_module so that dataclasses can resolve the
# enclosing module's namespace at class-definition time.
spec_path = REPO_ROOT / "scripts" / "benchmark_perception.py"
import importlib.util
_spec = importlib.util.spec_from_file_location("benchmark_perception", spec_path)
benchmark = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_perception"] = benchmark
_spec.loader.exec_module(benchmark)


def test_synthetic_image_and_truth_shapes():
    img, truth = benchmark.synthesise_image_and_truth(seed=42)
    assert img.ndim == 3 and img.shape[2] == 3
    assert "q" in truth
    h, w = img.shape[:2]
    # Truth boxes lie inside the image.
    for label, bbox in truth.items():
        x1, y1, x2, y2 = bbox
        assert 0 <= x1 < x2 < w
        assert 0 <= y1 < y2 < h


def test_evaluate_one_with_stub_backend(tmp_path):
    from cybertyping.perception.detect.vlm_anchors import StubBackend
    img, truth = benchmark.synthesise_image_and_truth(seed=0)
    # Use the truth itself as the stub's output -> perfect detector.
    backend = StubBackend({k: v for k, v in truth.items()
                           if k in ["q", "p", "z", "m", "space", "enter"]})
    res = benchmark._evaluate_one(
        name="stub_perfect",
        backend=backend,
        image=img,
        anchor_labels=["q", "p", "z", "m", "space", "enter"],
        crossval_labels=["r", "l", "g", "n"],
        truth=truth,
        ocr=None,
        out_dir=tmp_path,
    )
    assert res.error is None
    assert res.n_anchors >= 4
    # On a perfect stub, anchor_pixel_rmse should be tiny.
    assert res.anchor_pixel_rmse is not None
    assert res.anchor_pixel_rmse < 1.0
    assert res.homography_rmse_px is not None


def test_evaluate_one_records_errors(tmp_path):
    class _Boom:
        def detect_anchors(self, image, labels):
            raise RuntimeError("kaboom")
    img, _ = benchmark.synthesise_image_and_truth(seed=0)
    res = benchmark._evaluate_one(
        name="boom", backend=_Boom(), image=img,
        anchor_labels=["q", "p"], crossval_labels=[],
        truth=None, ocr=None, out_dir=tmp_path,
    )
    assert res.error is not None
    assert "kaboom" in res.error
    assert res.n_anchors == 0


def test_html_report_writes(tmp_path):
    from cybertyping.perception.detect.vlm_anchors import StubBackend
    img, truth = benchmark.synthesise_image_and_truth(seed=0)
    anchors = {k: v for k, v in truth.items() if k in ["q", "p", "z", "m", "space", "enter"]}
    backend = StubBackend(anchors)
    res = benchmark._evaluate_one(
        name="stub", backend=backend, image=img,
        anchor_labels=list(anchors.keys()), crossval_labels=["r", "l"],
        truth=truth, ocr=None, out_dir=tmp_path,
    )
    html_path = benchmark.render_html_report([res], "<test>", tmp_path)
    assert Path(html_path).exists()
    content = Path(html_path).read_text()
    assert "stub" in content
