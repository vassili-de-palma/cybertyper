"""Tests for VLM-backend output parsers (no API calls)."""

import json

import numpy as np
import pytest

from cybertyping.perception.detect.vlm_anchors import (
    StubBackend,
    _parse_strict_json,
    _validate_bboxes,
    _parse_molmo_points,
    bbox_center,
)


def test_strict_json_plain():
    out = _parse_strict_json('{"keys": {"q": [1,2,3,4]}}')
    assert out == {"keys": {"q": [1, 2, 3, 4]}}


def test_strict_json_with_code_fence():
    text = '```json\n{"keys": {"q": [1,2,3,4]}}\n```'
    out = _parse_strict_json(text)
    assert out["keys"]["q"] == [1, 2, 3, 4]


def test_strict_json_extracts_from_noise():
    text = "Here is your answer: {\"keys\": {\"q\": [1,2,3,4]}} thanks!"
    out = _parse_strict_json(text)
    assert out["keys"]["q"] == [1, 2, 3, 4]


def test_validate_bboxes_clips_and_filters():
    raw = {"keys": {"q": [10, 20, 5, 30],       # invalid -- x2<x1
                    "p": [100, 100, 110, 110],   # ok
                    "z": [-10, -10, 5, 5]}}      # clipped
    out = _validate_bboxes(raw, ["q", "p", "z"], frame_w=200, frame_h=200)
    assert "q" not in out
    assert out["p"] == (100, 100, 110, 110)
    assert out["z"][0] == 0 and out["z"][1] == 0


def test_molmo_point_parser():
    text = '<point x="50.0" y="25.0" alt="Q key">Q</point>'
    pts = _parse_molmo_points(text, frame_w=1280, frame_h=720)
    assert "q" in pts
    px, py = pts["q"]
    assert px == pytest.approx(640.0)
    assert py == pytest.approx(180.0)


def test_molmo_points_plural_form():
    text = '<points x1="10" y1="20" x2="30" y2="40">A</points>'
    pts = _parse_molmo_points(text, frame_w=100, frame_h=100)
    assert "a" in pts
    assert pts["a"] == (10.0, 20.0)


def test_stub_backend_returns_baked_anchors():
    baked = {"q": (10, 20, 30, 40), "p": (100, 20, 130, 40)}
    backend = StubBackend(baked)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = backend.detect_anchors(img, ["q", "p", "z"])
    assert out == baked


def test_bbox_center():
    assert bbox_center((0, 0, 10, 20)) == (5.0, 10.0)
