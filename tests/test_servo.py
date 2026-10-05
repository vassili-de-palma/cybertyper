"""Offline tests for the visual-servo package (no camera, no robot)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cybertyping.perception.detect.dots import detect_dot
from cybertyping.perception.keymap.layout import load_layout
from cybertyping.servo.targets import (
    ColorDotTarget, KeyTarget, find_tip, normalize_key, tokenize_sentence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- tokenising

def test_normalize_key():
    assert normalize_key("a") == "a"
    assert normalize_key("Q") == "q"
    assert normalize_key(" ") == "space"
    assert normalize_key("space") == "space"
    assert normalize_key("enter") == "enter"
    assert normalize_key("1") is None
    assert normalize_key("ab") is None


def test_tokenize_sentence_drops_unmappable():
    assert tokenize_sentence("Hi, you") == ["h", "i", "space", "y", "o", "u"]
    assert tokenize_sentence("") == []


# ---------------------------------------------------------------- colour dots

def _frame_with_dot(color_bgr, center=(200, 150), radius=18):
    img = np.full((300, 400, 3), 235, dtype=np.uint8)
    cv2.circle(img, center, radius, color_bgr, -1, lineType=cv2.LINE_AA)
    return img


@pytest.mark.parametrize("color,bgr", [
    ("blue", (230, 20, 20)), ("green", (20, 200, 20)),
    ("red", (20, 20, 230)), ("yellow", (0, 220, 230)), ("cyan", (230, 230, 0)),
])
def test_detect_dot_finds_centroid(color, bgr):
    pt = detect_dot(_frame_with_dot(bgr), color)
    assert pt is not None
    assert abs(pt[0] - 200) < 2 and abs(pt[1] - 150) < 2


def test_find_tip_rejects_large_jump():
    frame = _frame_with_dot((230, 20, 20))
    assert find_tip(frame, "blue", prior_xy=(205, 148)) is not None
    assert find_tip(frame, "blue", prior_xy=(10, 10), max_jump_px=50) is None


def test_color_target_persists_last_position():
    t = ColorDotTarget("blue")
    assert t.update(_frame_with_dot((230, 20, 20))) is not None
    blank = np.full((300, 400, 3), 235, dtype=np.uint8)
    assert t.update(blank) is not None       # keeps the last seen pixel


# ---------------------------------------------------------------- key target

def _synthetic_anchors(layout, labels, scale=40.0, offset=(60.0, 80.0)):
    out = {}
    for l in labels:
        cx, cy = layout.center(l)
        px, py = cx * scale + offset[0], cy * scale + offset[1]
        out[l] = (int(px - 12), int(py - 12), int(px + 12), int(py + 12))
    return out


def test_key_target_predicts_unseen_key_from_homography():
    layout = load_layout()
    anchors = _synthetic_anchors(layout, ["q", "p", "z", "m", "t", "g"])
    calls = {"n": 0}

    def detector(frame):
        calls["n"] += 1
        return anchors

    target = KeyTarget("h", detector, layout=layout, refresh_every_ticks=5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    px = target.update(frame)
    assert target.status == "ok"
    hx, hy = layout.center("h")
    assert px is not None
    assert abs(px[0] - (hx * 40 + 60)) < 1.0 and abs(px[1] - (hy * 40 + 80)) < 1.0

    # Detector is only re-run every refresh_every_ticks updates.
    for _ in range(4):
        target.update(frame)
    assert calls["n"] == 1
    target.update(frame)
    assert calls["n"] == 2
    target.draw(frame)   # overlay must not raise


def test_key_target_reports_few_anchors():
    layout = load_layout()
    target = KeyTarget("a", lambda f: _synthetic_anchors(layout, ["q", "p"]), layout=layout)
    assert target.update(np.zeros((10, 10, 3), dtype=np.uint8)) is None
    assert target.status == "few_anchors"


# ---------------------------------------------------------------- control math

def test_enforce_camera_dir_is_idempotent_at_scan_pose():
    pytest.importorskip("ikpy")
    from cybertyping.core import primitives as prim
    from cybertyping.servo.control import enforce_camera_dir

    cal = json.loads((REPO_ROOT / "configs" / "click_homography.json").read_text())
    q = np.array([cal["joint_angles_deg"][n] for n in prim.BODY_MOTOR_ORDER], dtype=float)
    # The recorded scan pose has wrist_flex slightly past the +-95 deg URDF
    # limit that the hold clamps to; start the test from a pose inside it.
    wf = prim.BODY_MOTOR_ORDER.index("wrist_flex")
    q[wf] = float(np.clip(q[wf], -90.0, 90.0))
    chain = prim.build_chain(prim.URDF_PATH, locked_joints=prim.LOCKED_JOINTS)
    cam_dir = prim.fk_T(chain, q)[:3, 2]
    q_out = enforce_camera_dir(chain, q, cam_dir)
    assert np.allclose(q_out, q, atol=0.05)

    # Perturb the wrist by 3 deg: the hold must pull it back toward the target.
    q_bad = q.copy()
    q_bad[wf] += 3.0
    q_fix = enforce_camera_dir(chain, q_bad, cam_dir)
    err_before = np.linalg.norm(prim.fk_T(chain, q_bad)[:3, 2] - cam_dir)
    err_after = np.linalg.norm(prim.fk_T(chain, q_fix)[:3, 2] - cam_dir)
    assert err_after < err_before
