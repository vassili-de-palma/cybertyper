#!/usr/bin/env python
"""
keyboard_full_scan_gemini.py

One photo of the keyboard → Gemini 2.5 Pro detects ALL keys at once →
maps every key to robot XY+Z → saves vision/key_map.json + vision/key_map.jpg.

Usage:
    python vision/keyboard_full_scan_gemini.py --camera 0
    python vision/keyboard_full_scan_gemini.py --image kb.jpg

Requires:
    pip install google-generativeai opencv-python pillow numpy
    world_settings.json  (legacy plane fit from the earliest calibration workflow)
"""

import argparse
import base64
import json
import os
import platform
import re
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import google.generativeai as genai

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cybertyping.core import primitives as ts
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ──── CONFIG ────────────────────────────────────────────────────────────────
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL       = "gemini-2.5-pro-preview-05-06"
PORT               = "COM5" if platform.system() == "Windows" else "/dev/ttyACM0"
ROBOT_ID           = "team08_follower_arm"
URDF_PATH          = str(Path(__file__).resolve().parents[2] / "SO101" / "so101_new_calib.urdf")
WORLD_SETTING_PATH = Path(__file__).resolve().parents[2] / "world_settings.json"
CAMERA_FOV_H_DEG   = 60.0
OUT_DIR            = Path(__file__).resolve().parent

# Fixed camera XYZ at the scan pose; None = read live from the robot via FK.
SCAN_POSE_XYZ      = None   # e.g. (+0.123456, -0.056789, +0.321000)


# ──── GEMINI BACKEND ────────────────────────────────────────────────────────

def init_gemini() -> genai.GenerativeModel:
    key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("ERROR: set GEMINI_API_KEY env var or GEMINI_API_KEY constant")
        sys.exit(1)
    genai.configure(api_key=key)
    print(f"[gemini] Using model: {GEMINI_MODEL}")
    return genai.GenerativeModel(GEMINI_MODEL)


def frame_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def gemini_scan_keyboard(frame: np.ndarray, model: genai.GenerativeModel) -> dict:
    """One Gemini call → all key bounding boxes as {label: [x1,y1,x2,y2]}."""
    prompt = """This image shows a keyboard. Detect every letter key (a-z) and the space bar.

Return a single JSON object where each key is a letter (a-z) or "space",
and each value is a bounding box [x1, y1, x2, y2] in pixel coordinates (0,0 = top-left).

Example:
{
  "a": [120, 340, 160, 380],
  "b": [200, 400, 240, 440],
  "space": [300, 450, 700, 490]
}

Return ONLY the JSON object. No explanation, no markdown, no extra text."""

    print("[gemini] Scanning full keyboard (one call for all keys)...")
    pil_img = frame_to_pil(frame)
    response = model.generate_content([prompt, pil_img])
    raw = response.text.strip()
    print(f"[gemini] Raw output ({len(raw)} chars):\n{raw}\n")

    # Parse JSON — try direct, then extract first {...} block
    result = None
    try:
        result = json.loads(raw)
    except Exception:
        # strip markdown fences
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        candidate = fence.group(1) if fence else None
        if candidate is None:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start >= 0 and end > start:
                candidate = raw[start:end]
        if candidate:
            try:
                result = json.loads(candidate)
            except Exception:
                pass

    if result is None:
        print("[gemini] ERROR: Could not parse JSON from model output")
        return {}

    bboxes = {}
    for label, val in result.items():
        label = label.lower().strip()
        if not isinstance(val, list) or len(val) != 4:
            continue
        try:
            bboxes[label] = [int(v) for v in val]
        except Exception:
            continue

    print(f"[gemini] Detected {len(bboxes)} keys: {sorted(bboxes.keys())}")
    return bboxes


# ──── COORDINATE MAPPING ────────────────────────────────────────────────────

def pixel_to_robot_xy(cx: float, cy: float, frame_w: int, frame_h: int,
                      ee_xyz: np.ndarray, world: dict) -> tuple:
    rx_cam, ry_cam, rz_cam = ee_xyz
    plane = world["plane"]
    rz_plane = plane["a"] * rx_cam + plane["b"] * ry_cam + plane["c"]
    camera_height = rz_cam - rz_plane
    if camera_height <= 0:
        raise ValueError(f"Camera height {camera_height*1000:.1f} mm <= 0 — is robot in scan pose?")
    fov_rad       = np.radians(CAMERA_FOV_H_DEG)
    meters_per_px = (camera_height * 2.0 * np.tan(fov_rad / 2.0)) / frame_w
    dx_m = (cx - frame_w / 2.0) * meters_per_px
    dy_m = (cy - frame_h / 2.0) * meters_per_px
    return rx_cam + dx_m, ry_cam + dy_m


def build_key_map(bboxes: dict, frame_w: int, frame_h: int,
                  ee_xyz: np.ndarray, world: dict) -> dict:
    plane = world["plane"]
    key_map = {}
    for label, (x1, y1, x2, y2) in bboxes.items():
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        try:
            rx, ry = pixel_to_robot_xy(cx, cy, frame_w, frame_h, ee_xyz, world)
        except ValueError as e:
            print(f"[map] Skipping '{label}': {e}")
            continue
        rz = plane["a"] * rx + plane["b"] * ry + plane["c"]
        key_map[label] = {
            "pixel_center": [round(cx, 1), round(cy, 1)],
            "bbox_px":      [x1, y1, x2, y2],
            "robot_xy":     [round(rx, 6), round(ry, 6)],
            "robot_z":      round(rz, 6),
        }
    return key_map


# ──── VISUALISATION ─────────────────────────────────────────────────────────

def annotate(frame: np.ndarray, key_map: dict) -> np.ndarray:
    out = frame.copy()
    for label, info in key_map.items():
        x1, y1, x2, y2 = info["bbox_px"]
        cx, cy = int(info["pixel_center"][0]), int(info["pixel_center"][1])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)
        cv2.putText(out, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return out


# ──── MAIN ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  help="Path to keyboard image (skips camera)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--port",   default=PORT, help="Robot serial port")
    args = parser.parse_args()

    # World settings
    if not WORLD_SETTING_PATH.exists():
        print(f"ERROR: {WORLD_SETTING_PATH} not found — legacy plane file; see reference/vlm/README.md")
        sys.exit(1)
    with open(WORLD_SETTING_PATH) as f:
        world = json.load(f)
    plane = world["plane"]
    print(f"[world] Plane: z = {plane['a']:.4f}x + {plane['b']:.4f}y + {plane['c']:.4f}")

    # Gemini
    model = init_gemini()

    # Camera position
    chain = ts.build_chain(URDF_PATH, locked_joints=ts.LOCKED_JOINTS)
    if SCAN_POSE_XYZ is not None:
        ee_xyz = np.array(SCAN_POSE_XYZ, dtype=float)
        print(f"[pose] Using fixed scan pose: ({ee_xyz[0]:+.4f}, {ee_xyz[1]:+.4f}, {ee_xyz[2]:+.4f}) m")
    else:
        print("[robot] Connecting to read current pose via FK...")
        cfg   = SO101FollowerConfig(port=args.port, id=ROBOT_ID)
        robot = SO101Follower(cfg)
        robot.connect()
        obs   = robot.get_observation()
        q_now = np.array([float(obs[f"{n}.pos"]) for n in ts.BODY_MOTOR_ORDER], dtype=float)
        robot.disconnect()
        print("[robot] Disconnected")
        ee_xyz = ts.fk_tip(chain, q_now, ts.TOOL_TIP_OFFSET_LOCAL_XYZ)
        print(f"[fk] Camera XYZ: ({ee_xyz[0]:+.4f}, {ee_xyz[1]:+.4f}, {ee_xyz[2]:+.4f}) m")

    rz_plane      = plane["a"] * ee_xyz[0] + plane["b"] * ee_xyz[1] + plane["c"]
    camera_height = ee_xyz[2] - rz_plane
    print(f"[pose] Height above keyboard: {camera_height*1000:.1f} mm")

    # Capture frame
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: Could not read {args.image}")
            sys.exit(1)
    else:
        print(f"[camera] Capturing from camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if platform.system() == "Windows" \
              else cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("ERROR: Could not capture frame")
            sys.exit(1)

    frame_h, frame_w = frame.shape[:2]
    scale_mm_px = camera_height * 2 * np.tan(np.radians(CAMERA_FOV_H_DEG / 2)) / frame_w * 1000
    print(f"[frame] {frame_w}x{frame_h} px  |  {scale_mm_px:.2f} mm/px\n")

    # Save raw capture
    raw_img = OUT_DIR / "raw_capture.jpg"
    cv2.imwrite(str(raw_img), frame)
    print(f"[frame] Saved raw capture -> {raw_img}")

    # One Gemini call → all keys
    bboxes = gemini_scan_keyboard(frame, model)
    if not bboxes:
        print("ERROR: No keys detected — check raw_capture.jpg to verify the camera saw the keyboard")
        sys.exit(1)

    # Build key map with robot coordinates
    key_map = build_key_map(bboxes, frame_w, frame_h, ee_xyz, world)

    # Save key_map.json
    out_json = OUT_DIR / "key_map.json"
    with open(out_json, "w") as f:
        json.dump(key_map, f, indent=2)
    print(f"[out] Saved key map -> {out_json}")

    # Save annotated image
    out_img = OUT_DIR / "key_map.jpg"
    cv2.imwrite(str(out_img), annotate(frame, key_map))
    print(f"[out] Saved annotated image -> {out_img}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  {'Key':<8} {'Pixel cx,cy':<20} {'Robot X':>10} {'Robot Y':>10} {'Robot Z':>10}")
    print(f"  {'-'*8} {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for label in sorted(key_map.keys()):
        info = key_map[label]
        cx, cy = info["pixel_center"]
        rx, ry = info["robot_xy"]
        rz     = info["robot_z"]
        print(f"  {label:<8} ({cx:6.1f}, {cy:6.1f})       {rx:+.4f}   {ry:+.4f}   {rz:+.4f}")
    print(f"{'='*60}")
    print(f"\n{len(key_map)} keys mapped. Load key_map.json for the press pipeline.")

    cv2.imshow("Key Map", annotate(frame, key_map))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
