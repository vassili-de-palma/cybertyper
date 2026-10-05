#!/usr/bin/env python
"""
Keyboard key detection and coordinate mapping using Qwen2.5-VL.

One picture of the whole keyboard → Qwen returns ALL key positions at once.
After that, any key lookup is instant (no re-running Qwen, no re-capturing).

Usage:
    python reference/vlm/keyboard_full_scan_qwen.py --camera 0        # capture from camera
    python reference/vlm/keyboard_full_scan_qwen.py --image kb.jpg    # use saved image

Output:
    key_map.json   — {label: {pixel_center, robot_xy, robot_z}} for every key
    key_map.jpg    — annotated image showing all detected keys

Then for any key:
    key_map["r"]["robot_xy"]  →  (rx, ry) in meters, ready to pass to press logic

Requires:
    world_settings.json  (legacy plane fit from the earliest calibration workflow)
    pip install transformers accelerate qwen-vl-utils Pillow torch
"""

import argparse
import json
import platform
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("ERROR: pip install qwen-vl-utils")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cybertyping.core import primitives as ts
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID           = "Qwen/Qwen2.5-VL-3B-Instruct"
PORT               = "COM5" if platform.system() == "Windows" else "/dev/ttyACM0"
ROBOT_ID           = "team08_follower_arm"
URDF_PATH          = str(Path(__file__).resolve().parents[2] / "SO101" / "so101_new_calib.urdf")
WORLD_SETTING_PATH = Path(__file__).resolve().parents[2] / "world_settings.json"
CAMERA_FOV_H_DEG   = 60.0   # horizontal FOV — adjust to your webcam spec
OUT_DIR            = Path(__file__).resolve().parent

# Fixed camera XYZ at the scan pose; None = read live from the robot via FK.
SCAN_POSE_XYZ      = None   # e.g. (+0.123456, -0.056789, +0.321000)


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    print("[model] Loading Qwen2.5-VL...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("[model] Ready")
    return model, processor


# ── Qwen inference ────────────────────────────────────────────────────────────

def qwen_infer(frame: np.ndarray, prompt: str, model, processor) -> str:
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},
            {"type": "text",  "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=2048)
    generated = out_ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ── Full keyboard scan ────────────────────────────────────────────────────────

SCAN_PROMPT = """This image shows a keyboard. Detect every letter key (a-z) and the space bar.

Return a single JSON object where each key in the JSON is a letter (a-z) or "space",
and each value is a bounding box [x1, y1, x2, y2] in pixel coordinates (0,0 = top-left).

Example format:
{
  "a": [120, 340, 160, 380],
  "b": [200, 400, 240, 440],
  "space": [300, 450, 700, 490]
}

Return ONLY the JSON object. No explanation, no markdown, no extra text."""


def scan_keyboard(frame: np.ndarray, model, processor) -> dict:
    """
    Ask Qwen2.5-VL to detect ALL keys in one shot.
    Returns {label: [x1, y1, x2, y2]} for every key it found.
    """
    print("[qwen] Scanning full keyboard (one call for all keys)...")
    raw = qwen_infer(frame, SCAN_PROMPT, model, processor)
    print(f"[qwen] Raw output ({len(raw)} chars):\n{raw[:500]}{'...' if len(raw)>500 else ''}\n")

    # Extract JSON
    result = None
    try:
        result = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(raw[start:end])
            except Exception:
                pass

    if result is None:
        print("[qwen] ERROR: Could not parse JSON from model output")
        return {}

    # Validate: keep only entries that are 4-element bbox lists
    bboxes = {}
    for label, val in result.items():
        label = label.lower().strip()
        if not isinstance(val, list) or len(val) != 4:
            continue
        try:
            bboxes[label] = [int(v) for v in val]
        except Exception:
            continue

    print(f"[qwen] Detected {len(bboxes)} keys: {sorted(bboxes.keys())}")
    return bboxes


# ── Coordinate mapping ────────────────────────────────────────────────────────

def pixel_to_robot_xy(cx: float, cy: float,
                      frame_w: int, frame_h: int,
                      ee_xyz: np.ndarray, world: dict) -> tuple:
    rx_cam, ry_cam, rz_cam = ee_xyz
    plane = world["plane"]
    rz_plane = plane["a"] * rx_cam + plane["b"] * ry_cam + plane["c"]
    camera_height = rz_cam - rz_plane

    if camera_height <= 0:
        raise ValueError(f"Camera height {camera_height*1000:.1f} mm ≤ 0 — is robot in scan pose?")

    fov_rad       = np.radians(CAMERA_FOV_H_DEG)
    meters_per_px = (camera_height * 2.0 * np.tan(fov_rad / 2.0)) / frame_w

    dx_m = (cx - frame_w / 2.0) * meters_per_px
    dy_m = (cy - frame_h / 2.0) * meters_per_px
    return rx_cam + dx_m, ry_cam + dy_m


def build_key_map(bboxes: dict, frame_w: int, frame_h: int,
                  ee_xyz: np.ndarray, world: dict) -> dict:
    """
    Convert every bbox → pixel center → robot XY + Z.
    Returns full key map ready for the pressing pipeline.
    """
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


# ── Visualisation ─────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

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

    # Load model
    model, processor = load_model()

    # Camera position — use fixed scan pose if set, otherwise read from robot FK
    chain = ts.build_chain(URDF_PATH, locked_joints=ts.LOCKED_JOINTS)
    if SCAN_POSE_XYZ is not None:
        ee_xyz = np.array(SCAN_POSE_XYZ, dtype=float)
        robot = None
        print(f"[pose] Using fixed scan pose: ({ee_xyz[0]:+.4f}, {ee_xyz[1]:+.4f}, {ee_xyz[2]:+.4f}) m")
    else:
        print("[robot] Connecting to read current pose via FK...")
        cfg   = SO101FollowerConfig(port=args.port, id=ROBOT_ID)
        robot = SO101Follower(cfg)
        robot.connect()
        print("[robot] Connected")
        obs   = robot.get_observation()
        q_now = np.array([float(obs[f"{n}.pos"]) for n in ts.BODY_MOTOR_ORDER], dtype=float)
        ee_xyz = ts.fk_tip(chain, q_now, ts.TOOL_TIP_OFFSET_LOCAL_XYZ)

    rz_plane      = plane["a"] * ee_xyz[0] + plane["b"] * ee_xyz[1] + plane["c"]
    camera_height = ee_xyz[2] - rz_plane
    print(f"[pose] Height above keyboard: {camera_height*1000:.1f} mm")

    # Get frame
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
    print(f"[frame] {frame_w}×{frame_h} px  |  {scale_mm_px:.2f} mm/px\n")

    # Save raw capture so you can verify what the camera saw
    raw_img = OUT_DIR / "raw_capture.jpg"
    cv2.imwrite(str(raw_img), frame)
    print(f"[frame] Saved raw capture → {raw_img}")

    # ONE Qwen call → all keys
    bboxes  = scan_keyboard(frame, model, processor)
    if not bboxes:
        print("ERROR: No keys detected — check raw_capture.jpg to verify the camera saw the keyboard")
        sys.exit(1)

    # Build full key map with robot coordinates
    key_map = build_key_map(bboxes, frame_w, frame_h, ee_xyz, world)

    # Save key_map.json
    out_json = OUT_DIR / "key_map.json"
    with open(out_json, "w") as f:
        json.dump(key_map, f, indent=2)
    print(f"[out] Saved key map → {out_json}")

    # Save annotated image
    out_img = OUT_DIR / "key_map.jpg"
    cv2.imwrite(str(out_img), annotate(frame, key_map))
    print(f"[out] Saved annotated image → {out_img}")

    # Print summary
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

    robot.disconnect()
    print("[robot] Disconnected")


if __name__ == "__main__":
    main()
