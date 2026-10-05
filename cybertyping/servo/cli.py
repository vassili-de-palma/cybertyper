"""Shared command-line plumbing for the eval entry points."""

from __future__ import annotations

import argparse
import os

from cybertyping.servo.press import ServoSettings
from cybertyping.servo.rig import DEFAULT_CAL_PATH, Rig
from cybertyping.servo.targets import (
    KeyTarget, ocr_anchor_detector, vlm_anchor_detector,
)


def add_rig_args(ap: argparse.ArgumentParser, default_tip_color: str) -> None:
    ap.add_argument("--camera", type=int, default=None,
                    help="wrist camera index (default: CAMERA_INDEX env, else 1)")
    ap.add_argument("--calibration", default=str(DEFAULT_CAL_PATH),
                    help="click_homography.json with the scan pose and plane_z")
    ap.add_argument("--tip-color", default=default_tip_color,
                    choices=["red", "yellow", "blue", "green", "cyan"],
                    help="colour of the marker on the gripper tip")
    ap.add_argument("--no-confirm", action="store_true",
                    help="start each motion automatically instead of waiting for ENTER")


def add_key_detector_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--detector", choices=["ocr", "vlm"], default="ocr",
                    help="how key anchors are found: Tesseract OCR (default) or a "
                         "vision-language model (needs ANTHROPIC_API_KEY or GEMINI_API_KEY)")
    ap.add_argument("--ocr-multi", action="store_true",
                    help="OCR with three preprocessing variants (more anchors, +1 s)")
    ap.add_argument("--refresh-ticks", type=int, default=None,
                    help="re-detect the keyboard every N servo ticks "
                         "(default 15 for OCR, 60 for VLM; env OCR_REFRESH_TICKS)")


def add_speed_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--fast", action="store_true",
                    help="looser pixel tolerance, higher loop rate, no confirmation")
    ap.add_argument("--tol-px", type=float, default=None)
    ap.add_argument("--hover-mm", type=float, default=None,
                    help="height above the table for the XY phases")


def rig_from_args(args) -> Rig:
    return Rig(cal_path=args.calibration, camera_index=args.camera)


def settings_from_args(args) -> ServoSettings:
    s = ServoSettings.fast() if getattr(args, "fast", False) else ServoSettings()
    if getattr(args, "no_confirm", False):
        s.confirm = False
    if getattr(args, "tol_px", None) is not None:
        s.tol_px = args.tol_px
    if getattr(args, "hover_mm", None) is not None:
        s.hover_m = args.hover_mm / 1000.0
    return s


def key_target_factory(args):
    """Return ``make(label) -> KeyTarget`` using the detector chosen on the CLI.

    The anchor detector (and its OCR/VLM backend) is built once and shared
    by every key of the session.
    """
    if args.detector == "vlm":
        detect = vlm_anchor_detector()
        default_refresh = 60
    else:
        detect = ocr_anchor_detector(multi_preprocess=args.ocr_multi or None)
        default_refresh = int(os.environ.get("OCR_REFRESH_TICKS", "15"))
    refresh = args.refresh_ticks or default_refresh
    print(f"[detector] {args.detector} refresh_every={refresh} ticks")

    def make(label: str) -> KeyTarget:
        return KeyTarget(label, detect, refresh_every_ticks=refresh)
    return make
