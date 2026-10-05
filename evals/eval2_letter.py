"""Eval 2 (50 pts): press one given letter per rollout.

RUBRIC
    The policy receives one character a-z and must press it within 10 s.
    16 rollouts, 3.125 pts each.

APPROACH  (OCR visual servo, the demo-day keyboard pipeline)
    Letter anchors are read from the wrist-camera frame by OCR (Tesseract)
    and a RANSAC homography maps the canonical US layout onto the image, so
    the target key's centre is known even when its own glyph is not
    readable. The red tip marker is then servoed onto that pixel in image
    space, the detection is refreshed every few ticks while the camera
    moves, and the press finishes with a load-sensed vertical descent.

    ``--detector vlm`` swaps OCR for a vision-language model (Claude or
    Gemini via API key, or a local Molmo/Qwen); same homography afterwards.

USAGE
    python -m evals.eval2_letter                 # interactive: one letter per line
    python -m evals.eval2_letter --letters r l   # scripted list, then exit
    python -m evals.eval2_letter --detector vlm
    python -m evals.eval2_letter --no-confirm    # start as soon as tip+key are seen
    OCR_MULTI=1 python -m evals.eval2_letter     # heavier OCR, more anchors
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cybertyping.servo import press_target                              # noqa: E402
from cybertyping.servo.cli import (                                     # noqa: E402
    add_key_detector_args, add_rig_args, add_speed_args,
    key_target_factory, rig_from_args, settings_from_args,
)
from cybertyping.servo.targets import normalize_key                     # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_rig_args(ap, default_tip_color="red")
    add_key_detector_args(ap)
    add_speed_args(ap)
    ap.add_argument("--letters", nargs="*", default=None,
                    help="press these in order and exit (default: read stdin)")
    return ap.parse_args(argv)


def _read_letters_interactive():
    print("Type one key per line (a-z, space, enter); 'q' or Ctrl-D quits.")
    while True:
        try:
            raw = input("key> ").strip().lower()
        except EOFError:
            return
        if raw in ("q", "quit", "exit"):
            return
        label = normalize_key(raw)
        if label is None:
            print(f"  unknown key '{raw}'")
            continue
        yield label


def main(argv=None) -> int:
    args = _parse_args(argv)
    settings = settings_from_args(args)
    make_target = key_target_factory(args)

    if args.letters is not None:
        labels = [normalize_key(l) for l in args.letters]
        if any(l is None for l in labels):
            print(f"error: unknown key in {args.letters}")
            return 2
        source = iter(labels)
    else:
        source = _read_letters_interactive()

    with rig_from_args(args) as rig:
        rig.go_to_scan_pose()
        rig.open_camera()
        n_ok = n_total = 0
        for label in source:
            n_total += 1
            print(f"\n===== {label.upper()} =====")
            t0 = time.perf_counter()
            res = press_target(rig, make_target(label), args.tip_color, settings)
            n_ok += int(res["ok"])
            status = "ok" if res["ok"] else (res.get("reason") or res["info"]["reason"])
            print(f"[{label}] {status} in {time.perf_counter() - t0:.1f} s")
        print(f"\n[done] {n_ok}/{n_total} presses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
