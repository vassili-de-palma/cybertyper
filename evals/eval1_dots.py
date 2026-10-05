"""Eval 1 (50 pts): hover above four coloured dots in order.

RUBRIC
    Four ~1.5 cm dots (#FF0000 red, #00FF00 green, #0000FF blue, #7FFFFF cyan)
    at random positions on an A4 sheet. Hover the gripper at most 3 cm above
    each dot, overlapping at least half of it, in that order. 12.5 pts per
    dot, 25 s total.

APPROACH  (colour visual servo, the demo-day pipeline)
    For each colour: detect the dot and the coloured tip marker in the
    wrist-camera image, servo the tip over the dot at scan height, then
    descend vertically to 1 cm above the table while a closed-loop XY hold
    cancels the parallax drift. Hold, return to the scan pose, next colour.
    No pixel-to-robot calibration is involved; only configs/click_homography.json
    for the scan pose and the table height.

USAGE
    python -m evals.eval1_dots                       # red green blue cyan
    python -m evals.eval1_dots --colors green blue   # subset / custom order
    python -m evals.eval1_dots --no-confirm          # auto-start each dot
    python -m evals.eval1_dots --tip-color yellow    # marker on the gripper

    The tip marker must not share a colour with any dot. Yellow is the
    default for this eval.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cybertyping.servo import ColorDotTarget, hover_target          # noqa: E402
from cybertyping.servo.cli import add_rig_args, rig_from_args       # noqa: E402

RUBRIC_ORDER = ["red", "green", "blue", "cyan"]


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_rig_args(ap, default_tip_color="yellow")
    ap.add_argument("--colors", nargs="+", default=RUBRIC_ORDER,
                    choices=["red", "green", "blue", "cyan"],
                    help="dot colours in the order to visit")
    ap.add_argument("--hover-mm", type=float, default=10.0,
                    help="final tip height above the table")
    ap.add_argument("--hold-s", type=float, default=1.0)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.tip_color in args.colors:
        print(f"error: tip colour '{args.tip_color}' is also a target colour")
        return 2

    with rig_from_args(args) as rig:
        rig.go_to_scan_pose()
        rig.open_camera()
        t0 = time.perf_counter()
        results = []
        for color in args.colors:
            print(f"\n===== {color.upper()} =====")
            res = hover_target(
                rig, ColorDotTarget(color), args.tip_color,
                final_hover_m=args.hover_mm / 1000.0, hold_s=args.hold_s,
                confirm=not args.no_confirm,
            )
            results.append(res)
            print(f"[{color}] {'ok' if res['ok'] else res.get('reason')}")
        total = time.perf_counter() - t0
        n_ok = sum(1 for r in results if r["ok"])
        print(f"\n[done] {n_ok}/{len(args.colors)} dots in {total:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
