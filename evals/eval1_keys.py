"""Eval 1, keyboard variant (50 pts): press SPACE, ENTER, R, L in order.

RUBRIC
    12.5 pts per key pressed in the right order, 40 s total.

APPROACH
    Same OCR/VLM visual-servo press as Eval 2, run over a fixed key list.

USAGE
    python -m evals.eval1_keys                          # space enter r l
    python -m evals.eval1_keys --keys r l               # custom list
    python -m evals.eval1_keys --detector vlm           # VLM anchors
    python -m evals.eval1_keys --no-confirm --fast
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

DEFAULT_KEYS = ["space", "enter", "r", "l"]


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_rig_args(ap, default_tip_color="red")
    add_key_detector_args(ap)
    add_speed_args(ap)
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS,
                    help="key labels to press in order (a-z, space, enter)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    labels = [normalize_key(k) for k in args.keys]
    if any(l is None for l in labels):
        print(f"error: unknown key in {args.keys}")
        return 2
    settings = settings_from_args(args)
    make_target = key_target_factory(args)

    with rig_from_args(args) as rig:
        rig.go_to_scan_pose()
        rig.open_camera()
        t0 = time.perf_counter()
        n_ok = 0
        for label in labels:
            print(f"\n===== {label.upper()} =====")
            res = press_target(rig, make_target(label), args.tip_color, settings)
            n_ok += int(res["ok"])
            print(f"[{label}] {'ok' if res['ok'] else res.get('reason') or res['info']['reason']}")
        print(f"\n[done] {n_ok}/{len(labels)} keys in {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
