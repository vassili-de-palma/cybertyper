#!/usr/bin/env python
"""Hardware demo: servo the coloured tip onto a coloured dot and touch it.

The smallest end-to-end exercise of the visual-servo stack. Put a blue dot
(or any colour from ``--target-color``) on the table, mark the gripper tip
with a different colour, and run::

    python scripts/touch_color.py                     # blue dot, yellow tip
    python scripts/touch_color.py --target-color green --tip-color red
    CAMERA_INDEX=0 python scripts/touch_color.py

Each cycle: wait for ENTER, diagonal approach, planar refinement, vertical
descent until contact, retreat, back to the scan pose. q/ESC quits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cybertyping.servo import ColorDotTarget, press_target              # noqa: E402
from cybertyping.servo.cli import (                                     # noqa: E402
    add_rig_args, add_speed_args, rig_from_args, settings_from_args,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_rig_args(ap, default_tip_color="yellow")
    add_speed_args(ap)
    ap.add_argument("--target-color", default="blue",
                    choices=["red", "green", "blue", "cyan", "yellow"])
    args = ap.parse_args(argv)
    if args.target_color == args.tip_color:
        print("error: target and tip colours must differ")
        return 2
    settings = settings_from_args(args)

    with rig_from_args(args) as rig:
        rig.go_to_scan_pose()
        rig.open_camera()
        while True:
            res = press_target(rig, ColorDotTarget(args.target_color),
                               args.tip_color, settings)
            if not res["ok"] and res.get("reason") == "skipped by user":
                break
            print(f"[cycle] {res.get('info', {}).get('reason', res.get('reason'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
