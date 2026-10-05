"""Bonus entry point: type Eval-3 sentences as fast as possible.

RUBRIC:
    Sum of typing time across all 10 Eval-3 sentences (with maxed-out
    fallback time for any sentence typed incorrectly). Lowest sum wins
    50pts; 2nd 40pts; ...; 5th 10pts; no bonus from 6th onward.

STRATEGY:
    Same flow as Eval 3 but with the BLITZ preset (3.5 deg/tick, very
    short lifts, no verify+retry). Optionally `--use-cached-keymap` to
    skip the scan if `runs/last_keymap.json` is fresh.

USAGE:
    python -m evals.open_loop.bonus_fast
    python -m evals.open_loop.bonus_fast --use-cached-keymap

DEFAULTS (overridable on the CLI):
    --interactive          read sentences from stdin
    --preset BLITZ         3.5 deg/tick, very short lifts, no verify
    --multi-preprocess     OCR with 3 preprocessing variants (+1s setup)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cybertyping.tasks.session import main as _session_main

DEFAULTS = ["--interactive", "--preset", "BLITZ", "--multi-preprocess"]


if __name__ == "__main__":
    sys.exit(_session_main(DEFAULTS + sys.argv[1:]))
