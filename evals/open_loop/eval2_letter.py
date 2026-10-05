"""Eval 2 entry point: press a single letter.

RUBRIC:
    Robot receives one character (a-z) at a time and must press the
    correct key within 10 seconds. 16 rollouts per group. 3.125 pts each
    on success.

STRATEGY:
    One OCR-based scan at startup, then loop reading single letters from
    stdin. Each letter -> full-stack press via cybertyping.tasks.session.

USAGE:
    python -m evals.open_loop.eval2_letter             # uses Eval-2 defaults
    python -m evals.open_loop.eval2_letter --no-escalate
    python evals/open_loop/eval2_letter.py --use-cached-keymap

DEFAULTS (overridable on the CLI):
    --interactive          read letters from stdin
    --preset FAST          2.2 deg/tick, no contact-sense per press
    --multi-preprocess     OCR with 3 preprocessing variants (+1s setup)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cybertyping.tasks.session import main as _session_main

DEFAULTS = ["--interactive", "--preset", "FAST", "--multi-preprocess"]


if __name__ == "__main__":
    # Defaults first so user-provided CLI wins (argparse takes the last value).
    sys.exit(_session_main(DEFAULTS + sys.argv[1:]))
