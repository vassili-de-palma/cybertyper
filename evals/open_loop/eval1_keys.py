"""Eval 1 path 2 — press SPACE, ENTER, R, L in order.

RUBRIC:
    Press the four keys SPACE, ENTER, R, L sequentially. 12.5 pts each.
    Must finish in 40 seconds. (Eval 1 path 1, colored-dots, is in
    `eval1_dots.py` and is the recommended path; this is the keyboard
    alternative.)

STRATEGY:
    Reuse the full Eval-2/3 scan + press pipeline via session.main, but
    feed it the literal key list "space enter r l" through the --keys
    flag (which bypasses character tokenisation, so 'space' / 'enter'
    stay as named keys rather than being split into characters).

USAGE:
    python -m evals.open_loop.eval1_keys
    python evals/open_loop/eval1_keys.py --preset SAFE
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cybertyping.tasks.session import main as _session_main

# SAFE preset for Eval 1 because the budget is generous (40s for 4 keys),
# we want the contact-sensing press, and we have no time pressure.
DEFAULTS = ["--keys", "space enter r l", "--preset", "SAFE", "--multi-preprocess"]


if __name__ == "__main__":
    sys.exit(_session_main(DEFAULTS + sys.argv[1:]))
