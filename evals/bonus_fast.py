"""Bonus (50 pts): type the Eval-3 sentences as fast as possible.

RUBRIC
    Total typing time over the ten Eval-3 sentences; a sentence with any
    error counts as the full 10 s per character. Top five groups score
    50 / 40 / 30 / 20 / 10.

APPROACH
    Eval 3 with the ``fast`` servo settings: looser pixel tolerance, higher
    loop rate, faster descent, no operator confirmation. Only switch to
    this once the default settings press reliably, since one miss costs
    more than the time saved on every other key.

USAGE
    python -m evals.bonus_fast --type "hello world"
    python -m evals.bonus_fast --sentences-file rollouts.txt --verify
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.eval3_sentence import parse_args, run    # noqa: E402


def main(argv=None) -> int:
    return run(parse_args(argv, fast_default=True))


if __name__ == "__main__":
    sys.exit(main())
