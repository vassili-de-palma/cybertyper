"""Eval 3 entry point: type a sentence on an unseen keyboard.

RUBRIC:
    Robot receives a 2-4 word sentence (a-z + spaces, len <= 15). Per
    rollout, scores max(0, 5 - levenshtein(typed, target)) after
    10 * len(sentence) seconds. 10 rollouts per group.

STRATEGY:
    One OCR-based scan at startup, then loop reading sentences from
    stdin. Each sentence -> per-letter trajectory stitch -> stream ->
    verify -> retry on miss. Falls back to VLM if OCR finds <4 anchors.

USAGE:
    python -m evals.open_loop.eval3_sentence
    python -m evals.open_loop.eval3_sentence --type "hello world"
    python -m evals.open_loop.eval3_sentence --sentences-file rollouts.txt

DEFAULTS (overridable on the CLI):
    --interactive          read sentences from stdin
    --preset FAST          2.2 deg/tick, adjacent/near/far lift 5/10/30 mm
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
    sys.exit(_session_main(DEFAULTS + sys.argv[1:]))
