"""Eval 3 (50 pts): type a short sentence on an unseen keyboard.

RUBRIC
    2-4 words, a-z and spaces, length <= 15. Score max(0, 5 - d) where d is
    the Levenshtein distance between the typed and the requested text.
    Time budget 10 s per character. 10 rollouts.

APPROACH
    The Eval-2 press (OCR or VLM anchors + homography + visual servo +
    load-sensed descent), applied to each character in turn. Between keys
    the arm returns to the scan pose so the keyboard is fully in view.

    With ``--verify`` the script listens to the laptop's own keyboard
    events (pynput) and reports the Levenshtein score after each sentence.

USAGE
    python -m evals.eval3_sentence --type "hello world"
    python -m evals.eval3_sentence                     # one sentence per stdin line
    python -m evals.eval3_sentence --sentences-file rollouts.txt
    python -m evals.eval3_sentence --no-confirm --verify
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
from cybertyping.servo.targets import tokenize_sentence                 # noqa: E402
from cybertyping.tasks._levenshtein import levenshtein                  # noqa: E402


def parse_args(argv=None, *, fast_default: bool = False) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_rig_args(ap, default_tip_color="red")
    add_key_detector_args(ap)
    add_speed_args(ap)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--type", dest="sentence", default=None,
                     help="type this one sentence and exit")
    src.add_argument("--sentences-file", default=None,
                     help="one sentence per line")
    ap.add_argument("--verify", action="store_true",
                    help="listen to the host keyboard and score each sentence")
    args = ap.parse_args(argv)
    if fast_default:
        args.fast = True
    return args


def _sentences(args):
    if args.sentence is not None:
        yield args.sentence
        return
    if args.sentences_file:
        for line in Path(args.sentences_file).read_text().splitlines():
            if line.strip():
                yield line.strip()
        return
    print("Type one sentence per line; empty line or Ctrl-D quits.")
    while True:
        try:
            raw = input("sentence> ").strip()
        except EOFError:
            return
        if not raw:
            return
        yield raw


def _labels_to_text(labels) -> str:
    return "".join(" " if l == "space" else ("\n" if l == "enter" else l) for l in labels)


def type_sentence(rig, make_target, sentence: str, tip_color: str, settings,
                  listener=None) -> dict:
    labels = tokenize_sentence(sentence)
    print(f"\n===== '{sentence}' -> {len(labels)} keys =====")
    t0 = time.perf_counter()
    n_ok = 0
    for i, label in enumerate(labels, 1):
        print(f"\n--- key {i}/{len(labels)}: {label.upper()} ---")
        res = press_target(rig, make_target(label), tip_color, settings)
        n_ok += int(res["ok"])
    elapsed = time.perf_counter() - t0
    out = {"sentence": sentence, "n_keys": len(labels), "n_contact": n_ok,
           "elapsed_s": elapsed}
    if listener is not None:
        typed = _labels_to_text(e.label for e in listener.drain())
        d = levenshtein(typed.strip(), sentence.strip().lower())
        out.update(typed=typed, levenshtein=d, score=max(0, 5 - d))
        print(f"[verify] typed={typed!r} d={d} score={out['score']}")
    print(f"[sentence] {n_ok}/{len(labels)} contacts in {elapsed:.1f} s "
          f"(budget {10 * len(sentence)} s)")
    return out


def run(args) -> int:
    settings = settings_from_args(args)
    make_target = key_target_factory(args)
    listener = None
    if args.verify:
        from cybertyping.tasks._verify import KeystrokeListener
        listener = KeystrokeListener().__enter__()
        if not listener.available():
            print("[verify] keystroke listener unavailable on this platform")

    results = []
    try:
        with rig_from_args(args) as rig:
            rig.go_to_scan_pose()
            rig.open_camera()
            for sentence in _sentences(args):
                if listener is not None:
                    listener.drain()
                results.append(type_sentence(rig, make_target, sentence,
                                             args.tip_color, settings, listener))
    finally:
        if listener is not None:
            listener.__exit__(None, None, None)

    if results:
        total = sum(r["elapsed_s"] for r in results)
        print(f"\n[done] {len(results)} sentences, total {total:.1f} s")
        if args.verify:
            print(f"[done] total score {sum(r.get('score', 0) for r in results)}")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
