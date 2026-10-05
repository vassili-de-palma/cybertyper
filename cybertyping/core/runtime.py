"""Shared helpers for eval entry scripts.

Keeps each tasks/eval*.py file thin and readable. Things that live here:
- robot connect + safe disconnect
- scan pose loading
- keymap caching
- dry-run mode plumbing
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Repo root on sys.path so `evals/` and `scripts/` resolve when run directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env so SO101_PORT / SO101_ROBOT_ID etc. are available to every
# script. Best-effort: if python-dotenv is not installed, env
# vars must come from the shell.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass


def env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None else default


PORT      = env("SO101_PORT", "COM5")
ROBOT_ID  = env("SO101_ROBOT_ID", "team08_follower_arm")
URDF_PATH = str(REPO_ROOT / "SO101" / "so101_new_calib.urdf")
WORLD_PATH = REPO_ROOT / "world_settings_load.json"
# Scan pose of the legacy open-loop pipeline (the servo pipeline reads its
# scan pose from configs/click_homography.json instead).
SCAN_POSE_PATH = REPO_ROOT / "configs" / "scan_pose.json"



def is_dry_run() -> bool:
    return os.environ.get("CYBERTYPING_DRY_RUN", "").lower() in ("1", "true", "yes")


@contextmanager
def maybe_robot():
    """Context manager that yields a connected SO101Follower, or None in dry-run.

    Eval scripts should branch on the result:

        with maybe_robot() as robot:
            if robot is None:
                # dry-run: just plan & log
                ...
            else:
                # real hardware
                ...
    """
    if is_dry_run():
        yield None
        return

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    cfg   = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
    robot = SO101Follower(cfg)
    print(f"[robot] Connecting on {PORT} id={ROBOT_ID}")
    robot.connect()
    print("[robot] OK")

    # Disconnect on PROCESS exit, not on with-block exit. session.py's
    # setup_session() returns a Session from inside `with maybe_robot()`
    # and the caller (main + type_one_sentence) then uses session.robot.
    # If we disconnected on __exit__, the returned robot would already be
    # dead. atexit guarantees we always clean up the bus even if the user
    # ctrl-C's or an exception bubbles up.
    import atexit
    def _close_bus():
        try:
            robot.bus.enable_torque()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        print("[robot] Disconnected")
    atexit.register(_close_bus)

    yield robot
    # NB: NO disconnect here. atexit handles it.



def load_scan_pose() -> dict | None:
    """Return {x,y,z} of the calibrated scan pose, or None if not yet captured."""
    if SCAN_POSE_PATH.exists():
        with open(SCAN_POSE_PATH) as f:
            return json.load(f)
    return None


def load_world() -> dict | None:
    if not WORLD_PATH.exists():
        return None
    with open(WORLD_PATH) as f:
        return json.load(f)



@dataclass
class TrialLog:
    task: str
    started_at: float
    rows: list[dict]

    def add(self, **kw):
        self.rows.append({"t": time.time() - self.started_at, **kw})

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"task": self.task, "rows": self.rows}, f, indent=2)


def new_log(task: str) -> TrialLog:
    return TrialLog(task=task, started_at=time.time(), rows=[])



def banner(msg: str, char: str = "=") -> None:
    line = char * max(60, len(msg) + 4)
    print(f"\n{line}\n  {msg}\n{line}")


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)
