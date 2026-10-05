#!/usr/bin/env python
"""
SO-101 follower test with keyboard (ergonomic layout).

LAYOUT (hold to move):
    Main arm (right hand, arrows):
        LEFT  / RIGHT : shoulder_pan   (base rotation - / +)
        UP    / DOWN  : shoulder_lift  (raise / lower shoulder)

    Joints (left hand):
        W / S : elbow_flex   (+ / -)
        Q / E : wrist_flex   (+ / -)
        A / D : wrist_roll   (- / +)

    Gripper:
        Z : close   (hold)
        C : open    (hold)

    Utility:
        + / -  : increase / decrease speed (step per tick)
        R      : "relax" -> reset target to current position (soft stop)
        P      : print current positions
        X/ESC  : exit

Before running:
    1) lerobot-find-port
    2) lerobot-calibrate --robot.type=so101_follower --robot.port=COMx --robot.id=<ROBOT_ID>

Edit PORT and ROBOT_ID below.
"""

import os
import time

from pynput import keyboard

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ====== CONFIG ======
PORT = os.environ.get("SO101_PORT", "COM5")
ROBOT_ID = os.environ.get("SO101_ROBOT_ID", "team08_follower_arm")

STEP_DEG = 1.5            # degrees per tick (joints)
GRIPPER_STEP = 2.0        # 0..100 per tick (gripper)
LOOP_HZ = 30

STEP_MIN = 0.3
STEP_MAX = 6.0
STEP_DELTA = 0.3          # how much speed changes with +/-

# Anti-beep: goal cannot be more than MAX_LEAD from present -> no overload
MAX_LEAD_DEG = 8.0
MAX_LEAD_GRIPPER = 10.0

DEBUG = False             # if True, print every key press
# ============================


KEY_TO_JOINT_SPECIAL = {
    # arrows and special keys mapped via keyboard.Key.<name>
    "left":  ("shoulder_pan",  -1),
    "right": ("shoulder_pan",  +1),
    "up":    ("shoulder_lift", +1),
    "down":  ("shoulder_lift", -1),
}

KEY_TO_JOINT_CHAR = {
    "w": ("elbow_flex",  +1),
    "s": ("elbow_flex",  -1),
    "q": ("wrist_flex",  +1),
    "e": ("wrist_flex",  -1),
    "a": ("wrist_roll",  -1),
    "d": ("wrist_roll",  +1),
    "z": ("gripper",     -1),  # close
    "c": ("gripper",     +1),  # open
}


class KeyState:
    def __init__(self):
        self.pressed_chars: set[str] = set()
        self.pressed_special: set[str] = set()
        self.should_quit = False
        self.should_print = False
        self.should_relax = False
        self.speed_bump = 0  # +1 / -1 / 0

    def on_press(self, key):
        # Exit
        if key == keyboard.Key.esc:
            self.should_quit = True
            return False

        # Special keys (arrows)
        if isinstance(key, keyboard.Key):
            name = key.name
            if name in KEY_TO_JOINT_SPECIAL:
                self.pressed_special.add(name)
                if DEBUG:
                    print(f"[press] <{name}>")
            return

        # Char keys
        if hasattr(key, "char") and key.char is not None:
            c = key.char.lower()
            if c == "x":
                self.should_quit = True
                return False
            if c == "p":
                self.should_print = True
                return
            if c == "r":
                self.should_relax = True
                return
            if c in ("+", "="):
                self.speed_bump = +1
                return
            if c == "-":
                self.speed_bump = -1
                return
            self.pressed_chars.add(c)
            if DEBUG:
                print(f"[press] {c!r}")

    def on_release(self, key):
        if isinstance(key, keyboard.Key):
            if key.name in KEY_TO_JOINT_SPECIAL:
                self.pressed_special.discard(key.name)
            return
        if hasattr(key, "char") and key.char is not None:
            self.pressed_chars.discard(key.char.lower())


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def print_help(step_deg):
    print("\n" + "=" * 60)
    print("  SO-101 follower test — keyboard layout")
    print("=" * 60)
    print("  Shoulder:  ←/→ pan     ↑/↓ lift")
    print("  Left hand: W/S elbow   Q/E wrist_flex   A/D wrist_roll")
    print("  Gripper:   Z close     C open")
    print(f"  Speed: +/-  (now {step_deg:.2f}°/tick)")
    print("  R = relax (reset target)   P = print   X/ESC = exit")
    print("=" * 60 + "\n")


def main():
    cfg = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
    robot = SO101Follower(cfg)
    print(f"Connecting to {PORT} (id={ROBOT_ID})...")
    robot.connect()
    print("Connected.")

    obs = robot.get_observation()
    target = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
    print("Initial position:")
    for k, v in target.items():
        print(f"  {k}: {v:.2f}")

    step_deg = STEP_DEG
    print_help(step_deg)

    state = KeyState()
    listener = keyboard.Listener(on_press=state.on_press, on_release=state.on_release)
    listener.start()

    period = 1.0 / LOOP_HZ
    try:
        while not state.should_quit:
            t0 = time.perf_counter()

            # Speed
            if state.speed_bump != 0:
                step_deg = clamp(step_deg + state.speed_bump * STEP_DELTA, STEP_MIN, STEP_MAX)
                print(f"[speed] step = {step_deg:.2f}°/tick")
                state.speed_bump = 0

            # Relax: re-anchor target to current position
            if state.should_relax:
                obs = robot.get_observation()
                target = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
                print("[relax] target re-aligned to present")
                state.should_relax = False

            # Apply step for active keys
            active = [(name, KEY_TO_JOINT_SPECIAL[name]) for name in state.pressed_special]
            active += [(c, KEY_TO_JOINT_CHAR[c]) for c in state.pressed_chars if c in KEY_TO_JOINT_CHAR]
            for _, (joint, sign) in active:
                step = GRIPPER_STEP if joint == "gripper" else step_deg
                target[f"{joint}.pos"] += sign * step

            # Clamp gripper in 0..100
            if "gripper.pos" in target:
                target["gripper.pos"] = clamp(target["gripper.pos"], 0.0, 100.0)

            # Anti-beep: limit goal-present distance
            obs = robot.get_observation()
            for k, v in obs.items():
                if not k.endswith(".pos") or k not in target:
                    continue
                lead = MAX_LEAD_GRIPPER if k == "gripper.pos" else MAX_LEAD_DEG
                present = float(v)
                target[k] = clamp(target[k], present - lead, present + lead)

            robot.send_action(target)

            if state.should_print:
                print("--- current positions ---")
                for k, v in obs.items():
                    if k.endswith(".pos"):
                        print(f"  {k}: {float(v):.2f}")
                state.should_print = False

            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[interrupt]")
    finally:
        listener.stop()
        print("Disconnecting...")
        robot.disconnect()
        print("Bye.")


if __name__ == "__main__":
    main()
