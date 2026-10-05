#!/usr/bin/env python
"""
Live (x, y, z) read-out of the SO-101 end-effector.

Move the arm by hand (torque DISABLED) and watch the EE position update on
the terminal. Use this to:
  - measure Z_PAPER and target XYs for touch_xy.py,
  - sanity-check FK vs the URDF,
  - learn the workspace by hand.

Press P (then Enter) to "pin" the current pose to the scroll log so you can
copy it later. Ctrl-C to quit. On exit, torque is re-enabled holding the
current pose so the arm doesn't flop.
"""

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from ikpy.chain import Chain

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ====== CONFIG ======
PORT = os.environ.get("SO101_PORT", "COM5")
ROBOT_ID = os.environ.get("SO101_ROBOT_ID", "team08_follower_arm")
URDF_PATH = str(
    Path(__file__).resolve().parent.parent / "SO101" / "so101_new_calib.urdf"
)
LOOP_HZ = 20
# Tip position in the gripper_frame_link LOCAL frame (m).
# That frame's +Z points toward the gripper tip (urdf joint has rpy=0,pi,0),
# so a typical SO-101 fingertip is roughly +10 cm along local Z. Refine with
# a caliper: close the jaws, measure from gripper_frame_link origin to the
# tip of the closed jaws, set the components accordingly.
TOOL_TIP_OFFSET_LOCAL_XYZ = (-0.04278, +0.00196, +0.01283)

# ============================

BODY_MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def build_chain(urdf_path):
    tmp = Chain.from_urdf_file(urdf_path)
    mask = [
        getattr(l, "joint_type", None) in ("revolute", "prismatic", "continuous")
        for l in tmp.links
    ]
    if sum(mask) != len(BODY_MOTOR_ORDER):
        raise SystemExit(
            f"[chain] {sum(mask)} movable joints, expected {len(BODY_MOTOR_ORDER)}. "
            f"Check URDF_PATH."
        )
    chain = Chain.from_urdf_file(urdf_path, active_links_mask=mask)
    chain._motor_link_idx = dict(
        zip(BODY_MOTOR_ORDER, [i for i, m in enumerate(mask) if m])
    )
    return chain


def motor_deg_to_full_rad(chain, q_body_deg):
    full = np.zeros(len(chain.links))
    for i, name in enumerate(BODY_MOTOR_ORDER):
        full[chain._motor_link_idx[name]] = np.deg2rad(float(q_body_deg[i]))
    return full


def fk_tip_pos(chain, q_body_deg, tip_local_xyz):
    """Position of the gripper TIP in base frame.

    The tip is offset from gripper_frame_link by `tip_local_xyz` expressed in
    the gripper's LOCAL frame; the full 4x4 FK transform takes care of any
    wrist orientation.
    """
    T = np.asarray(chain.forward_kinematics(motor_deg_to_full_rad(chain, q_body_deg)))
    tip_local_h = np.append(np.asarray(tip_local_xyz, dtype=float), 1.0)
    return (T @ tip_local_h)[:3]


class PinFlag:
    """Set by the stdin thread when the user types 'p'<Enter>."""

    def __init__(self):
        self.flag = False
        self.quit = False

    def listen(self):
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "p":
                self.flag = True
            elif cmd in ("q", "x", "quit", "exit"):
                self.quit = True
                return


def main():
    chain = build_chain(URDF_PATH)

    cfg = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
    robot = SO101Follower(cfg)
    print(f"[connect] {PORT} id={ROBOT_ID}")
    robot.connect()
    robot.bus.disable_torque()
    print("[connect] OK -- torque DISABLED, move the arm freely.")
    print("Type 'p' + Enter to pin the current pose to the log.")
    print("Type 'q' + Enter or Ctrl-C to quit.\n")

    pin = PinFlag()
    threading.Thread(target=pin.listen, daemon=True).start()

    period = 1.0 / LOOP_HZ
    try:
        while not pin.quit:
            t0 = time.perf_counter()
            obs = robot.get_observation()
            q = np.array(
                [float(obs[f"{n}.pos"]) for n in BODY_MOTOR_ORDER], dtype=float
            )
            p = fk_tip_pos(chain, q, TOOL_TIP_OFFSET_LOCAL_XYZ)

            live = (
                f"  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}  m   "
                f"|  q(deg) = {np.round(q, 1).tolist()}"
            )
            print(live + "  ", end="\r", flush=True)

            if pin.flag:
                pin.flag = False
                # Newline so the live line stays in the scrollback.
                print(live)

            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[stop]")
    finally:
        # Hold the current pose so the arm doesn't flop when torque re-enables.
        obs = robot.get_observation()
        hold = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
        robot.bus.enable_torque()
        robot.send_action(hold)
        print("\n[disconnect] re-enabled torque holding current pose")
        robot.disconnect()
        print("Bye.")


if __name__ == "__main__":
    main()
