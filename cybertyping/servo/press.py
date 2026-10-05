"""Task recipes built from the servo primitives.

* :func:`wait_for_start`  live preview with overlays; ENTER starts (or
                          auto-start when both signals are visible).
* :func:`press_target`    three-phase press: diagonal approach, planar
                          refinement, vertical contact descent, retreat.
* :func:`hover_target`    Eval-1 variant: planar servo at scan height, then
                          a closed-loop vertical descent to a hover height,
                          hold, return. Never touches the paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from cybertyping.core import primitives as prim
from cybertyping.servo.camera import draw_cross, put_status
from cybertyping.servo.control import (
    descend_to_z, descend_until_contact, servo_to_target,
)
from cybertyping.servo.rig import Rig
from cybertyping.servo.targets import Target, find_tip

GREEN = (0, 255, 0)


@dataclass
class ServoSettings:
    """Speed/accuracy trade-off for one press. ``fast()`` is used by the
    bonus eval; the defaults are the demo-day values."""
    tol_px: float = 8.0
    loop_hz: float = 10.0
    max_iters: int = 400
    hover_m: float = 0.015          # tip height above plane_z for the XY phases
    descent_speed_mps: float = 0.005
    max_descent_m: float = 0.05
    load_thresh: int = 60
    stall_thresh_deg: float = 10.0
    confirm: bool = True            # wait for ENTER before each motion

    @classmethod
    def fast(cls) -> "ServoSettings":
        return cls(tol_px=10.0, loop_hz=15.0, max_iters=300, confirm=False,
                   descent_speed_mps=0.008)


def wait_for_start(rig: Rig, target: Target, tip_color: str, *,
                   confirm: bool = True, auto_start_after_s: float = 1.0,
                   timeout_s: float | None = None):
    """Show the live feed with tip/target overlays until the operator presses
    ENTER (``confirm=True``) or both signals have been visible for
    ``auto_start_after_s`` seconds (``confirm=False``).

    Returns ``(ok, last_tip_px, last_target_px)``. ``ok`` is False on q/ESC
    or timeout. The last seen pixels seed the servo so it can start moving
    even if a signal flickers out at the moment of launch.
    """
    hint = "ENTER=start  q/ESC=skip" if confirm else "auto-start when both visible  q/ESC=skip"
    print(f"[wait] target '{target.label}': {hint}")
    last_tip = last_target = None
    both_since = None
    t0 = time.time()
    while True:
        frame = rig.read_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        tp = find_tip(frame, tip_color)
        tg = target.update(frame)
        if tp is not None:
            last_tip = tp
            draw_cross(frame, tp, GREEN, 25, 3)
        if tg is not None:
            last_target = tg
        target.draw(frame)
        put_status(frame, f"{tip_color.upper()} tip -> {target.label.upper()}. {hint}", scale=0.55)
        key = rig.show(frame, 20)
        if key in (ord("q"), 27):
            return False, last_tip, last_target
        if confirm and key == 13:
            return True, last_tip, last_target
        if not confirm:
            if tp is not None and tg is not None:
                both_since = both_since or time.time()
                if time.time() - both_since >= auto_start_after_s:
                    return True, last_tip, last_target
            else:
                both_since = None
        if timeout_s is not None and time.time() - t0 > timeout_s:
            print("[wait] timeout")
            return False, last_tip, last_target


def press_target(rig: Rig, target: Target, tip_color: str,
                 settings: ServoSettings | None = None,
                 return_to_scan: bool = True) -> dict:
    """Press ``target`` with the tip. Starts from wherever the arm is
    (normally the scan pose) and ends back at the scan pose.

    Phases:
      1. DIAGONAL  XY servo toward the target while Z descends to
                   ``plane_z + hover_m``; exits when Z is reached.
      2. PLANAR    pure XY refinement at the hover height until the pixel
                   error is below ``tol_px``.
      3. VERTICAL  XY frozen, straight descent until contact.
      RETREAT      back to the hover pose, then to the scan pose.

    Returns a dict with ``ok`` and the phase-3 ``info``.
    """
    s = settings or ServoSettings()
    t_start = time.perf_counter()
    hover_z = rig.plane_z + s.hover_m

    ok, tip_px, target_px = wait_for_start(rig, target, tip_color, confirm=s.confirm)
    if not ok:
        return {"ok": False, "reason": "skipped by user", "label": target.label}

    print(f"[phase 1] diagonal toward '{target.label}', z -> {hover_z:+.4f} m")
    q_hover = servo_to_target(
        rig, rig.read_joints_deg(), target, tip_color,
        target_z=hover_z, tol_px=s.tol_px, max_iters=s.max_iters,
        loop_hz=s.loop_hz, exit_mode="z_only", phase_label="phase 1",
        initial_tip_px=tip_px, initial_target_px=target_px,
    )
    print("[phase 2] planar refinement at hover height")
    q_target = servo_to_target(
        rig, q_hover, target, tip_color,
        target_z=None, tol_px=s.tol_px, max_iters=s.max_iters,
        loop_hz=s.loop_hz, exit_mode="xy_only", phase_label="phase 2",
    )
    print("[phase 3] vertical descent until contact")
    q_contact, info = descend_until_contact(
        rig, q_target, target=target,
        descent_speed_mps=s.descent_speed_mps, max_descent_m=s.max_descent_m,
        stall_thresh_deg=s.stall_thresh_deg, load_thresh=s.load_thresh,
    )
    print(f"[phase 3] {info['reason']} after {info['z_traveled_mm']:.1f} mm")

    print("[retreat] back to hover")
    ticks = prim.slew_segment(q_contact, q_target, prim.MAX_STEP_DEG_PER_TICK_RETREAT)
    prim.stream(rig.robot, ticks, rig.gripper_pos)
    time.sleep(0.2)
    if return_to_scan:
        rig.go_to_scan_pose()

    contact = "max descent" not in info["reason"] and "singular" not in info["reason"]
    return {"ok": contact, "label": target.label, "info": info,
            "elapsed_s": time.perf_counter() - t_start}


def hover_target(rig: Rig, target: Target, tip_color: str, *,
                 final_hover_m: float = 0.010, hold_s: float = 1.0,
                 coarse_tol_px: float = 35.0, confirm: bool = True,
                 return_to_scan: bool = True) -> dict:
    """Hover the tip ``final_hover_m`` above ``target`` (Eval 1).

    Phase 1 servos in XY at the scan-pose height until the pixel error is
    below ``coarse_tol_px`` (loose: parallax at that height is large).
    Phase 2 descends vertically to ``plane_z + final_hover_m`` while the
    closed-loop XY hold cancels the parallax drift. The pose is held for
    ``hold_s`` seconds, then the arm returns to the scan pose.
    """
    t_start = time.perf_counter()
    ok, tip_px, target_px = wait_for_start(rig, target, tip_color, confirm=confirm)
    if not ok:
        return {"ok": False, "reason": "skipped by user", "label": target.label}

    print(f"[phase 1] planar XY toward '{target.label}' at scan height")
    q_xy = servo_to_target(
        rig, rig.read_joints_deg(), target, tip_color,
        target_z=None, tol_px=coarse_tol_px, max_iters=400, loop_hz=15,
        exit_mode="xy_only", phase_label=f"phase 1 ({target.label})",
        wait_for_signals=False,
        initial_tip_px=tip_px, initial_target_px=target_px,
    )
    target_z = rig.plane_z + final_hover_m
    print(f"[phase 2] vertical descent to z = {target_z:+.4f} m with XY hold")
    q_hover = descend_to_z(rig, q_xy, target, target_z,
                           descent_speed_mps=0.005, loop_hz=15)

    print(f"[hold] {hold_s:.1f} s above '{target.label}'")
    t_end = time.time() + hold_s
    while time.time() < t_end:
        frame = rig.read_frame()
        if frame is not None:
            put_status(frame, f"[hold {target.label}]", GREEN)
            rig.show(frame, 20)

    if return_to_scan:
        rig.go_to_scan_pose()
    return {"ok": True, "label": target.label, "q_hover": q_hover,
            "elapsed_s": time.perf_counter() - t_start}
