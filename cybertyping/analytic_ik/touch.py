"""Rolling-baseline contact detector.

Watches Present_Load on shoulder_lift + elbow_flex (the two joints that bear
the vertical reaction force when the EE presses down). A signed deviation
from the rolling baseline that persists for `persist` consecutive ticks
triggers contact.

Matches the convention used elsewhere in the repo
(cybertyping/core/primitives.py and cybertyping/core/descent_jac.py).
"""
from __future__ import annotations

import collections
import time
from typing import Iterable


class TouchDetector:
    def __init__(self, robot, motor_names: Iterable[str], *,
                 window: int = 20, threshold: int = 80,
                 persist: int = 3, sign: int = +1):
        """
        Args:
            robot:       a connected lerobot SO101Follower (has `.bus`).
            motor_names: motors to watch (typically shoulder_lift + elbow_flex).
            window:      rolling baseline length in ticks.
            threshold:   |Δ load - baseline| required to count as a spike (raw).
            persist:     consecutive spike ticks required to fire contact.
            sign:        +1 if pushing down increases load in the sensed
                         direction; -1 otherwise. Calibrate empirically.
        """
        self.robot = robot
        self.motor_names = list(motor_names)
        self.window = int(window)
        self.threshold = int(threshold)
        self.persist = int(persist)
        self.sign = int(sign)
        self.history: dict[str, collections.deque] = {
            n: collections.deque(maxlen=self.window) for n in self.motor_names
        }
        self.streak = 0

    def _read_loads(self) -> dict[str, int]:
        loads = self.robot.bus.sync_read("Present_Load")
        return {n: int(loads[n]) for n in self.motor_names}

    def prime(self, ticks: int = 30, dt: float = 0.01) -> None:
        """Fill the rolling buffer. Run while the arm is stationary (or in
        steady-state descent — matches touch_sequence_load.py's convention)."""
        for _ in range(ticks):
            loads = self._read_loads()
            for n in self.motor_names:
                self.history[n].append(loads[n])
            time.sleep(dt)

    def __call__(self) -> bool:
        loads = self._read_loads()
        spiking = False
        for n in self.motor_names:
            hist = self.history[n]
            if hist:
                baseline = sum(hist) / len(hist)
                deviation = self.sign * (loads[n] - baseline)
                if deviation > self.threshold:
                    spiking = True
            hist.append(loads[n])
        self.streak = self.streak + 1 if spiking else 0
        return self.streak >= self.persist
