"""Press verification helpers.

KeystrokeListener wraps pynput to confirm a robot press registered on the
control laptop. VisualRetreatVerifier re-images post-press to check the
tip retreated cleanly (catches mechanical jams the keystroke listener
would miss).

    with KeystrokeListener() as ks:
        # ... robot presses 'h' at t_press ...
        ok = ks.was_received("h", t_press, window_s=1.5)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass



@dataclass
class KeyEvent:
    timestamp: float
    label: str          # canonical label: 'a'-'z' | 'space' | 'enter'


_PYNPUT_TO_LABEL = {
    "Key.space": "space",
    "Key.enter": "enter",
    "Key.tab":   "tab",
}


def _pynput_key_to_label(key) -> str | None:
    """Map a pynput key event to our canonical label, or None to skip."""
    name = str(key)
    if name in _PYNPUT_TO_LABEL:
        return _PYNPUT_TO_LABEL[name]
    # printable single character (e.g. "'a'")
    if name.startswith("'") and name.endswith("'") and len(name) == 3:
        ch = name[1].lower()
        if ch.isalpha() or ch == " ":
            return "space" if ch == " " else ch
    return None


class KeystrokeListener:
    """Global keystroke listener. Use as a context manager.

    Thread-safe; events appended in the listener thread, polled in main.

    NOTE: pynput requires platform-specific permissions (macOS: accessibility
    + input monitoring; Linux: X server access). On platforms without
    permissions, the listener silently produces no events; the eval code
    still works, it just can't verify.
    """

    def __init__(self, ring_size: int = 256):
        self._events: deque[KeyEvent] = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        self._listener = None
        self._available = False

    def __enter__(self) -> "KeystrokeListener":
        try:
            from pynput import keyboard
        except ImportError:
            print("[verify] pynput not installed; keystroke verification disabled")
            return self
        try:
            self._listener = keyboard.Listener(on_press=self._on_press)
            self._listener.start()
            self._available = True
        except Exception as e:
            print(f"[verify] pynput listener failed to start: {e!r}")
        return self

    def __exit__(self, *exc) -> None:
        if self._listener is not None:
            try: self._listener.stop()
            except Exception: pass

    @property
    def available(self) -> bool:
        return self._available

    def _on_press(self, key) -> None:
        label = _pynput_key_to_label(key)
        if label is None: return
        with self._lock:
            self._events.append(KeyEvent(timestamp=time.time(), label=label))

    def was_received(self, label: str, since_t: float, window_s: float = 1.5
                     ) -> bool:
        """Return True if `label` was pressed in (since_t, since_t + window_s]."""
        end_t = since_t + window_s
        with self._lock:
            for ev in self._events:
                if since_t < ev.timestamp <= end_t and ev.label == label:
                    return True
        return False

    def drain(self) -> list[KeyEvent]:
        with self._lock:
            evs = list(self._events)
            self._events.clear()
        return evs



def verify_retreat_by_image(
    grab_frame, pixel_to_table,
    expected_robot_xy: tuple[float, float],
    tip_color: str = "orange",
    max_dwell_s: float = 0.3,
    success_distance_mm: float = 15.0,
) -> bool:
    """Re-image and confirm the tip is ABOVE the expected key (i.e. has
    retreated). 'Above' is signalled by the tip pixel mapping back to a
    robot XY within `success_distance_mm` of the expected target, if the
    tip is still on the key cap, the back-projection through the table
    plane gives a pixel near the target; if the tip has retreated, the
    back-projection drifts.

    NOTE: this is a coarse check (the table-plane back-projection isn't
    valid above the plane), but it catches gross mechanical failures.
    Use the keystroke listener for the real verification when you can.
    """
    from cybertyping.perception.tip_detector import detect_tip
    import numpy as np

    t_start = time.time()
    while time.time() - t_start < max_dwell_s:
        frame = grab_frame()
        det = detect_tip(frame, color=tip_color)
        if det is None:
            time.sleep(0.03); continue
        rx, ry, _ = pixel_to_table(det.pixel_xy[0], det.pixel_xy[1])
        d_mm = float(np.hypot(rx - expected_robot_xy[0],
                              ry - expected_robot_xy[1])) * 1000.0
        if d_mm > success_distance_mm:
            return True
        time.sleep(0.03)
    return False


if __name__ == "__main__":
    # Manual exercise: open the listener and dump events for 5s.
    print("Listening for keys for 5 seconds (press any keys)...")
    with KeystrokeListener() as ks:
        time.sleep(5.0)
        events = ks.drain()
    print(f"received {len(events)} events:")
    for ev in events:
        print(f"  t={ev.timestamp:.3f} label={ev.label}")
