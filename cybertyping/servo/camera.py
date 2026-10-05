"""OpenCV capture helpers shared by the servo loop and the diagnostic scripts.

The wrist camera is a plain UVC webcam. Which OpenCV backend works depends on
the operating system (Media Foundation / DirectShow on Windows, AVFoundation
on macOS, V4L2 on Linux), so :func:`open_camera` walks a per-platform chain
and returns the first backend that opens.
"""

from __future__ import annotations

import os
import platform

import cv2
import numpy as np

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


def backend_chain() -> list[int]:
    """Capture backends to try, in preference order for this platform."""
    sysname = platform.system()
    if sysname == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sysname == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def default_camera_index() -> int:
    """``CAMERA_INDEX`` env var, else 1 (index 0 is usually the laptop cam)."""
    return int(os.environ.get("CAMERA_INDEX", "1"))


def open_camera(index: int, width: int = DEFAULT_WIDTH,
                height: int = DEFAULT_HEIGHT) -> cv2.VideoCapture | None:
    """Open camera ``index`` with the first backend that works, or None."""
    for backend in backend_chain():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"[camera] opened index {index}")
        return cap
    return None


def read_frame(cap: cv2.VideoCapture, flush: int = 1) -> np.ndarray | None:
    """Return the freshest frame, discarding ``flush`` buffered ones first."""
    for _ in range(max(0, flush)):
        cap.read()
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def put_status(frame: np.ndarray, text: str, color=(0, 255, 255),
               y: int = 30, scale: float = 0.6) -> None:
    """Draw one line of status text at the top of the frame (in place)."""
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)


def draw_cross(frame: np.ndarray, xy, color, size: int = 20,
               thickness: int = 2) -> None:
    """Draw a cross marker at pixel ``xy`` (in place)."""
    pt = tuple(int(round(v)) for v in xy)
    cv2.drawMarker(frame, pt, color, cv2.MARKER_CROSS, size, thickness)
