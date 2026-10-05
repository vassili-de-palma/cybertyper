"""Wrist-mounted UVC webcam capture.

Cross-platform (Windows DSHOW / macOS AVFoundation / Linux V4L2). Waits
for autofocus to stabilise and flushes the driver buffer before each
`grab()`.
"""

from __future__ import annotations

import platform
import time

import cv2
import numpy as np


def _backend_for_platform() -> int:
    sys = platform.system()
    if sys == "Windows":
        return cv2.CAP_DSHOW
    if sys == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


class Camera:
    """Thin wrapper around cv2.VideoCapture.

    Usage:
        with Camera(index=0, width=1280, height=960) as cam:
            frame = cam.grab()
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 960,
        warmup_frames: int = 8,
        autofocus: bool | None = None,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames
        self.autofocus = autofocus
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(self.index, _backend_for_platform())
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.autofocus is not None:
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.autofocus else 0)
        for _ in range(self.warmup_frames):
            self._cap.read()
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def grab(self, flush: int = 3) -> np.ndarray:
        """Return a single fresh frame. `flush` re-reads to bypass driver buffer."""
        if self._cap is None:
            raise RuntimeError("Camera not opened. Use `with Camera(...) as cam:`")
        for _ in range(max(0, flush)):
            self._cap.read()
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return frame
