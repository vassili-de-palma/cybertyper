"""File-based mock capture — for dry-runs and offline tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class MockCamera:
    """A drop-in replacement that returns a single saved image.

    Usage:
        with MockCamera("data/keyboard_capture.jpg") as cam:
            frame = cam.grab()
    """

    def __init__(self, image_path: str | Path):
        self.path = Path(image_path)
        self._frame: np.ndarray | None = None

    def __enter__(self):
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._frame = cv2.imread(str(self.path))
        if self._frame is None:
            raise RuntimeError(f"cv2 could not decode {self.path}")
        return self

    def __exit__(self, *exc):
        self._frame = None

    def grab(self, flush: int = 0) -> np.ndarray:
        assert self._frame is not None, "Use MockCamera as context manager"
        return self._frame.copy()
