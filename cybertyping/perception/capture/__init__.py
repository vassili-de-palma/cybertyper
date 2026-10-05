"""Frame capture components.

Each file is one capture strategy with a common interface:

    with open_camera(...) as cam:
        frame = cam.grab()

- wrist_uvc.Camera:  real UVC webcam (Windows/macOS/Linux backends)
- file_mock.MockCamera:  returns a saved JPEG; for dry-runs and tests

`open_camera(mock_path=...)` is the convenience factory.
"""

from contextlib import contextmanager
from pathlib import Path

from .wrist_uvc import Camera
from .file_mock import MockCamera


@contextmanager
def open_camera(mock_path: str | Path | None = None, **kwargs):
    """Pick MockCamera if `mock_path` is given, else a real Camera."""
    if mock_path is not None:
        with MockCamera(mock_path) as cam:
            yield cam
    else:
        with Camera(**kwargs) as cam:
            yield cam


__all__ = ["Camera", "MockCamera", "open_camera"]
