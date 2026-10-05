#!/usr/bin/env python
"""Preview every OpenCV camera index so you can see which is the wrist cam.

Shows a window per working camera. Read the title bar of each window to
identify the wrist camera index, then pass it to touch_color.py.
"""
from __future__ import annotations

import os
import platform
import sys

if platform.system() == "Windows":
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2


def main() -> int:
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    caps: dict[int, cv2.VideoCapture] = {}
    for i in range(6):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        if not ret:
            cap.release()
            continue
        caps[i] = cap
        print(f"[probe] camera index {i} is alive")

    if not caps:
        print("[probe] no working cameras found")
        return 1

    print("Press 'q' or ESC to quit.")
    while True:
        for i, cap in caps.items():
            ret, frame = cap.read()
            if ret and frame is not None:
                cv2.putText(frame, f"index {i}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow(f"camera {i}", frame)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
