"""Extrinsics round-trip test: project a known base-frame point, back-project, recover."""

import numpy as np
import cv2

from cybertyping.perception.extrinsics.fov_approx import (
    CameraIntrinsics, EyeToHandExtrinsics, PlaneModel,
)


def test_pixel_to_table_roundtrip():
    K = CameraIntrinsics.from_hfov(1280, 960, hfov_deg=60.0).K
    T_base_cam = np.eye(4)
    T_base_cam[:3, 3] = [0.20, 0.0, 0.40]
    R = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]
    T_base_cam[:3, :3] = R
    plane = PlaneModel(a=0.0, b=0.0, c=0.05)
    extr = EyeToHandExtrinsics(K=K, dist=None, T_base_cam=T_base_cam, plane=plane)

    for target in [
        np.array([0.25, 0.05, 0.05, 1.0]),
        np.array([0.15, -0.04, 0.05, 1.0]),
        np.array([0.22, 0.00, 0.05, 1.0]),
    ]:
        cam = np.linalg.inv(T_base_cam) @ target
        px = K @ (cam[:3] / cam[2])
        rx, ry, rz = extr.pixel_to_table(float(px[0]), float(px[1]))
        err = np.linalg.norm(np.array([rx, ry, rz]) - target[:3])
        assert err < 1e-4, f"round-trip error too large: {err}"
