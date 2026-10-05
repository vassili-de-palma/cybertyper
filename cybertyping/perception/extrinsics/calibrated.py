"""Calibrated pixel -> robot mapping for the eval pipeline.

Loads `configs/rig_calibration.json` (produced by
`cybertyping.calibration.session_calibration` and refined by
`cybertyping.calibration.charuco_bias_calibration`) and back-projects
wrist-camera pixels through:

    1. cv2.undistortPoints with (K, dist)            -> normalised ray (x_n, y_n, 1)
    2. R_base_camera = R_base_gripper @ R_gripper_camera, t_base_camera = ...
    3. Intersect ray with the workspace plane z = plane_z
    4. Apply position-dependent pen bias from `bias_samples`
       (linear fit `bias(x, y) = a + b*x + c*y` when >= 3 samples,
        mean if 1-2, scalar fallback if none).

This is the pixel-to-plane back-projection of the legacy open-loop pipeline. Keeping it
into a shared module so `tasks/session.py` and `evals/eval1_dots.py`
benefit from the calibration without reimplementing it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RigCalibration:
    K: np.ndarray
    dist: np.ndarray
    T_gripper_camera: np.ndarray
    plane_z: float
    pen_bias_scalar: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float),
    )
    bias_samples: list = field(default_factory=list)
    reproj_error_px: float = 0.0
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "RigCalibration":
        path = Path(path)
        d = json.loads(path.read_text())
        return cls(
            K=np.array(d["K"], dtype=np.float64),
            dist=np.array(d["dist"], dtype=np.float64),
            T_gripper_camera=np.array(d["T_gripper_camera"], dtype=np.float64),
            plane_z=float(d["plane_z"]),
            pen_bias_scalar=np.array(
                d.get("pen_bias_xy", [0.0, 0.0]), dtype=float,
            ),
            bias_samples=list(d.get("bias_samples", [])),
            reproj_error_px=float(d.get("reproj_error_px", 0.0)),
            source_path=path,
        )

    @classmethod
    def try_load(cls, path: str | Path) -> "RigCalibration | None":
        p = Path(path)
        if not p.exists():
            return None
        try:
            return cls.load(p)
        except Exception as e:
            print(f"[calibrated] could not load {p}: {e!r}")
            return None

    def T_base_camera(self, T_base_gripper: np.ndarray) -> np.ndarray:
        return np.asarray(T_base_gripper) @ self.T_gripper_camera


DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "rig_calibration.json"


def pixel_to_workspace_xy(
    px: float, py: float,
    K: np.ndarray, dist: np.ndarray,
    T_base_camera: np.ndarray,
    plane_z: float,
) -> tuple[float, float] | None:
    """Back-project a pixel through the camera + intersect with workspace plane.

    Returns (x, y) in robot base meters at z = plane_z, or None if the ray
    is parallel to the plane / behind the camera.
    """
    pts = np.array([[[px, py]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, K, dist)
    x_n, y_n = float(und[0, 0, 0]), float(und[0, 0, 1])
    d_cam = np.array([x_n, y_n, 1.0])
    R_bc = T_base_camera[:3, :3]
    o_base = T_base_camera[:3, 3]
    d_base = R_bc @ d_cam
    if abs(d_base[2]) < 1e-9:
        return None
    t = (plane_z - o_base[2]) / d_base[2]
    if t <= 0:
        return None
    hit = o_base + t * d_base
    return float(hit[0]), float(hit[1])


def lookup_bias(
    target_xy: tuple[float, float],
    bias_samples: list,
    pen_bias_scalar: np.ndarray,
) -> tuple[float, float]:
    """Position-dependent pen bias from multi-touch samples.

    >= 3 samples -> linear fit `bias(x,y) = a + b*x + c*y` per axis.
    1-2 samples  -> mean of sample biases.
    0 samples    -> scalar `pen_bias_scalar`.
    """
    n = len(bias_samples)
    if n == 0:
        return float(pen_bias_scalar[0]), float(pen_bias_scalar[1])
    if n < 3:
        mean_bias = np.mean(
            [s["bias_xy"] for s in bias_samples], axis=0,
        )
        return float(mean_bias[0]), float(mean_bias[1])
    A = np.array(
        [[1.0, s["perception_target_xy"][0], s["perception_target_xy"][1]]
         for s in bias_samples],
    )
    bx = np.array([s["bias_xy"][0] for s in bias_samples])
    by = np.array([s["bias_xy"][1] for s in bias_samples])
    coef_x, *_ = np.linalg.lstsq(A, bx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, by, rcond=None)
    bias_x = coef_x[0] + coef_x[1] * target_xy[0] + coef_x[2] * target_xy[1]
    bias_y = coef_y[0] + coef_y[1] * target_xy[0] + coef_y[2] * target_xy[1]
    return float(bias_x), float(bias_y)


def pixel_to_robot(
    px: float, py: float,
    T_base_gripper: np.ndarray,
    cal: RigCalibration,
    *,
    apply_bias: bool = True,
) -> tuple[float, float, float] | None:
    """Full calibrated pixel -> (rx, ry, plane_z) mapping.

    Steps: undistort -> back-project -> intersect plane -> apply bias.
    Returns None if back-projection fails.
    """
    T_bc = cal.T_base_camera(T_base_gripper)
    hit = pixel_to_workspace_xy(px, py, cal.K, cal.dist, T_bc, cal.plane_z)
    if hit is None:
        return None
    x, y = hit
    if apply_bias:
        bx, by = lookup_bias((x, y), cal.bias_samples, cal.pen_bias_scalar)
        x += bx
        y += by
    return float(x), float(y), float(cal.plane_z)


__all__ = [
    "RigCalibration",
    "DEFAULT_PATH",
    "pixel_to_workspace_xy",
    "pixel_to_robot",
    "lookup_bias",
]
