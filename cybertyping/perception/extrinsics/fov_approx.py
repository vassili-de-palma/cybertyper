"""Pixel <-> robot-XY mapping.

Two regimes:

  EyeInHand:  camera is wrist-mounted; pose comes from FK on the URDF.
              Camera-on-gripper offset is a small constant transform we
              calibrate ONCE (see calibration/hand_eye.py).

  EyeToHand:  camera is rigidly mounted in the world (clamped tripod,
              propped laptop, etc.). Camera-to-base transform comes from
              a one-shot PnP solve on an ArUco marker placed on the table.

Both regimes expose the SAME interface:

    extr.pixel_to_table(px_x, px_y) -> (robot_x, robot_y, robot_z)

The `robot_z` returned is the z of the table/keyboard plane at that (x,y),
computed from the world_settings plane fit (a*x + b*y + c). The press
primitive (touch_sequence_load) descends from above with contact sensing,
so this z is treated as a soft starting point, not a hard target.

Important: the legacy `vision/` math approximates the wrist-camera optical
center as the FK tip position and assumes the optical axis is +z down.
That's wrong in detail (the camera mount has its own offset and tilt) but
the ERROR averages to a constant pixel-bias that the homography in
perception/homography.py absorbs via the anchor-key fit. As long as the
anchors are detected reliably, this is fine. See README for the full
treatment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np



@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics. fx/fy in pixels, cx/cy in pixels."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: np.ndarray | None = None  # (k1,k2,p1,p2,k3) optional

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0,    self.cx],
                         [0.0,    self.fy, self.cy],
                         [0.0,    0.0,    1.0]], dtype=np.float64)

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float = 60.0) -> "CameraIntrinsics":
        """Estimate intrinsics from horizontal FOV. Use ONLY as a fallback before
        a real calibration is available."""
        f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height, dist=None)

    @classmethod
    def load(cls, path: str | Path) -> "CameraIntrinsics":
        with open(path) as f:
            d = json.load(f)
        return cls(
            fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
            width=d["width"], height=d["height"],
            dist=np.array(d.get("dist"), dtype=np.float64) if d.get("dist") else None,
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump({
                "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "width": self.width, "height": self.height,
                "dist": self.dist.tolist() if self.dist is not None else None,
            }, f, indent=2)



@dataclass(frozen=True)
class PlaneModel:
    a: float
    b: float
    c: float

    def z_at(self, x: float, y: float) -> float:
        return self.a * x + self.b * y + self.c

    @classmethod
    def from_world_settings(cls, world: dict) -> "PlaneModel":
        p = world["plane"]
        return cls(a=float(p["a"]), b=float(p["b"]), c=float(p["c"]))



@dataclass
class EyeToHandExtrinsics:
    """Pose of camera in robot base frame: T_base_cam (4x4 homogeneous).

    Built once at start-of-session from a single ArUco detection (see
    calibration/hand_eye.py for the helper). After that we cast every
    image pixel onto the table plane analytically.
    """
    K: np.ndarray              # 3x3 intrinsics
    dist: np.ndarray | None    # distortion coeffs (or None)
    T_base_cam: np.ndarray     # 4x4
    plane: PlaneModel          # table/keyboard plane in robot base frame

    @property
    def R_base_cam(self) -> np.ndarray: return self.T_base_cam[:3, :3]
    @property
    def t_base_cam(self) -> np.ndarray: return self.T_base_cam[:3,  3]

    def pixel_to_ray_base(self, px: float, py: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (ray_origin_base, ray_dir_base) for a single pixel.
        Undistorts the pixel first if dist coeffs are available."""
        pts = np.array([[[px, py]]], dtype=np.float64)
        if self.dist is not None:
            pts = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        u, v = pts[0, 0]
        # ray in camera frame (z forward, x right, y down)
        d_cam = np.array([(u - self.K[0, 2]) / self.K[0, 0],
                          (v - self.K[1, 2]) / self.K[1, 1],
                          1.0])
        d_cam /= np.linalg.norm(d_cam)
        d_base = self.R_base_cam @ d_cam
        o_base = self.t_base_cam
        return o_base, d_base

    def pixel_to_table(self, px: float, py: float) -> tuple[float, float, float]:
        """Intersect the back-projected ray with z = a*x + b*y + c."""
        o, d = self.pixel_to_ray_base(px, py)
        # Solve  o.z + t*d.z = a*(o.x+t*d.x) + b*(o.y+t*d.y) + c
        # => t * (d.z - a*d.x - b*d.y) = a*o.x + b*o.y + c - o.z
        denom = d[2] - self.plane.a * d[0] - self.plane.b * d[1]
        if abs(denom) < 1e-9:
            raise ValueError("Camera ray is parallel to the table plane.")
        t = (self.plane.a * o[0] + self.plane.b * o[1] + self.plane.c - o[2]) / denom
        if t <= 0:
            raise ValueError("Plane intersection is behind the camera; check extrinsics.")
        p = o + t * d
        return (float(p[0]), float(p[1]), float(self.plane.z_at(p[0], p[1])))



@dataclass
class EyeInHandExtrinsics:
    """Wrist-camera extrinsics: T_gripper_cam constant, T_base_gripper from FK.

    Pixel-to-table is computed the same way as Eye-to-hand but with
    T_base_cam updated every time the arm moves.
    """
    K: np.ndarray
    dist: np.ndarray | None
    T_gripper_cam: np.ndarray   # 4x4 constant (calibration result)
    plane: PlaneModel

    def at_pose(self, T_base_gripper: np.ndarray) -> EyeToHandExtrinsics:
        T_base_cam = T_base_gripper @ self.T_gripper_cam
        return EyeToHandExtrinsics(
            K=self.K, dist=self.dist, T_base_cam=T_base_cam, plane=self.plane
        )



def legacy_fov_pixel_to_robot(
    px: float, py: float,
    frame_w: int, frame_h: int,
    camera_xyz_base: tuple[float, float, float],
    plane: PlaneModel,
    hfov_deg: float = 60.0,
) -> tuple[float, float, float]:
    """The math used by vision/keyboard_key_finder_2.py and qwen_press.py.

    Approximations (DOCUMENTED BUGS):
      - Treats the FK tip position as the camera optical center.
      - Assumes the optical axis is exactly straight down (+z in base frame).
      - Uses a fixed HFOV guess.

    Kept here so the eval pipeline can fall back to it if a real intrinsics +
    extrinsics aren't yet available. Replace ASAP with EyeInHandExtrinsics
    once you have a calibration.
    """
    cx, cy, cz = camera_xyz_base
    rz_plane = plane.z_at(cx, cy)
    camera_height = cz - rz_plane
    scale_m_per_px = camera_height * 2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0) / frame_w
    # Image axes (origin top-left, +y down) -> robot axes (assumption: image +x = robot +y,
    # image +y = robot -x, when the camera looks straight down with no roll). This
    # is the LEGACY convention; the homography pipeline ignores it because it
    # works directly in image space until the final reprojection.
    dx_px = px - frame_w / 2.0
    dy_px = py - frame_h / 2.0
    rx = cx - dy_px * scale_m_per_px
    ry = cy + dx_px * scale_m_per_px
    rz = plane.z_at(rx, ry)
    return (float(rx), float(ry), float(rz))



def load_plane(world_path: str | Path) -> PlaneModel:
    with open(world_path) as f:
        world = json.load(f)
    return PlaneModel.from_world_settings(world)


if __name__ == "__main__":
    # Sanity: build a synthetic eye-to-hand extrinsics, project a known robot point
    # to a pixel, then back-project and check round-trip.
    K = CameraIntrinsics.from_hfov(1280, 960, hfov_deg=60).K
    T_base_cam = np.eye(4)
    T_base_cam[:3, 3] = [0.20, 0.0, 0.40]   # camera 40cm above origin
    R = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]  # look straight down
    T_base_cam[:3, :3] = R
    plane = PlaneModel(a=0.0, b=0.0, c=0.05)
    extr = EyeToHandExtrinsics(K=K, dist=None, T_base_cam=T_base_cam, plane=plane)

    # Known table point:
    target = np.array([0.25, 0.05, 0.05, 1.0])
    cam = np.linalg.inv(T_base_cam) @ target
    px = K @ (cam[:3] / cam[2])
    print(f"projected pixel: ({px[0]:.2f}, {px[1]:.2f})")
    rx, ry, rz = extr.pixel_to_table(px[0], px[1])
    print(f"back-projected   : ({rx:.4f}, {ry:.4f}, {rz:.4f})")
    print(f"expected         : ({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f})")
