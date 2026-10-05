"""Pixel -> robot transforms. Each file is one regime:

  fov_approx.py    eye-in-hand with a nominal FOV (no calibrated intrinsics)
  flat_plane.py    (future) eye-in-hand with calibrated intrinsics + plane

Plus a convenience helper for applying the transform to every entry in a
KeyMap (used after homography fits).
"""

from __future__ import annotations


def populate_keymap_robot_xyz(
    keymap,
    frame_shape: tuple[int, int, int],
    camera_pose_xyz: tuple[float, float, float],
    plane,
    hfov_deg: float = 60.0,
    *,
    cal=None,
    T_base_camera=None,
):
    """Fill in `entry.robot_xy` and `entry.robot_z` for every key in `keymap`.

    If `cal` (RigCalibration) and `T_base_camera` are provided, uses the
    calibrated pinhole + plane-intersect + bias mapping. Otherwise falls
    back to `legacy_fov_pixel_to_robot`.

    Returns the same keymap (mutated in place).
    """
    from .fov_approx import legacy_fov_pixel_to_robot

    h, w = frame_shape[:2]
    use_calibrated = cal is not None and T_base_camera is not None
    if use_calibrated:
        from .calibrated import pixel_to_workspace_xy, lookup_bias
        K = cal.K
        dist = cal.dist
        plane_z = cal.plane_z

    for entry in keymap.entries.values():
        px, py = entry.image_xy
        if use_calibrated:
            hit = pixel_to_workspace_xy(px, py, K, dist, T_base_camera, plane_z)
            if hit is not None:
                bx, by = lookup_bias(hit, cal.bias_samples, cal.pen_bias_scalar)
                entry.robot_xy = (float(hit[0] + bx), float(hit[1] + by))
                entry.robot_z = float(plane_z)
                continue
        # Either no calibration or back-projection failed; fall back.
        rx, ry, rz = legacy_fov_pixel_to_robot(
            px, py, w, h, camera_pose_xyz, plane, hfov_deg,
        )
        entry.robot_xy = (rx, ry)
        entry.robot_z = rz
    return keymap


__all__ = ["populate_keymap_robot_xyz"]
