"""
Analytical IK for the SO-101 with the wrist axis constrained to point
straight down (along -z_world).

CONVENTION
----------
Joint indexing matches BODY_MOTOR_ORDER from config.py:
    q[0] = shoulder_pan   (yaw; axis is world -z at the pan-axis offset)
    q[1] = shoulder_lift  (pitch)
    q[2] = elbow_flex     (pitch, axis parallel to shoulder_lift)
    q[3] = wrist_flex     (pitch, axis parallel to elbow_flex)
    q[4] = wrist_roll     (roll about the gripper axis; held at 0 here)

All angles are URDF-frame radians. The motion layer converts to/from
lerobot servo degrees.

DERIVATION
----------
1. Orientation closure. The three pitch joints have parallel axes, so the
   EE pitch from world horizontal is a linear function of (q[1], q[2], q[3]).
   We *auto-calibrate* the linear coefficients (C0, S1, S2, S3) at module
   load by probing the URDF-walking FK at four known poses:

       pitch_world(q) ≈ C0 + S1·q[1] + S2·q[2] + S3·q[3]

   "EE points straight down" ⇒ pitch_world = -π/2 ⇒
       q[3] = (-π/2 - C0 - S1·q[1] - S2·q[2]) / S3

2. Shoulder pan. The shoulder_pan axis is the vertical line at world
   x = X_PAN, y = 0. For an EE-down pose, the EE always lands at y_arm = 0
   in the arm frame (the gripper's perpendicular offset cancels the arm's),
   so:
       θ1 = -atan2(y, x - X_PAN)
       r_eff = sqrt((x - X_PAN)^2 + y^2)
       u_EE  = X_PAN + r_eff       (EE x in arm frame)

3. Wrist-flex joint position. When EE points down, the wrist_flex joint
   has a FIXED arm-frame offset from the EE: (WF_FORWARD, WF_LATERAL,
   WF_VERT). The 2R arm is solved for the wrist_flex (u, v) coordinates:

       u' = (u_EE + WF_FORWARD) - U_SHOULDER
       v' = (z   + WF_VERT)    - V_SHOULDER

4. 2R inverse. Standard law-of-cosines, elbow-up branch.

       D² = u'² + v'²
       cos θ3_geom = (D² - L1² - L2²) / (2 L1 L2)
       sin θ3_geom = -sqrt(1 - cos²)              (elbow up)
       θ3_geom = atan2(sin, cos)
       θ2_geom = atan2(v', u') - atan2(L2·sin, L1 + L2·cos)

5. Joint-angle mapping. The geometric 2R angles map to URDF joint values:

       q[1] = ALPHA1  - θ2_geom
       q[2] = -ALPHA12 - θ3_geom

   The signs and constants come from the URDF zero-pose configuration —
   see config.py for the empirical derivation.

The whole IK is closed-form. No iteration, no Jacobian, no ikpy.
"""
from __future__ import annotations

import numpy as np

from .config import (
    ALPHA1, ALPHA12, BODY_MOTOR_ORDER, CALIBRATION_OFFSET_DEG, JOINT_LIMITS_RAD,
    L1, L2, U_SHOULDER, V_SHOULDER, WF_FORWARD, WF_VERT, WRIST_ROLL_NEUTRAL_DEG,
    X_PAN,
)


class OutOfReach(Exception):
    """The requested Cartesian target is outside the arm's workspace."""


class JointLimitViolation(Exception):
    """The IK solution exceeds one or more URDF joint limits."""


# ============================================================================
# URDF-walking FK (numpy only) — used as the trusted oracle and as the
# coupling that auto-calibrates the wrist-closure constants below.
# ============================================================================

# From SO101/so101_new_calib.urdf. Order matches BODY_MOTOR_ORDER.
_JOINTS = [
    # (name, origin xyz, rpy, axis)
    ("shoulder_pan",
        [0.0388353, -8.97657e-09, 0.0624],
        [3.14159, 4.18253e-17, -3.14159],
        [0.0, 0.0, 1.0]),
    ("shoulder_lift",
        [-0.0303992, -0.0182778, -0.0542],
        [-1.5708, -1.5708, 0.0],
        [0.0, 0.0, 1.0]),
    ("elbow_flex",
        [-0.11257, -0.028, 1.73763e-16],
        [-3.63608e-16, 8.74301e-16, 1.5708],
        [0.0, 0.0, 1.0]),
    ("wrist_flex",
        [-0.1349, 0.0052, 3.62355e-17],
        [4.02456e-15, 8.67362e-16, -1.5708],
        [0.0, 0.0, 1.0]),
    ("wrist_roll",
        [5.55112e-17, -0.0611, 0.0181],
        [1.5708, 0.0486795, 3.14159],
        [0.0, 0.0, 1.0]),
]
# gripper_frame_joint (fixed) — gripper_link → gripper_frame_link
_TIP_OFFSET = [-0.0079, -0.000218121, -0.0981274]
_TIP_RPY    = [0.0, 3.14159, 0.0]


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy convention: R = R_z(yaw) · R_y(pitch) · R_x(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_rot(axis, q: float) -> np.ndarray:
    """Rodrigues — rotation about a unit axis by q radians."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1.0 - np.cos(q)) * (K @ K)


def _T(origin, rpy, axis, q: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_R(*rpy) @ _axis_rot(axis, q)
    T[:3, 3]  = origin
    return T


def fk_T(q_rad) -> np.ndarray:
    """Full 4x4 transform from base_link to gripper_frame_link."""
    q = np.asarray(q_rad, dtype=float).reshape(5)
    T = np.eye(4)
    for i, (_n, origin, rpy, axis) in enumerate(_JOINTS):
        T = T @ _T(origin, rpy, axis, float(q[i]))
    T_fixed = np.eye(4)
    T_fixed[:3, :3] = _rpy_to_R(*_TIP_RPY)
    T_fixed[:3, 3]  = _TIP_OFFSET
    return T @ T_fixed


def forward_kinematics(q_rad) -> tuple[float, float, float]:
    """Return (x, y, z) of gripper_frame_link origin for q_rad in URDF radians."""
    T = fk_T(q_rad)
    return float(T[0, 3]), float(T[1, 3]), float(T[2, 3])


def _ee_z_axis_world(q_rad) -> np.ndarray:
    """The gripper_frame_link's +z axis expressed in the base frame."""
    return fk_T(q_rad)[:3, 2]


def _ee_pitch_world(q_rad) -> float:
    """Signed pitch angle of the gripper +z axis below the world horizontal.
    -π/2 means pointing straight down (+z_local → -z_world)."""
    z = _ee_z_axis_world(q_rad)
    # In the in-plane formulation, the EE z-axis stays in the vertical plane
    # through the arm. We use atan2 of its z-component over its horizontal
    # magnitude to recover a single pitch angle.
    horiz = float(np.sqrt(z[0] ** 2 + z[1] ** 2))
    return float(np.arctan2(z[2], horiz))


# ============================================================================
# Auto-calibrated wrist-closure constants. Computed once at module import
# by probing FK at q=0 and unit-perturbations of each pitch joint. Avoids
# hand-deriving signs from the URDF rpy chain.
# ============================================================================

def _calibrate_wrist_closure() -> tuple[float, float, float, float]:
    """Return (C0, s1, s2, s3) such that:
       _ee_pitch_world(q) ≈ C0 + s1*q[1] + s2*q[2] + s3*q[3]
    valid in a neighborhood of q=0 with q[0]=q[4]=0. The three pitch axes
    are parallel, so the relation is linear within ±π.
    """
    base = _ee_pitch_world([0.0, 0.0, 0.0, 0.0, 0.0])
    p1   = _ee_pitch_world([0.0, 0.3, 0.0, 0.0, 0.0])
    p2   = _ee_pitch_world([0.0, 0.0, 0.3, 0.0, 0.0])
    p3   = _ee_pitch_world([0.0, 0.0, 0.0, 0.3, 0.0])
    s1 = (p1 - base) / 0.3
    s2 = (p2 - base) / 0.3
    s3 = (p3 - base) / 0.3
    return base, s1, s2, s3


_C0, _S1, _S2, _S3 = _calibrate_wrist_closure()


# ============================================================================
# Joint-angle ↔ lerobot servo-degree conversion
# ============================================================================

def lerobot_deg_to_urdf_rad(q_deg) -> np.ndarray:
    """Convert lerobot servo degrees to URDF radians (applies any offsets)."""
    q_deg = np.asarray(q_deg, dtype=float).reshape(5)
    corrected = np.array([
        q_deg[i] - CALIBRATION_OFFSET_DEG[BODY_MOTOR_ORDER[i]] for i in range(5)
    ])
    return np.deg2rad(corrected)


def urdf_rad_to_lerobot_deg(q_rad) -> np.ndarray:
    """Convert URDF radians to lerobot servo degrees (inverse of the above)."""
    q_rad = np.asarray(q_rad, dtype=float).reshape(5)
    q_deg = np.rad2deg(q_rad)
    return np.array([
        q_deg[i] + CALIBRATION_OFFSET_DEG[BODY_MOTOR_ORDER[i]] for i in range(5)
    ])


# ============================================================================
# Joint-limit check
# ============================================================================

def _check_limits(q_rad: np.ndarray) -> None:
    for i, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
        if not (lo <= q_rad[i] <= hi):
            raise JointLimitViolation(
                f"{BODY_MOTOR_ORDER[i]}={np.rad2deg(q_rad[i]):+.2f}° "
                f"outside [{np.rad2deg(lo):+.2f}, {np.rad2deg(hi):+.2f}]"
            )


# ============================================================================
# Analytical IK
# ============================================================================

def solve_ik(x: float, y: float, z: float, *, elbow_up: bool = True) -> np.ndarray:
    """Solve for [θ1..θ5] (URDF radians) placing the EE at (x, y, z) with the
    wrist axis pointing straight down.

    Raises:
        OutOfReach            if the target is geometrically unreachable.
        JointLimitViolation   if the analytical solution exceeds a joint limit.
    """
    # ----- 1. Shoulder pan (axis at world x=X_PAN, y=0) ------------------
    dx = x - X_PAN
    r_eff = float(np.hypot(dx, y))
    if r_eff < 1e-6:
        # Target sits on the pan axis → orientation undefined.
        raise OutOfReach(f"target ({x:.3f}, {y:.3f}) coincides with pan axis")
    theta1 = float(-np.arctan2(y, dx))
    u_ee = X_PAN + r_eff      # EE x in arm frame

    # ----- 2. Wrist-flex joint position in the arm-frame (u, v) plane ----
    u_wf = u_ee + WF_FORWARD
    v_wf = z + WF_VERT

    # Translate to shoulder-lift origin.
    u_prime = u_wf - U_SHOULDER
    v_prime = v_wf - V_SHOULDER
    D_sq = u_prime * u_prime + v_prime * v_prime
    D = float(np.sqrt(D_sq))

    # ----- 3. 2R inverse for shoulder_lift + elbow_flex ------------------
    cos_t3 = (D_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if abs(cos_t3) > 1.0:
        raise OutOfReach(
            f"target ({x:.3f},{y:.3f},{z:.3f}): D={D:.3f}, |cos θ3_geom|={abs(cos_t3):.3f}"
        )
    if elbow_up:
        sin_t3 = -float(np.sqrt(max(0.0, 1.0 - cos_t3 * cos_t3)))
    else:
        sin_t3 = +float(np.sqrt(max(0.0, 1.0 - cos_t3 * cos_t3)))
    theta3_geom = float(np.arctan2(sin_t3, cos_t3))
    theta2_geom = float(
        np.arctan2(v_prime, u_prime)
        - np.arctan2(L2 * sin_t3, L1 + L2 * cos_t3)
    )

    # ----- 4. Map geometric 2R angles to URDF joint values ---------------
    theta2 = ALPHA1 - theta2_geom
    theta3 = -ALPHA12 - theta3_geom

    # ----- 5. Close orientation with wrist_flex --------------------------
    # pitch_world(q) = C0 + S1*q[1] + S2*q[2] + S3*q[3] = -π/2  (EE down)
    theta4 = (-np.pi / 2.0 - _C0 - _S1 * theta2 - _S2 * theta3) / _S3

    # ----- 6. Wrist roll held at neutral ---------------------------------
    theta5 = float(np.deg2rad(WRIST_ROLL_NEUTRAL_DEG))

    q = np.array([theta1, theta2, theta3, theta4, theta5], dtype=float)
    _check_limits(q)
    return q


__all__ = [
    "OutOfReach", "JointLimitViolation",
    "forward_kinematics", "fk_T", "solve_ik",
    "lerobot_deg_to_urdf_rad", "urdf_rad_to_lerobot_deg",
]
