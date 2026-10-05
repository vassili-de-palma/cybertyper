"""Single source of truth for SO-101 reach-and-touch constants.

All link geometry derived from SO101/so101_new_calib.urdf. See
kinematics.py header for the full derivation and sign conventions.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Link geometry (meters). Constants here are derived by PROBING the
# URDF-walking FK (kinematics._JOINTS), not by hand-deriving the rpy chain.
# The probe lives in test_kinematics.py — if you change the URDF, re-run it
# and update these numbers.
#
# Coordinate framework
# ====================
# "Arm frame" = world frame rotated about the shoulder_pan axis so that the
# arm plane (the plane containing shoulder_lift, elbow_flex, wrist_flex
# joints) lies in a constant-y_arm slice. With shoulder_pan = 0 the arm
# frame coincides with the world frame.
#
# At all-zero joint angles (URDF reference pose):
#   shoulder_lift joint:  (0.0692, -0.0183, 0.1166)   arm-frame xyz
#   elbow_flex   joint:   (0.0972, -0.0183, 0.2292)
#   wrist_flex   joint:   (0.2321, -0.0183, 0.2344)
#   EE (gripper_frame):   (0.3914,  0.0000, 0.2265)
#
# Crucially: every EE-down pose lands the EE at y_arm = 0 (the gripper's
# lateral offset exactly cancels the arm's lateral offset of -0.0183 m).
# So there is NO lateral-offset correction needed on shoulder_pan beyond
# the pan-axis x-offset (X_PAN below).
# ---------------------------------------------------------------------------

# Shoulder_pan axis offset in world x (from URDF base→shoulder joint origin).
X_PAN: float = 0.0388353

# Shoulder_lift joint location in the arm frame at q=0.
U_SHOULDER: float = 0.0692
V_SHOULDER: float = 0.1166

# Wrist_flex joint → EE constant offset in the arm frame (wf - ee). When
# the EE points straight down, this offset is INVARIANT — the wrist_flex
# joint always sits at this displacement from the EE regardless of where in
# the workspace the EE is. WF_VERT is what the spec calls "L3".
WF_FORWARD: float = 0.0079        # along arm-frame +x (forward)
WF_LATERAL: float = -0.0183       # along arm-frame +y (= arm plane offset)
WF_VERT:    float = 0.1592        # along world +z (above the EE)

# Planar link lengths between successive joint origins.
L1: float = 0.1160        # shoulder_lift → elbow_flex distance
L2: float = 0.1350        # elbow_flex → wrist_flex distance

# Built-in geometric angles (radians). ALPHA1 is the pitch (from horizontal)
# of the upper-arm vector at q=0; ALPHA12 is ALPHA1 minus the pitch of the
# lower-arm vector at q=0. These are the offsets between the standard 2R
# geometric angles and the actual joint angles:
#   q[1] (shoulder_lift) = ALPHA1  - theta2_geom
#   q[2] (elbow_flex)    = -ALPHA12 - theta3_geom
ALPHA1:  float = float(np.arctan2(0.1126, 0.028))           # ≈ 76.04°
ALPHA12: float = ALPHA1 - float(np.arctan2(0.0052, 0.1349)) # ≈ 73.83°

# ---------------------------------------------------------------------------
# Joint limits (radians) — from URDF <limit lower=... upper=.../>.
# Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.
# ---------------------------------------------------------------------------
JOINT_LIMITS_RAD: np.ndarray = np.array([
    [-1.91986,  1.91986],   # shoulder_pan
    [-1.74533,  1.74533],   # shoulder_lift
    [-1.69000,  1.69000],   # elbow_flex
    [-1.65806,  1.65806],   # wrist_flex
    [-2.74385,  2.84121],   # wrist_roll
])

# ---------------------------------------------------------------------------
# Lerobot servo → URDF zero-pose offset (degrees). Measured by diagnostic.py.
# Lerobot ships positions in servo-degree space; fk.py's existing convention
# is to treat them directly as URDF degrees (no offset). We keep that as the
# default; populate this dict from a diagnostic.py run if you find a residual
# bias between commanded and actual EE pose.
# ---------------------------------------------------------------------------
CALIBRATION_OFFSET_DEG: dict[str, float] = {
    "shoulder_pan":  0.0,
    "shoulder_lift": 0.0,
    "elbow_flex":    0.0,
    "wrist_flex":    0.0,
    "wrist_roll":    0.0,
}

# ---------------------------------------------------------------------------
# Motor bus config (passed through to lerobot's SO101FollowerConfig via
# cybertyping.core.runtime.maybe_robot, which already respects these env vars).
# ---------------------------------------------------------------------------
PORT: str      = os.environ.get("SO101_PORT", "COM5")
ROBOT_ID: str  = os.environ.get("SO101_ROBOT_ID", "team08_follower_arm")
URDF_PATH: str = str(
    Path(__file__).resolve().parents[2] / "SO101" / "so101_new_calib.urdf"
)

BODY_MOTOR_ORDER: list[str] = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
]
SHOULDER_NAME: str = "shoulder_lift"
ELBOW_NAME: str    = "elbow_flex"

# ---------------------------------------------------------------------------
# Motion parameters
# ---------------------------------------------------------------------------
Z_HOVER: float    = 0.05        # default hover height (m). Higher values run
                                # into the wrist_flex ±95° limit at most (x,y)
                                # because the EE-down constraint forces a
                                # large wrist bend. Override per-target.
Z_FLOOR: float    = -0.02       # safety floor; abort descent if reached
V_DESCENT: float  = 0.01        # 1 cm/s descent
CONTROL_HZ: int   = 100
WRIST_ROLL_NEUTRAL_DEG: float = 0.0    # held constant during reach + descent

# ---------------------------------------------------------------------------
# Touch detection (Present_Load on shoulder + elbow).
# Calibrate TOUCH_THRESHOLD with a "descend into empty air" run; see README.
# ---------------------------------------------------------------------------
TOUCH_WINDOW: int        = 20        # rolling baseline window (ticks)
TOUCH_THRESHOLD: int     = 80        # signed deviation from baseline (raw)
TOUCH_PERSIST: int       = 3         # consecutive ticks required
TOUCH_WARMUP_TICKS: int  = 30        # skip PID transient before priming baseline
