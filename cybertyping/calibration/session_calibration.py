#!/usr/bin/env python
"""One-shot session calibration for SO-101 + wrist UVC camera.

Outputs configs/rig_calibration.json containing everything per-eval needs:

  - K (3x3 intrinsics) and distortion vector  (from cv2.calibrateCamera)
  - T_gripper_camera (4x4)                    (from cv2.calibrateHandEye)
  - plane_z                                    (from load-contact probe)
  - scan_pose joints + xyz                     (saved for fast re-slewing)
  - contact_threshold_baseline_noise           (auto-tuned from idle samples)

Run ONCE before evals (untimed setup). Evals then load this and pay zero
calibration time.

WORKFLOW (~5 minutes):
  1. Place the 5x7 A4 checkerboard flat in the workspace, roughly centered.
  2. Run:   python -m cybertyping.calibration.session_calibration
  3. Robot moves to ~12 varied poses; at each it snaps wrist cam and finds
     the board. ~30 seconds of motion.
  4. cv2.calibrateCamera (intrinsics) + cv2.calibrateHandEye (extrinsics)
     produce the rig calibration. Saved automatically.
  5. Robot probes plane height with a load-contact descent.
  6. Done. Run evals.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
except Exception:
    pass

from cybertyping.core.primitives import (  # noqa: E402
    BODY_MOTOR_ORDER,
    ENFORCE_GRIPPER_DOWN,
    LOCKED_JOINTS,
    LOOP_HZ,
    MAX_STEP_DEG_PER_TICK_APPROACH,
    MAX_STEP_DEG_PER_TICK_RETREAT,
    TOOL_TIP_OFFSET_LOCAL_XYZ,
    URDF_PATH,
    build_chain,
    fk_T,
    fk_tip,
    load_world,
    slew_segment,
    solve_ik_tip,
    stream,
)
from cybertyping.core.descent_jac import descent_jacobian_until_contact  # noqa: E402
from cybertyping.perception.capture.wrist_uvc import Camera  # noqa: E402
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402

WORLD_SETTING_PATH = REPO / "world_settings_load.json"
CONFIG_PATH = REPO / "configs" / "rig_calibration.json"
RUNS = REPO / "runs" / "session_cal"


# ---------- ChArUco board detection ----------
#
# The user's board has alternating black/white squares, with an ArUco marker
# embedded inside each white square. This is the OpenCV "ChArUco" pattern,
# which is more robust than a plain checkerboard: partial views work, and the
# ArUco IDs disambiguate corner identity (so rotation/occlusion don't confuse
# the detector).
#
# Detection pipeline:
#   1. cv2.aruco.detectMarkers   -> find all ArUco squares
#   2. cv2.aruco.interpolateCornersCharuco -> recover the chessboard inner
#      corners that border at least one detected marker, with their IDs
#
# `interpolateCornersCharuco` requires us to know which ArUco DICTIONARY was
# used when the board was printed. Standard for 35 mm squares is DICT_5X5_*
# or DICT_4X4_*. We auto-probe a few common dictionaries on the first frame
# and lock onto the one with the most detected markers.

_ARUCO_DICTS_TRY = [
    ("4X4_50",  cv2.aruco.DICT_4X4_50),
    ("4X4_100", cv2.aruco.DICT_4X4_100),
    ("4X4_250", cv2.aruco.DICT_4X4_250),
    ("5X5_50",  cv2.aruco.DICT_5X5_50),
    ("5X5_100", cv2.aruco.DICT_5X5_100),
    ("5X5_250", cv2.aruco.DICT_5X5_250),
    ("6X6_50",  cv2.aruco.DICT_6X6_50),
    ("6X6_250", cv2.aruco.DICT_6X6_250),
]


def autodetect_aruco_dict(frame_bgr: np.ndarray) -> tuple[int, str, int]:
    """Try common ArUco dictionaries; return (dict_const, name, n_detected)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    best = (None, None, 0)
    for name, dict_const in _ARUCO_DICTS_TRY:
        d = cv2.aruco.getPredefinedDictionary(dict_const)
        params = cv2.aruco.DetectorParameters()
        det = cv2.aruco.ArucoDetector(d, params)
        corners, ids, _ = det.detectMarkers(gray)
        n = 0 if ids is None else len(ids)
        if n > best[2]:
            best = (dict_const, name, n)
    return best  # may be (None, None, 0) if nothing detected


def detect_charuco(
    frame_bgr: np.ndarray,
    board: "cv2.aruco.CharucoBoard",
    aruco_dict: "cv2.aruco.Dictionary",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (corners_2d (M,1,2) float32, corner_ids (M,1) int32) or None.

    Uses the modern OpenCV 4.7+ CharucoDetector API (the old
    interpolateCornersCharuco was removed).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    charuco_det = cv2.aruco.CharucoDetector(board)
    corners_charuco, ids_charuco, marker_corners, marker_ids = (
        charuco_det.detectBoard(gray)
    )
    if (corners_charuco is None or ids_charuco is None
            or len(corners_charuco) < 6):
        return None
    return corners_charuco, ids_charuco


def estimate_pose_charuco(
    corners_charuco: np.ndarray,
    ids_charuco: np.ndarray,
    board: "cv2.aruco.CharucoBoard",
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (rvec, tvec) for the board pose in camera frame, via solvePnP
    over the matched 3D-2D corner pairs."""
    obj_pts_all = board.getChessboardCorners()  # (N_corners, 3) in board frame
    ids_flat = ids_charuco.flatten()
    obj_pts = np.array([obj_pts_all[i] for i in ids_flat], dtype=np.float32)
    img_pts = corners_charuco.reshape(-1, 2).astype(np.float32)
    if len(obj_pts) < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return rvec, tvec


# ---------- Pose generation ----------

def generate_capture_poses(
    world: dict,
) -> list[tuple[np.ndarray, np.ndarray | None, str]]:
    """Pose plan for ChArUco capture.

    Returns list of (xyz_target, orientation_target_or_None, label).

    For cv2.calibrateHandEye to be well-conditioned the poses must span
    multiple rotation axes (otherwise AX=XB is under-determined). On a 5-DOF
    SO-101 with wrist_roll locked, the only way to get rotation diversity is
    to *not* enforce strict gripper-down on every pose: we mix
        * 8 poses with gripper-down (best intrinsics images)
        * 6 poses with gripper tilted +/- 30 deg about base x (front/back tilt)
        * 6 poses with gripper tilted +/- 30 deg about base y (side tilt)
    Each tilt uses ikpy's target_orientation + orientation_mode='Z'.
    """
    ox, oy, oz = world["origin"]["x"], world["origin"]["y"], world["origin"]["z"]
    cx, cy = 0.20, 0.0
    poses: list[tuple[np.ndarray, np.ndarray | None, str]] = []

    # 8 gripper-down poses, spread across XY at two heights.
    for cz, ztag in [(oz + 0.12, "h"), (oz + 0.08, "l")]:
        for dx, dy in [(-0.03, -0.04), (-0.03, +0.04),
                       (+0.03, -0.04), (+0.03, +0.04)]:
            label = f"{ztag}_down_dx{int(dx*1000):+d}_dy{int(dy*1000):+d}"
            poses.append((np.array([cx + dx, cy + dy, cz]), None, label))

    # Tilted poses (rotation diversity for hand-eye). Each "tilt" sets the
    # gripper's +Z axis to a direction other than (0,0,-1). 30 deg off-axis is
    # the most we can ask before the arm hits joint limits.
    s = float(np.sin(np.deg2rad(30)))
    c = float(np.cos(np.deg2rad(30)))
    # +X tilt (gripper looks forward-down)
    poses.append((np.array([cx, cy, oz + 0.10]), np.array([+s, 0.0, -c]), "tilt_xp"))
    poses.append((np.array([cx + 0.02, cy, oz + 0.10]),
                  np.array([+s, 0.0, -c]), "tilt_xp_a"))
    poses.append((np.array([cx - 0.02, cy, oz + 0.10]),
                  np.array([+s, 0.0, -c]), "tilt_xp_b"))
    # -X tilt (gripper looks backward-down)
    poses.append((np.array([cx, cy, oz + 0.10]), np.array([-s, 0.0, -c]), "tilt_xn"))
    # +/- Y tilt (gripper looks side-down)
    poses.append((np.array([cx, cy + 0.02, oz + 0.10]),
                  np.array([0.0, +s, -c]), "tilt_yp"))
    poses.append((np.array([cx, cy - 0.02, oz + 0.10]),
                  np.array([0.0, -s, -c]), "tilt_yn"))
    return poses


# ---------- Robot motion helper ----------

def slew_safely(robot, chain, tip_off, q_from, target_xyz, gripper_pos,
                target_orientation=None):
    """Slew tip to target_xyz; if `target_orientation` is given, constrain the
    gripper's +Z axis to point that way (orientation_mode='Z')."""
    if target_orientation is None:
        q_try, res = solve_ik_tip(
            chain, target_xyz, q_from, tip_off, enforce_gripper_down=True,
        )
    else:
        # ikpy with orientation_mode='Z' and our custom solver wrapper
        import ikpy.chain as _ch  # noqa: F401
        # Use solve_ik_origin directly with target_orientation override
        target = np.asarray(target_xyz, dtype=float)
        tip = np.asarray(tip_off, dtype=float)
        q_full = q_from.copy()
        # iterate solve_ik with rotation
        for _ in range(5):
            T = np.asarray(
                chain.forward_kinematics(
                    np.array([np.deg2rad(q_full[i]) if i < len(BODY_MOTOR_ORDER) else 0
                              for i in range(len(chain.links))])
                )
            )
            R = T[:3, :3]
            # Re-target FK origin = tip_target - R @ tip_offset
            ik_target_origin = target - R @ tip
            from cybertyping.core.primitives import motor_deg_to_full_rad, full_rad_to_body_deg
            init_full = motor_deg_to_full_rad(chain, q_full)
            sol_full = chain.inverse_kinematics(
                target_position=ik_target_origin,
                target_orientation=np.asarray(target_orientation, dtype=float),
                orientation_mode="Z",
                initial_position=init_full,
            )
            q_full = full_rad_to_body_deg(chain, sol_full)
            # check residual
            T2 = np.asarray(
                chain.forward_kinematics(motor_deg_to_full_rad(chain, q_full))
            )
            tip_pos = (T2 @ np.append(tip, 1.0))[:3]
            res = float(np.linalg.norm(tip_pos - target))
            if res < 0.005:
                break
        q_try = q_full
    ticks = slew_segment(q_from, q_try, MAX_STEP_DEG_PER_TICK_APPROACH)
    stream(robot, ticks, gripper_pos)
    # Hold for settle
    period = 1.0 / LOOP_HZ
    action = {f"{n}.pos": float(q_try[i]) for i, n in enumerate(BODY_MOTOR_ORDER)}
    action["gripper.pos"] = gripper_pos
    for _ in range(int(0.6 * LOOP_HZ)):
        robot.send_action(action)
        time.sleep(period)
    obs = robot.get_observation()
    q_at = np.array([float(obs[f"{n}.pos"]) for n in BODY_MOTOR_ORDER], dtype=float)
    return q_at, res


# ---------- Main ----------

@dataclass
class RigCalibration:
    K: list[list[float]]
    dist: list[float]
    image_size: list[int]              # [w, h]
    reproj_error_px: float
    T_gripper_camera: list[list[float]]  # 4x4
    hand_eye_method: str
    plane_z: float
    scan_pose_q_deg: list[float] | None
    scan_pose_xyz: list[float] | None
    n_poses: int
    timestamp: int
    notes: str = ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1, help="wrist UVC index")
    ap.add_argument("--squares-x", type=int, default=7,
                    help="number of SQUARES along x (board cols) -- 7 for landscape A4")
    ap.add_argument("--squares-y", type=int, default=5,
                    help="number of SQUARES along y (board rows) -- 5 for landscape A4")
    ap.add_argument("--square-size-mm", type=float, default=35.0,
                    help="ChArUco square edge in mm (user's: 35 mm)")
    ap.add_argument("--marker-size-mm", type=float, default=30.0,
                    help="ArUco marker edge inside each white square in mm (user's: 30 mm)")
    ap.add_argument("--aruco-dict", type=str, default="auto",
                    help="ArUco dictionary name (e.g. 5X5_100) or 'auto' to probe")
    ap.add_argument("--gripper-open", type=float, default=100.0)
    ap.add_argument("--min-poses", type=int, default=6,
                    help="abort if fewer than this many poses got a valid detection")
    ap.add_argument("--probe-plane", action="store_true", default=True)
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    if not WORLD_SETTING_PATH.exists():
        raise SystemExit(f"[abort] {WORLD_SETTING_PATH} missing; "
                         "run touch_sequence_load.py first for the workspace bbox.")
    world = load_world(WORLD_SETTING_PATH)

    chain = build_chain(URDF_PATH, locked_joints=LOCKED_JOINTS)
    tip_off = np.asarray(TOOL_TIP_OFFSET_LOCAL_XYZ, dtype=float)

    port = os.environ.get("SO101_PORT", "COM5")
    robot_id = os.environ.get("SO101_ROBOT_ID", "team08_follower_arm")
    cfg = SO101FollowerConfig(port=port, id=robot_id, max_relative_target=None)
    robot = SO101Follower(cfg)
    print(f"[connect] {port} id={robot_id}")
    robot.connect()
    print("[connect] OK")

    ts = int(time.time())
    gripper_pos = float(args.gripper_open)
    square_size_m = args.square_size_mm / 1000.0
    marker_size_m = args.marker_size_mm / 1000.0

    # We don't know the ArUco dictionary yet -- need a frame to autodetect.
    # Created lazily after the first successful detection probe below.
    aruco_dict = None
    charuco_board = None

    all_corners: list[np.ndarray] = []   # per-pose (M,1,2) float32
    all_ids: list[np.ndarray] = []       # per-pose (M,1) int32
    pose_T_base_gripper: list[np.ndarray] = []
    pose_labels: list[str] = []
    image_size: tuple[int, int] | None = None
    plane_z = float(world["origin"]["z"])   # default; overwritten by probe

    try:
        # ---------- Pre-open gripper ----------
        obs = robot.get_observation()
        q_now = np.array(
            [float(obs[f"{n}.pos"]) for n in BODY_MOTOR_ORDER], dtype=float,
        )
        action = {f"{n}.pos": float(q_now[i]) for i, n in enumerate(BODY_MOTOR_ORDER)}
        action["gripper.pos"] = gripper_pos
        robot.send_action(action)
        time.sleep(0.3)

        # ---------- Multi-pose checkerboard capture ----------
        poses = generate_capture_poses(world)
        print(f"\n[cal] capturing {len(poses)} poses ...")
        for i, (target_xyz, target_orient, label) in enumerate(poses):
            print(f"\n[pose {i+1}/{len(poses)}] -> {label}  xyz={target_xyz.round(3).tolist()}"
                  + (f"  orient={np.asarray(target_orient).round(2).tolist()}"
                     if target_orient is not None else "  orient=down"))
            try:
                q_at, res = slew_safely(robot, chain, tip_off, q_now, target_xyz,
                                         gripper_pos, target_orientation=target_orient)
                print(f"  IK residual = {res*1000:.1f} mm")
            except Exception as e:
                print(f"  slew failed: {e}; skipping")
                continue
            q_now = q_at

            try:
                with Camera(index=args.camera, warmup_frames=10) as cam:
                    frame = cam.grab(flush=4)
            except Exception as e:
                print(f"  camera failed: {e}; skipping")
                continue
            cv2.imwrite(str(RUNS / f"cal_{ts}_pose{i:02d}_{label}.jpg"), frame)
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])

            # Lazy: autodetect ArUco dict + build CharucoBoard on first frame
            if aruco_dict is None:
                if args.aruco_dict == "auto":
                    dict_const, dict_name, n_det = autodetect_aruco_dict(frame)
                    if dict_const is None:
                        print(f"  [aruco] no markers found by ANY dictionary; skipping")
                        continue
                    print(f"  [aruco] auto-detected dict {dict_name} ({n_det} markers)")
                else:
                    dict_name = args.aruco_dict
                    dict_const = getattr(cv2.aruco, f"DICT_{dict_name}")
                aruco_dict = cv2.aruco.getPredefinedDictionary(dict_const)
                charuco_board = cv2.aruco.CharucoBoard(
                    (args.squares_x, args.squares_y),
                    squareLength=square_size_m,
                    markerLength=marker_size_m,
                    dictionary=aruco_dict,
                )

            result = detect_charuco(frame, charuco_board, aruco_dict)
            if result is None:
                print(f"  [DETECT FAIL] ChArUco corners not interpolated")
                continue
            corners, ids = result
            print(f"  detected {len(corners)} ChArUco corners")

            # Annotated debug
            vis = frame.copy()
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
            cv2.imwrite(str(RUNS / f"cal_{ts}_pose{i:02d}_{label}_annot.jpg"), vis)

            all_corners.append(corners)
            all_ids.append(ids)
            T_bg = fk_T(chain, q_at)
            pose_T_base_gripper.append(T_bg.copy())
            pose_labels.append(label)

        n_valid = len(all_corners)
        print(f"\n[cal] {n_valid}/{len(poses)} poses yielded valid checkerboard detections")
        if n_valid < args.min_poses:
            raise SystemExit(
                f"[abort] only {n_valid} valid poses (need {args.min_poses}). "
                "Check checkerboard placement / lighting / board-cols/rows args."
            )

        # ---------- Intrinsics ----------
        # Modern OpenCV: no aruco.calibrateCameraCharuco. Build per-pose
        # matched 3D/2D point lists from board.getChessboardCorners() and the
        # detected ids, then call standard cv2.calibrateCamera.
        print("\n[cal] running cv2.calibrateCamera on ChArUco-matched points ...")
        obj_pts_all = charuco_board.getChessboardCorners()   # (N, 3) board frame
        obj_pts_per_pose: list[np.ndarray] = []
        img_pts_per_pose: list[np.ndarray] = []
        for corners, ids in zip(all_corners, all_ids):
            ids_flat = ids.flatten()
            obj_pts = np.array([obj_pts_all[i] for i in ids_flat], dtype=np.float32)
            img_pts = corners.reshape(-1, 1, 2).astype(np.float32)
            obj_pts_per_pose.append(obj_pts.reshape(-1, 1, 3))
            img_pts_per_pose.append(img_pts)

        # Use standard 5-coefficient distortion model (k1, k2, p1, p2, k3) -
        # CALIB_RATIONAL_MODEL has 8 coefficients and is ill-conditioned with
        # the modest pose diversity we get on 5-DOF arm + workspace constraint.
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts_per_pose, img_pts_per_pose, image_size, None, None,
            flags=0,
        )
        print(f"[cal] intrinsics reprojection RMS = {ret:.3f} px  "
              f"(target < 0.5 px)")
        print(f"[cal] K =\n{K}")
        print(f"[cal] dist = {dist.flatten().round(5).tolist()}")

        # ---------- Hand-Eye (ChArUco) ----------
        # For each pose, get T_camera_board via estimatePoseCharucoBoard (uses
        # all detected ChArUco corners + their known 3D positions on the
        # printed board). More accurate than reusing calibrateCameraCharuco's
        # rvecs after rational-model fit.
        R_target2cam = []
        t_target2cam = []
        R_gripper2base = []
        t_gripper2base = []
        for corners, ids, T_bg in zip(all_corners, all_ids, pose_T_base_gripper):
            res = estimate_pose_charuco(corners, ids, charuco_board, K, dist)
            if res is None:
                continue
            rvec, tvec = res
            R_tc, _ = cv2.Rodrigues(rvec)
            R_target2cam.append(R_tc)
            t_target2cam.append(tvec.flatten())
            R_gripper2base.append(T_bg[:3, :3])
            t_gripper2base.append(T_bg[:3, 3])

        # Try all available methods and pick the one whose result is physically
        # plausible (translation magnitude < 30 cm and consistent with a
        # wrist-mounted camera). 5-DOF arms tend to fail one method or another
        # depending on which axes have diversity.
        method_candidates = [
            ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
            ("TSAI",       cv2.CALIB_HAND_EYE_TSAI),
            ("HORAUD",     cv2.CALIB_HAND_EYE_HORAUD),
            ("PARK",       cv2.CALIB_HAND_EYE_PARK),
            ("ANDREFF",    cv2.CALIB_HAND_EYE_ANDREFF),
        ]
        candidates: list[tuple[str, np.ndarray]] = []
        for mname, mflag in method_candidates:
            try:
                R_cg, t_cg = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=mflag,
                )
                T_cg = np.eye(4)
                T_cg[:3, :3] = R_cg
                T_cg[:3, 3] = t_cg.flatten()
                t_norm = float(np.linalg.norm(t_cg))
                print(f"[cal] hand-eye {mname:11s}: t = "
                      f"{(t_cg.flatten() * 1000).round(1).tolist()} mm  |t|={t_norm*1000:.0f} mm")
                if t_norm < 0.30:
                    candidates.append((mname, T_cg))
            except Exception as e:
                print(f"[cal] hand-eye {mname} FAILED: {e}")
        if not candidates:
            raise SystemExit("[abort] no hand-eye method produced a plausible result")
        # Pick consensus: median translation, then the method whose t is
        # closest to that median. With 5 candidates, outliers are excluded.
        all_t = np.stack([T[:3, 3] for _, T in candidates])
        t_median = np.median(all_t, axis=0)
        dists = [float(np.linalg.norm(T[:3, 3] - t_median)) for _, T in candidates]
        i_best = int(np.argmin(dists))
        method_name, T_gripper_camera = candidates[i_best]
        print(f"\n[cal] CHOSEN hand-eye (median consensus): {method_name}  "
              f"dist-to-median={dists[i_best]*1000:.1f} mm")
        print(f"[cal] T_gripper_camera =\n{T_gripper_camera}")

        # ---------- Plane probe (optional, via Jacobian descent w/ xy-drift abort) ----------
        if args.probe_plane:
            print("\n[cal] probing plane height via Jacobian descent at safe XY...")
            probe_xy = (0.18, 0.0)
            probe_hover = np.array([probe_xy[0], probe_xy[1], plane_z + 0.05])
            q_probe_pre, res_p = solve_ik_tip(
                chain, probe_hover, q_now, tip_off,
                enforce_gripper_down=ENFORCE_GRIPPER_DOWN,
            )
            ticks_p = slew_segment(q_now, q_probe_pre, MAX_STEP_DEG_PER_TICK_APPROACH)
            stream(robot, ticks_p, gripper_pos)
            period = 1.0 / LOOP_HZ
            action = {f"{n}.pos": float(q_probe_pre[i]) for i, n in enumerate(BODY_MOTOR_ORDER)}
            action["gripper.pos"] = gripper_pos
            for _ in range(int(0.4 * LOOP_HZ)):
                robot.send_action(action)
                time.sleep(period)
            contact_q, info = descent_jacobian_until_contact(
                chain, robot, BODY_MOTOR_ORDER, q_probe_pre, tip_off, gripper_pos,
                descent_speed_mps=0.005,
                max_descent_m=0.08,
                xy_drift_abort_mm=2.0,
                load_thresh=60,
                stall_thresh_deg=1.5,
            )
            plane_z_probe = float(fk_tip(chain, contact_q, tip_off)[2])
            print(f"\n[cal] plane probe ended at z = {plane_z_probe*1000:.1f} mm  "
                  f"reason: {info['reason']}")
            # Only accept the probe value if real contact was detected (LOAD or
            # STALL). xy-drift abort means the probe slid before reaching the
            # surface -> probe is INVALID, fall back to world_settings origin.
            reason = info.get("reason", "")
            if reason.startswith("LOAD") or reason.startswith("STALL"):
                plane_z = plane_z_probe
                print(f"[cal] plane_z accepted: {plane_z*1000:.1f} mm")
            else:
                print(f"[cal] plane probe INVALID ({reason}); "
                      f"falling back to world_settings origin "
                      f"z={world['origin']['z']*1000:.1f} mm")
                plane_z = float(world["origin"]["z"])
            q_now = contact_q

        # ---------- Scan pose: save the FIRST captured pose with lowest reproj err ----------
        # Heuristic: use the pose closest to base (lowest x) with valid detection,
        # since that's the one our touch_marked_pixel scan-pose ladder selects.
        scan_pose_xyz = None
        scan_pose_q_deg = None
        if len(pose_T_base_gripper) > 0:
            xs = [T[0, 3] for T in pose_T_base_gripper]
            i_scan = int(np.argmin(xs))
            T_scan = pose_T_base_gripper[i_scan]
            scan_pose_xyz = T_scan[:3, 3].tolist()
            # Reconstruct q (degrees) from FK pose isn't trivial; just store xyz.
            print(f"\n[cal] saved scan pose: {pose_labels[i_scan]}  "
                  f"xyz = {(np.array(scan_pose_xyz) * 1000).round(1).tolist()} mm")

        # ---------- Save ----------
        config = RigCalibration(
            K=K.tolist(),
            dist=dist.flatten().tolist(),
            image_size=list(image_size),
            reproj_error_px=float(ret),
            T_gripper_camera=T_gripper_camera.tolist(),
            hand_eye_method=method_name,
            plane_z=plane_z,
            scan_pose_q_deg=scan_pose_q_deg,
            scan_pose_xyz=scan_pose_xyz,
            n_poses=n_valid,
            timestamp=ts,
            notes=f"square_size={square_size_m*1000:.1f} mm, "
                  f"board={args.squares_x}x{args.squares_y} squares",
        )
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(asdict(config), f, indent=2)
        print(f"\n[cal] SAVED -> {CONFIG_PATH}")

    finally:
        # Park: Jacobian descent at safe XY so pen rests on surface (no drift).
        try:
            print("\n[park] descending to surface for safe rest position...")
            park_xy = (0.18, 0.0)
            park_hover = np.array([park_xy[0], park_xy[1], plane_z + 0.05])
            obs = robot.get_observation()
            q_cur = np.array([float(obs[f"{n}.pos"]) for n in BODY_MOTOR_ORDER], dtype=float)
            q_p, _ = solve_ik_tip(chain, park_hover, q_cur, tip_off,
                                  enforce_gripper_down=ENFORCE_GRIPPER_DOWN)
            ticks = slew_segment(q_cur, q_p, MAX_STEP_DEG_PER_TICK_RETREAT)
            stream(robot, ticks, gripper_pos)
            descent_jacobian_until_contact(
                chain, robot, BODY_MOTOR_ORDER, q_p, tip_off, gripper_pos,
                descent_speed_mps=0.005, max_descent_m=0.07,
                xy_drift_abort_mm=2.0, load_thresh=60, stall_thresh_deg=1.5,
            )
            time.sleep(0.4)
        except Exception as e:
            print(f"[park] failed: {e}")
        try:
            robot.bus.enable_torque()
        except Exception:
            pass
        robot.disconnect()
        print("[robot] disconnected")


if __name__ == "__main__":
    main()
