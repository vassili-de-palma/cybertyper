"""Eval-day entry point for Evals 2/3/bonus.

Scans once, then handles N rollouts (interactive, batch, or single sentence)
without re-scanning. See the README section on the legacy pipeline.

Common invocations:
    python -m cybertyping.tasks.session --interactive --preset FAST --multi-preprocess
    python -m cybertyping.tasks.session --type "hello world" --preset FAST
    python -m cybertyping.tasks.session --sentences-file inputs.txt --preset BLITZ
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keymap persistence
KEYMAP_CACHE_PATH = REPO_ROOT / "runs" / "last_keymap.json"
KEYMAP_SCENE_THUMB_PATH = REPO_ROOT / "runs" / "last_keymap_scene.png"
ADAPTIVE_STATE_PATH = REPO_ROOT / "runs" / "last_adaptive_state.json"
# Scene-similarity threshold for accepting a cached keymap. cv2.matchTemplate
# with TM_CCOEFF_NORMED returns ~0.99 for identical, ~0.85-0.95 for the same
# keyboard under different lighting, and <0.5 for a different keyboard.
SCENE_SIMILARITY_THRESHOLD = 0.85

from cybertyping.core.runtime import (
    banner, new_log, load_world, load_scan_pose, maybe_robot, is_dry_run,
    URDF_PATH,
)
from cybertyping.tasks._levenshtein import levenshtein
from cybertyping.tasks.fast_typer import (
    PreparedKeymap, prepare_keymap, plan_sentence_trajectory,
    type_sentence, estimate_sentence_time_s,
    PRESETS, PRESET_FAST, preset_from_env,
    serialize_adaptive_state, restore_adaptive_state,
)
from cybertyping.calibration.press_depth import (
    calibrate_press_depth, load_press_depth,
)
from cybertyping.perception.capture import open_camera
from cybertyping.perception.keymap.layout import load_layout
from cybertyping.perception.extrinsics.fov_approx import (
    PlaneModel, legacy_fov_pixel_to_robot,
)
from cybertyping.perception.extrinsics import calibrated as _cal



REHEARSAL_WORD = "abcdefg"   # touches a variety of rows/columns
REHEARSAL_PASS_THRESHOLD = 1.0   # all keys must register



@dataclass
class PhaseTimer:
    """Records per-phase wall-clock so we can see where time goes."""
    phases: dict[str, float] = field(default_factory=dict)
    _starts: dict[str, float] = field(default_factory=dict)
    t0: float = field(default_factory=time.perf_counter)

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def end(self, name: str) -> float:
        if name not in self._starts:
            return 0.0
        dt = time.perf_counter() - self._starts.pop(name)
        self.phases[name] = self.phases.get(name, 0.0) + dt
        return dt

    def total(self) -> float:
        return time.perf_counter() - self.t0

    def summary(self) -> str:
        lines = ["[timing] phase breakdown:"]
        for name, dt in sorted(self.phases.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:30s} {dt*1000:7.0f} ms")
        lines.append(f"  {'TOTAL':30s} {self.total()*1000:7.0f} ms")
        return "\n".join(lines)



@dataclass
class Session:
    robot: object
    chain: object
    prepared: PreparedKeymap
    keystroke_listener: object | None
    timer: PhaseTimer
    log: object
    home_q: np.ndarray | None = None



def _parallel_warmup(timer: PhaseTimer):
    """Spawn background threads for OCR backend init + keystroke listener
    start while the main thread is moving the arm. Returns a dict with
    the results once joined."""
    results: dict = {}

    def warm_ocr():
        from cybertyping.perception.detect.ocr_backends import get_default_ocr_backend
        timer.start("warmup_ocr")
        try:
            results["ocr"] = get_default_ocr_backend()
        except Exception as e:
            results["ocr_err"] = repr(e)
        timer.end("warmup_ocr")

    def warm_keystroke():
        from cybertyping.tasks._verify import KeystrokeListener
        timer.start("warmup_keystroke")
        ks = KeystrokeListener()
        ks.__enter__()
        results["keystroke"] = ks
        timer.end("warmup_keystroke")

    def warm_chain():
        timer.start("warmup_chain")
        from cybertyping import core
        chain = core.build_chain(URDF_PATH, locked_joints=core.LOCKED_JOINTS)
        results["chain"] = chain
        timer.end("warmup_chain")

    threads = [
        threading.Thread(target=warm_ocr,       daemon=True),
        threading.Thread(target=warm_keystroke, daemon=True),
        threading.Thread(target=warm_chain,     daemon=True),
    ]
    for t in threads: t.start()
    return threads, results


def _scene_thumb(frame_bgr, size: tuple[int, int] = (200, 150)):
    """Return a downscaled grayscale center-crop of a frame for scene
    similarity checks. The center crop (60% of the frame) ignores the
    workspace borders so a tiny shift in scan-pose landing doesn't blow up
    the match score."""
    import cv2
    h, w = frame_bgr.shape[:2]
    ch, cw = int(h * 0.6), int(w * 0.6)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = frame_bgr[y0:y0 + ch, x0:x0 + cw]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def _save_scene_thumb(frame_bgr) -> None:
    """Write the scene thumbnail next to the cached keymap so the next
    session can compare against it."""
    import cv2
    try:
        thumb = _scene_thumb(frame_bgr)
        KEYMAP_SCENE_THUMB_PATH.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(KEYMAP_SCENE_THUMB_PATH), thumb)
    except Exception as e:
        print(f"[cache] could not save scene thumb: {e!r}")


def _scene_similarity(frame_bgr) -> float | None:
    """Score how well `frame_bgr` matches the cached scene thumb.
    cv2.TM_CCOEFF_NORMED in [-1, 1]: 1.0 = identical, ~0.85-0.95 = same
    keyboard under different lighting, <0.5 = different keyboard.
    Returns None if no cached thumb exists."""
    import cv2
    if not KEYMAP_SCENE_THUMB_PATH.exists():
        return None
    cached = cv2.imread(str(KEYMAP_SCENE_THUMB_PATH), cv2.IMREAD_GRAYSCALE)
    if cached is None:
        return None
    current = _scene_thumb(frame_bgr, size=(cached.shape[1], cached.shape[0]))
    if current.shape != cached.shape:
        return None
    result = cv2.matchTemplate(current, cached, cv2.TM_CCOEFF_NORMED)
    return float(result[0, 0])


def _revalidate_cached_keymap(robot, chain, args, ocr, km,
                               *, similarity_threshold: float = SCENE_SIMILARITY_THRESHOLD,
                               ) -> bool:
    """Slew to scan pose, snap a frame, compare to the cached scene thumb.
    No OCR — that's the whole point: we skip the ~60s OCR cost when the
    keyboard hasn't moved.

    `ocr` is accepted for API compatibility with the previous OCR-based
    validator but isn't used.

    Returns True if scene similarity >= threshold. False means rescan.
    """
    from cybertyping import core
    import numpy as np

    scan = load_scan_pose()
    if scan is None or robot is None or is_dry_run():
        # No way to verify; trust the cache.
        return True
    cam_xyz = (float(scan["x"]), float(scan["y"]), float(scan["z"]))
    obs = robot.get_observation()
    q_now = np.array([float(obs[f"{n}.pos"]) for n in core.BODY_MOTOR_ORDER])
    q_scan = core.ik.solve_origin(chain, np.array(cam_xyz), q_now).q
    traj = core.slew_segment(q_now, q_scan, 1.6)
    core.stream(robot, traj, gripper_pos=float(obs["gripper.pos"]))
    time.sleep(0.3)

    with open_camera(mock_path=args.image, index=args.camera) as cam:
        frame = cam.grab()

    score = _scene_similarity(frame)
    if score is None:
        print("[cache] no scene thumb on disk; rescanning to capture one")
        return False
    ok = score >= similarity_threshold
    print(f"[cache] scene similarity = {score:.3f} "
          f"(threshold {similarity_threshold:.2f}) "
          f"-> {'accept' if ok else 'rescan'}")
    return ok


def _load_cached_keymap():
    """Return a KeyMap loaded from KEYMAP_CACHE_PATH, or None if it doesn't
    exist or can't be parsed. Freshness is no longer time-based: the caller
    runs `_revalidate_cached_keymap` (scene similarity vs the cached thumb)
    to decide whether the cache is still valid for the current keyboard."""
    if not KEYMAP_CACHE_PATH.exists():
        return None
    age = time.time() - KEYMAP_CACHE_PATH.stat().st_mtime
    try:
        from cybertyping.perception.keymap.homography import KeyMap, KeyEntry
        import numpy as np
        with open(KEYMAP_CACHE_PATH) as f:
            d = json.load(f)
        entries = {}
        for label, e in d["keys"].items():
            entries[label] = KeyEntry(
                label=label,
                image_xy=tuple(e["image_xy"]),
                canonical_xy=tuple(e["canonical_xy"]),
                robot_xy=tuple(e["robot_xy"]) if e["robot_xy"] is not None else None,
                robot_z=e["robot_z"],
                image_bbox=tuple(e["image_bbox"]) if e["image_bbox"] is not None else None,
            )
        km = KeyMap(
            entries=entries,
            homography_canon_to_image=np.array(d["homography_canon_to_image"]),
            anchors_used=d["anchors_used"],
            rmse_canonical_px=float(d["rmse_canonical_px"]),
            crossval=d.get("crossval", {}),
        )
        print(f"[cache] loaded keymap age={age:.0f}s, "
              f"{len(km.entries)} keys, RMSE={km.rmse_canonical_px:.2f} px")
        return km
    except Exception as e:
        print(f"[cache] could not load cached keymap: {e!r}")
        return None


def _scan_and_build_keymap(robot, chain, args, ocr, timer):
    """Capture frame(s), run OCR-primary, fit homography, return KeyMap."""
    from cybertyping import core
    from cybertyping.perception.detect.ocr_anchors import (
        detect_anchors_via_ocr, detect_anchors_via_ocr_multiframe,
    )
    from cybertyping.perception.keymap.homography import build_keymap_from_anchors

    layout = load_layout()
    world = load_world()
    # Prefer a flat plane at rig_calibration.json's plane_z when available
    # (the new calibration is the single source of truth). Fall back to
    # world_settings_load.json's {a,b,c} plane only if we don't have rig cal
    # and the world file actually carries a plane block.
    plane = None
    if world is not None and isinstance(world.get("plane"), dict):
        plane = PlaneModel.from_world_settings(world)

    scan = load_scan_pose()
    cam_xyz = (float(scan["x"]) if scan else 0.20,
               float(scan["y"]) if scan else 0.00,
               float(scan["z"]) if scan else 0.30)

    # Calibrated pixel->robot uses K, dist, T_gripper_camera + plane_z.
    # T_base_camera is read fresh after the slew so it matches the FK pose
    # the camera frame was actually captured at.
    cal = _cal.RigCalibration.try_load(_cal.DEFAULT_PATH)
    T_base_camera_at_scan: np.ndarray | None = None
    if cal is not None and plane is None:
        # Synthetic flat plane at rig_calibration's plane_z. The legacy-FOV
        # fallback path inside img_to_robot still needs a PlaneModel to
        # compute z_at(); a flat one is correct here because plane_z is the
        # rig's measured ground truth.
        plane = PlaneModel(a=0.0, b=0.0, c=cal.plane_z)

    # Move to scan pose
    timer.start("slew_to_scan")
    if robot is not None and scan is not None:
        obs = robot.get_observation()
        q_now = np.array([float(obs[f"{n}.pos"]) for n in core.BODY_MOTOR_ORDER])
        q_scan = core.ik.solve_origin(chain, np.array(cam_xyz), q_now).q
        traj = core.slew_segment(q_now, q_scan, 1.6)
        core.stream(robot, traj, gripper_pos=float(obs["gripper.pos"]))
        time.sleep(0.3)
        obs = robot.get_observation()
        q_at = np.array([float(obs[f"{n}.pos"]) for n in core.BODY_MOTOR_ORDER])
        cam_xyz = tuple(core.fk_tip(chain, q_at, core.TOOL_TIP_OFFSET_LOCAL_XYZ))
        if cal is not None:
            T_base_camera_at_scan = cal.T_base_camera(core.fk_T(chain, q_at))
    timer.end("slew_to_scan")

    # Capture
    timer.start("capture")
    with open_camera(mock_path=args.image, index=args.camera) as cam:
        if args.multi_frame and not args.image:
            # Real camera: take 3 frames with delays
            def _grab():
                return cam.grab()
            from cybertyping.perception.detect.ocr_anchors import (
                detect_anchors_via_ocr_multiframe,
            )
            timer.start("ocr_primary")
            anchors, frames = detect_anchors_via_ocr_multiframe(
                grab_frame=_grab, n_frames=3, inter_frame_delay_s=0.15,
                ocr=ocr,
                anchor_labels=layout.letters() + ["space", "enter"],
                min_confidence=0.3,
            )
            timer.end("ocr_primary")
            frame = frames[0]
        else:
            frame = cam.grab()
            timer.start("ocr_primary")
            anchors = detect_anchors_via_ocr(
                frame, ocr=ocr,
                anchor_labels=layout.letters() + ["space", "enter"],
                min_confidence=0.3,
                use_multi_preprocess=getattr(args, "multi_preprocess", False),
            )
            # Cascading recovery: if multi-preprocess underperformed, escalate
            # to the optimized (per-row strip) variant for an extra ~5-6s of
            # OCR time. Only triggers when we genuinely don't have enough
            # anchors, typical good scans skip this.
            ESCALATE_THRESHOLD = 12
            if (len(anchors) < ESCALATE_THRESHOLD
                and hasattr(ocr, "read_full_optimized")
                and not getattr(args, "no_escalate", False)):
                print(f"  [ocr] only {len(anchors)} anchors; escalating to "
                      f"per-row optimized OCR...")
                try:
                    anchors_opt = detect_anchors_via_ocr(
                        frame, ocr=ocr,
                        anchor_labels=layout.letters() + ["space", "enter"],
                        min_confidence=0.3,
                        use_optimized=True,
                    )
                    if len(anchors_opt) > len(anchors):
                        anchors = anchors_opt
                        print(f"  [ocr] escalation recovered {len(anchors)} anchors")
                except Exception as e:
                    print(f"  [ocr] escalation failed: {e!r}")
            timer.end("ocr_primary")
    timer.end("capture")

    h, w = frame.shape[:2]
    print(f"[scan] {w}x{h} -> {len(anchors)} OCR anchors")

    if len(anchors) < 4:
        print(f"WARNING: only {len(anchors)} anchors detected. Trying VLM fallback.")
        try:
            from cybertyping.perception.detect.vlm_anchors import get_default_backend as _vlm
            vlm = _vlm()
            anchors = vlm.detect_anchors(frame, layout.recommended_anchors())
            print(f"  VLM gave {len(anchors)} anchors")
        except Exception as e:
            print(f"  VLM fallback failed: {e!r}")
            if len(anchors) < 4:
                raise RuntimeError("Cannot build keymap: insufficient anchors.")

    # Pixel -> robot. Prefer the calibrated path (intrinsics + hand-eye +
    # bias map from rig_calibration.json) when available. Fall back to the
    # legacy FOV approx so dry-runs / un-calibrated setups still work.
    if cal is not None and T_base_camera_at_scan is not None:
        K_arr = cal.K
        dist_arr = cal.dist
        T_bc = T_base_camera_at_scan
        plane_z_cal = cal.plane_z
        bias_samples = cal.bias_samples
        pen_bias_scalar = cal.pen_bias_scalar
        # CYBERTYPING_NO_BIAS=1 disables the learned pen_bias_xy correction.
        # Useful for diagnosing whether the bias linear-extrapolation outside
        # the calibration sample region is what shifts the press targets.
        bias_enabled = (os.environ.get("CYBERTYPING_NO_BIAS", "").lower()
                        not in ("1", "true", "yes"))
        print(f"[scan] using calibrated pixel->robot "
              f"(reproj={cal.reproj_error_px:.2f} px, "
              f"bias_samples={len(bias_samples)}, "
              f"bias={'ON' if bias_enabled else 'OFF'})")

        def img_to_robot(px, py):
            hit = _cal.pixel_to_workspace_xy(
                px, py, K_arr, dist_arr, T_bc, plane_z_cal,
            )
            if hit is None:
                # Pathological ray, fall back to legacy approx so we don't
                # crash the keymap fit on a single bad anchor.
                return legacy_fov_pixel_to_robot(
                    px, py, w, h, cam_xyz, plane, 60.0,
                )
            if bias_enabled:
                bx, by = _cal.lookup_bias((hit[0], hit[1]), bias_samples,
                                         pen_bias_scalar)
            else:
                bx, by = 0.0, 0.0
            return float(hit[0] + bx), float(hit[1] + by), float(plane_z_cal)

        image_to_robot = img_to_robot
    elif plane is not None:
        print("[scan] using legacy FOV approx (no rig_calibration.json)")

        def img_to_robot(px, py):
            return legacy_fov_pixel_to_robot(px, py, w, h, cam_xyz, plane, 60.0)

        image_to_robot = img_to_robot
    else:
        image_to_robot = None

    timer.start("homography_fit")
    keymap = build_keymap_from_anchors(
        anchors, layout=layout,
        image_to_robot=image_to_robot,
    )
    timer.end("homography_fit")
    print(f"[scan] homography RMSE = {keymap.rmse_canonical_px:.2f} px  "
          f"({len(keymap.anchors_used)} anchors used)")

    # Save annotated keymap for debugging
    try:
        import cv2
        from cybertyping.perception.keymap.homography import annotate
        out_path = REPO_ROOT / "runs" / f"session_keymap_{int(time.time())}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotate(frame, keymap))
        print(f"[scan] annotated -> {out_path}")
    except Exception:
        pass

    # Persist the keymap + a scene thumbnail. Next session compares its
    # own scan-pose frame to this thumb to decide whether the keyboard
    # has moved enough to justify a fresh OCR pass.
    try:
        KEYMAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        keymap.save(KEYMAP_CACHE_PATH)
        _save_scene_thumb(frame)
        print(f"[cache] keymap saved -> {KEYMAP_CACHE_PATH}")
        print(f"[cache] scene thumb saved -> {KEYMAP_SCENE_THUMB_PATH}")
    except Exception as e:
        print(f"[cache] save failed: {e!r}")

    return keymap


def setup_session(args) -> Session:
    """Run all the per-session setup (scan, calibrate, pre-IK)."""
    timer = PhaseTimer()
    timer.start("total_setup")

    log = new_log("session")
    preset = (PRESETS[args.preset] if args.preset in PRESETS
              else preset_from_env(default="FAST"))
    print(f"[session] preset = {preset.name}")

    # Connect to the robot FIRST, before kicking off any heavy parallel
    # work. EasyOCR + torch/MPS loading in a sibling thread starves the
    # USB-serial timing the Feetech handshake relies on; we saw the bus
    # drop motor pings at random IDs (1, then 3, then full timeout) until
    # connect was isolated. touch_clicked_pixel.py works precisely because
    # it has no parallel warmup competing with connect.
    with maybe_robot() as robot:
        # Parallel warmup of OCR + KeystrokeListener + ikpy chain — only
        # after the bus is up.
        timer.start("parallel_warmup_spawn")
        warm_threads, warm_results = _parallel_warmup(timer)
        timer.end("parallel_warmup_spawn")

        # Wait for chain (needed for slew_to_scan)
        warm_threads[2].join()
        chain = warm_results.get("chain")
        if chain is None and not is_dry_run():
            raise RuntimeError("ikpy chain failed to build")

        # Scan + keymap build (or load from cache if requested + fresh)
        keymap = None
        if args.use_cached_keymap:
            keymap = _load_cached_keymap()
            if keymap is not None:
                # Verify cached anchors still line up before trusting it
                timer.start("cache_revalidate")
                ok = _revalidate_cached_keymap(
                    robot, chain, args, warm_results.get("ocr"), keymap,
                )
                timer.end("cache_revalidate")
                if not ok:
                    keymap = None   # fall through to fresh scan
        if keymap is None:
            keymap = _scan_and_build_keymap(robot, chain, args,
                                            warm_results.get("ocr"), timer)

        # Press-depth calibration.
        # DEFAULT IS OFF (--calibrate-press to enable). Reason: every second
        # of setup costs us a second of the eval budget.
        # Default depth is 6mm: with no keystroke feedback to drive +2mm
        # retries, the first press has to actuate the dome on its own.
        # Typical laptop key travel is 1.5-2mm; the rest is keyboard-body
        # deflection + tip compliance. 6mm is hardware-validated as a safe
        # over-press for the standardized US keyboard in the brief.
        press_offset_m = 0.006   # 6mm default if we skip cal
        # Prefer a cached value from a previous session (~free)
        cached = load_press_depth()
        if cached is not None:
            press_offset_m = cached
            print(f"[session] cached press_offset = {press_offset_m*1000:.2f} mm "
                  f"(skip --calibrate-press to use this)")

        if args.calibrate_press and not args.no_press_cal and robot is not None:
            timer.start("press_depth_cal")
            sample_labels = args.cal_keys.split(",")
            if args.fast_cal:
                sample_labels = sample_labels[:1]
            try:
                cal = calibrate_press_depth(
                    robot, chain, keymap,
                    sample_labels=sample_labels,
                    out_path="configs/press_depth.json",
                )
                press_offset_m = cal.offset_m
            except Exception as e:
                print(f"[session] press-depth calibration failed: {e!r}")
            timer.end("press_depth_cal")
        elif not cached:
            print(f"[session] no cache + no --calibrate-press; using default "
                  f"offset {press_offset_m*1000:.2f} mm. Smart-retry will "
                  f"learn per-key offsets if any keys miss.")

        # Pre-IK every key for this preset
        timer.start("pre_ik_all_keys")
        if robot is not None:
            obs = robot.get_observation()
            q_now = np.array([float(obs[f"{n}.pos"]) for n in
                              [n for n in __import__('cybertyping.core',
                                fromlist=['BODY_MOTOR_ORDER']).BODY_MOTOR_ORDER]])
        else:
            q_now = np.zeros(5)
        prepared = prepare_keymap(
            chain=chain, keymap=keymap, press_offset_m=press_offset_m,
            preset=preset, q_warm_start=q_now,
        )
        timer.end("pre_ik_all_keys")
        print(f"[session] pre-IK'd {len(prepared.keys)} keys "
              f"for preset={preset.name}")
        # Predict the worst-case IK residual to flag any unreachable keys
        worst = max((k.res_pre_m for k in prepared.keys.values()), default=0)
        print(f"[session]   worst IK residual = {worst*1000:.2f} mm")

        # Restore per-key adaptive state if available (from a recent run)
        if ADAPTIVE_STATE_PATH.exists() and args.use_cached_keymap:
            try:
                with open(ADAPTIVE_STATE_PATH) as f:
                    state = json.load(f)
                n = restore_adaptive_state(prepared, state)
                print(f"[session] restored adaptive state for {n} keys "
                      f"from {ADAPTIVE_STATE_PATH.name}")
            except Exception as e:
                print(f"[session] could not restore adaptive state: {e!r}")

        # Join remaining warmup threads
        for t in warm_threads: t.join()
        ocr_err = warm_results.get("ocr_err")
        if ocr_err: print(f"[session] (warmup) OCR error: {ocr_err}")

        timer.end("total_setup")
        print(timer.summary())
        log.add(event="setup_complete", **timer.phases,
                preset=preset.name, n_keys=len(prepared.keys),
                press_offset_mm=press_offset_m * 1000)

        return Session(
            robot=robot, chain=chain, prepared=prepared,
            keystroke_listener=warm_results.get("keystroke"),
            timer=timer, log=log,
            home_q=q_now.copy(),
        )



class MissingKeyError(RuntimeError):
    """Raised when an input character maps to a label not in the keymap.

    Typing through a missing key would silently drop characters and hand
    Eval 3 a guaranteed Levenshtein penalty per drop. We'd rather fail
    fast so the operator can rescan or rehearse.
    """


def _expand_sentence(sentence: str, prepared: PreparedKeymap,
                     *, strict: bool = True) -> list[str]:
    """Tokenize `sentence` into keymap labels.

    With `strict=True` (default), raises `MissingKeyError` if any required
    label is absent from the prepared keymap. With `strict=False`, missing
    labels are skipped with a warning (legacy behavior; only useful for
    rehearsal where dropping the missing key is preferable to crashing).
    """
    out = []
    missing: list[str] = []
    for ch in sentence.lower():
        if ch == " ": label = "space"
        elif ch == "\n": label = "enter"
        elif ch.isalpha(): label = ch
        else: continue
        if label in prepared.keys:
            out.append(label)
        else:
            missing.append(label)
    if missing:
        uniq = sorted(set(missing))
        msg = (f"required labels {uniq} not in prepared keymap "
               f"(have {len(prepared.keys)} keys); "
               f"rescan or rehearse before continuing.")
        if strict:
            raise MissingKeyError(msg)
        print(f"  [warn] {msg} (skipping)")
    return out


def type_one_sentence(session: Session, sentence: str,
                      *, labels: list[str] | None = None) -> dict:
    """Type one sentence. Returns a dict with the metrics.

    If `labels` is given, use it directly (skipping character tokenisation
    via `_expand_sentence`). Used by the --keys path for named-key
    sequences like 'space enter r l'.
    """
    if labels is None:
        labels = _expand_sentence(sentence, session.prepared)
    est_s = estimate_sentence_time_s(session.prepared, labels)
    print(f"\n[type] '{sentence}' -> {len(labels)} keys "
          f"(estimated {est_s:.1f}s)")
    t0 = time.perf_counter()

    if session.robot is None:
        # Dry-run
        traj = plan_sentence_trajectory(session.prepared, labels)
        dt = (len(traj) / 30.0)
        print(f"  [dry] planned {len(traj)} ticks "
              f"(~{dt:.2f}s @ 30Hz)")
        return {"sentence": sentence, "n_keys": len(labels),
                "estimated_s": est_s, "actual_s": dt, "typed": sentence,
                "levenshtein": 0, "score_5": 5, "dry_run": True}

    typed, verified = type_sentence(
        session.robot, session.chain, session.prepared, labels,
        keystroke_listener=session.keystroke_listener,
        log=session.log,
    )
    dt = time.perf_counter() - t0
    lev = levenshtein(typed, sentence)
    score = max(0, 5 - lev)
    n_ok = sum(1 for v in verified if v)
    print(f"  typed='{typed}' target='{sentence}'  lev={lev}  "
          f"score={score}/5  time={dt:.2f}s  verified={n_ok}/{len(verified)}")
    session.log.add(event="rollout", sentence=sentence, typed=typed,
                    levenshtein=lev, score=score, elapsed_s=dt,
                    n_verified=n_ok, n_keys=len(labels))
    return {"sentence": sentence, "typed": typed, "n_keys": len(labels),
            "estimated_s": est_s, "actual_s": dt,
            "levenshtein": lev, "score_5": score,
            "n_verified": n_ok, "dry_run": False}



def run_rehearsal(session: Session, word: str = REHEARSAL_WORD) -> float:
    """Type `word`, return per-press success rate."""
    print(f"\n[rehearse] '{word}'")
    res = type_one_sentence(session, word)
    if res.get("dry_run"):
        return 1.0
    n_keys = res["n_keys"]
    n_ok = res.get("n_verified", n_keys)
    return n_ok / max(1, n_keys)


def run_full_keyboard_rehearsal(session: Session) -> dict:
    """Press every one of the 28 keys (a-z + space + enter) once.

    Why: the eval gives us only ~100-150 presses total spread across 28
    keys (~4-5 presses each). Per-key online learning during the eval
    doesn't have enough data points to converge. So we front-load ALL the
    learning into a rehearsal that runs BEFORE the eval clock starts.

    After this routine:
      - Every key with a learned_press_offset_m != 0 has been calibrated
        to its actual press depth.
      - success_count + miss_count tells us per-key reliability.
      - Problematic keys (miss_count > 0 even after retry) are flagged
        for the operator.

    Returns a dict summarising per-key results.
    """
    import time as _t
    from cybertyping.tasks.fast_typer import (
        PreparedKeymap, type_sentence, update_key_after_press,
    )

    print("\n[full-rehearse] pressing all 28 keys to learn per-key calibration...")
    # Order keys in a typing-friendly path: row by row, left to right.
    # This minimizes lateral motion during rehearsal.
    layout_order = []
    for row_keys in (
        ["q","w","e","r","t","y","u","i","o","p"],
        ["a","s","d","f","g","h","j","k","l"],
        ["z","x","c","v","b","n","m"],
        ["space", "enter"],
    ):
        for k in row_keys:
            if k in session.prepared.keys:
                layout_order.append(k)

    t0 = _t.perf_counter()

    if session.robot is None:
        print("  [dry] would press: " + ",".join(layout_order))
        return {"n_keys": len(layout_order), "n_registered": len(layout_order),
                "miss_keys": [], "dry_run": True}

    # Type the full sequence with verify-and-retry enabled (the preset is
    # whatever the session is configured with; for ADAPTIVE this drives
    # learning, for FAST it still updates per-key state via our update calls).
    typed, verified = type_sentence(
        session.robot, session.chain, session.prepared, layout_order,
        keystroke_listener=session.keystroke_listener,
        log=session.log,
        verify_mode="stream_then_verify",
    )
    dt = _t.perf_counter() - t0

    n_ok = sum(1 for v in verified if v)
    miss_keys = [layout_order[i] for i, v in enumerate(verified) if not v]
    print(f"  [full-rehearse] {n_ok}/{len(layout_order)} keys registered "
          f"in {dt:.1f}s")
    # Show per-key learned offsets
    learned = [(k, session.prepared.keys[k].learned_press_offset_m * 1000)
               for k in layout_order
               if session.prepared.keys[k].learned_press_offset_m != 0.0]
    if learned:
        print("  [full-rehearse] learned per-key depth offsets:")
        for k, off_mm in learned:
            print(f"      {k:6s} +{off_mm:+.2f} mm")
    if miss_keys:
        print(f"  [full-rehearse] WARNING: keys that did NOT register even "
              f"after retry: {miss_keys}")
    return {"n_keys": len(layout_order), "n_registered": n_ok,
            "miss_keys": miss_keys, "elapsed_s": dt,
            "learned_offsets_mm": dict(learned)}



def run_interactive(session: Session, cap_planner=None) -> None:
    """Read sentences from stdin, type each one. Ctrl-D / 'q' to quit.

    If `cap_planner` is provided, each input line is treated as a natural-
    language instruction (e.g. 'type hello world', 'press every vowel') and
    expanded into a key sequence by the planner BEFORE typing.
    """
    print("\n[interactive] Type a sentence and press Enter. "
          "Empty line or 'q' to exit.")
    if cap_planner is not None:
        print("  (CaP planner active: input is natural-language instructions)")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in ("q", "quit", "exit"):
            break
        try:
            sentence_to_type = user_input
            if cap_planner is not None:
                from cybertyping.perception.keymap.layout import load_layout
                layout = load_layout()
                avail = layout.letters() + ["space", "enter"]
                result = cap_planner.plan(user_input, avail)
                # Convert label list back into a string for type_one_sentence
                # (which expects raw chars)
                sentence_to_type = "".join(
                    " " if l == "space" else ("\n" if l == "enter" else l)
                    for l in result.sequence
                )
                print(f"  [CaP] plan: {result.sequence}")
                if result.skipped_unsupported:
                    print(f"  [CaP] skipped: {result.skipped_unsupported}")
                if not sentence_to_type:
                    print("  [CaP] empty plan; skipping")
                    continue
            type_one_sentence(session, sentence_to_type)
        except Exception as e:
            print(f"  ERROR: {e!r}")
            session.log.add(event="rollout_error", err=repr(e))



def setup_session_background(args) -> tuple[threading.Thread, dict]:
    """Kick off setup_session in a background thread, return immediately.

    The caller can then read input (e.g. the first sentence) on the main
    thread; by the time the input arrives, setup is likely already done.
    When the caller needs the Session, it joins the thread and pulls
    `result['session']` out of the shared dict.

    Usage:
        thread, result = setup_session_background(args)
        first_sentence = input("> ")    # blocks; setup runs in parallel
        thread.join()
        if 'error' in result:
            raise RuntimeError(result['error'])
        session = result['session']
        type_one_sentence(session, first_sentence)
    """
    result: dict = {}
    def _worker():
        try:
            result['session'] = setup_session(args)
        except Exception as e:
            import traceback
            result['error'] = f"{type(e).__name__}: {e}"
            result['traceback'] = traceback.format_exc()
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t, result


def run_batch(session: Session, sentences: list[str]) -> list[dict]:
    results = []
    for i, s in enumerate(sentences):
        print(f"\n[batch] rollout {i+1}/{len(sentences)}")
        try:
            results.append(type_one_sentence(session, s))
        except Exception as e:
            print(f"  ERROR: {e!r}")
            results.append({"sentence": s, "error": repr(e)})
    # Summary
    print("\n[batch] summary:")
    total_t = sum(r.get("actual_s", 0) for r in results)
    total_score = sum(r.get("score_5", 0) for r in results)
    print(f"  {len(results)} sentences, total time {total_t:.1f}s, "
          f"total score {total_score}/{5*len(results)}")
    return results



def main(argv: list[str] | None = None) -> int:
    """Entry point. If `argv` is given, parse it instead of `sys.argv`.

    Eval-specific wrappers in `evals/` prepend their per-eval defaults
    and forward the user's CLI so the user can still override anything.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", type=str, default=None,
                    help="Single sentence to type. Mutually exclusive with "
                         "--interactive and --sentences-file.")
    ap.add_argument("--interactive", action="store_true",
                    help="Read sentences from stdin one per line.")
    ap.add_argument("--sentences-file", type=str, default=None,
                    help="Type each line of this file as one rollout.")
    ap.add_argument("--keys", type=str, default=None,
                    help="Space-separated key names to press in order "
                         "(e.g. 'space enter r l'). Bypasses --type's "
                         "character tokenisation; use this for named keys "
                         "like space/enter. Eval 1 path 2 uses this.")
    ap.add_argument("--preset", type=str, default="FAST",
                    choices=list(PRESETS.keys()),
                    help="Speed preset.")
    ap.add_argument("--rehearse", action="store_true",
                    help="Type a fixed test word (default 'abcdefg') before "
                         "the live rollouts. Faster but covers fewer keys.")
    ap.add_argument("--full-rehearse", action="store_true",
                    help="Press EVERY key (a-z + space + enter) once before "
                         "the live rollouts. Takes ~30-60s but populates "
                         "per-key calibration for every key the eval might "
                         "ask for. Strongly recommended for Eval 3 / bonus.")
    ap.add_argument("--rehearsal-word", type=str, default=REHEARSAL_WORD)
    # NB: by default we SKIP press-depth calibration to minimize setup time.
    # The bonus track and the per-rollout budget both count wall-clock from
    # script start, so every second of setup is a second of the budget.
    # Smart-retry (press 2mm deeper on miss + learn the new depth) is the
    # safety net that makes this trade-off work. If you want the safer (but
    # ~18s slower) calibration, pass --calibrate-press.
    ap.add_argument("--calibrate-press", action="store_true",
                    help="Run press-depth calibration at session start "
                         "(adds ~12-18s setup time, recommended for "
                         "OFFLINE practice / unconstrained sessions). "
                         "OFF by default to minimise setup time on eval day.")
    ap.add_argument("--no-press-cal", action="store_true",
                    help="(deprecated; press-depth cal is now off by default)")
    ap.add_argument("--cal-keys", type=str, default="f,j,space",
                    help="Sample keys for press-depth calibration "
                         "(if --calibrate-press is set).")
    ap.add_argument("--fast-cal", action="store_true",
                    help="If calibrating press depth, sample only ONE key "
                         "(~6s) instead of the default 3 (~18s). Use when "
                         "the keyboard is known to be flat/uniform.")
    ap.add_argument("--multi-frame", action="store_true",
                    help="Take 3 frames at scan time and union OCR detections.")
    ap.add_argument("--multi-preprocess", action="store_true",
                    help="Run OCR with 3 different preprocessing variants and "
                         "union detections. Costs ~1s extra, typically gets "
                         "us 15-30%% more anchors. Recommended default.")
    ap.add_argument("--no-escalate", action="store_true",
                    help="Disable automatic escalation to per-row optimized "
                         "OCR when initial scan finds < 12 anchors. By "
                         "default we escalate (adds ~5-6s only when needed).")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", type=str, default=None,
                    help="Use a saved JPEG instead of live camera "
                         "(dev/dry-run).")
    ap.add_argument("--use-cached-keymap", action="store_true",
                    help="Try to reuse runs/last_keymap.json by snapping a "
                         "frame at the scan pose and comparing it to the "
                         "cached scene thumbnail (cv2.matchTemplate); skip "
                         "the OCR pass if similarity is high. Cache lives "
                         "until the scene changes, not a time TTL.")
    ap.add_argument("--json-out", type=str, default=None,
                    help="Write per-rollout results to this JSON file "
                         "(machine-readable; for batch eval integration).")
    ap.add_argument("--no-overlap-setup", action="store_true",
                    help="In --interactive mode, do setup synchronously "
                         "(no background overlap with the first input wait). "
                         "Only useful for debugging.")
    ap.add_argument("--cap-mode", action="store_true",
                    help="Treat interactive input as NATURAL-LANGUAGE "
                         "instructions and pass them through the CaP "
                         "planner (Claude / Gemini / Mock) to expand into "
                         "a key sequence. Requires ANTHROPIC_API_KEY or "
                         "GEMINI_API_KEY (else uses the mock LLM).")
    ap.add_argument("--cap-mock", action="store_true",
                    help="Use the MockLLM for CaP planning (no API key "
                         "needed; only handles 'type X' instructions).")
    args = ap.parse_args(argv)

    if sum([bool(args.type), bool(args.interactive),
            bool(args.sentences_file), bool(args.keys)]) > 1:
        print("Pick one of --type / --interactive / --sentences-file / --keys.",
              file=sys.stderr)
        return 2
    if not (args.type or args.interactive or args.sentences_file or args.keys):
        print("Need --type, --interactive, --sentences-file, or --keys.",
              file=sys.stderr)
        return 2

    banner(f"Session start  (preset={args.preset})")

    # For --interactive mode, overlap setup with the wait-for-first-input
    # period: start scan/IK in a background thread, prompt the user, and
    # by the time they hit Enter the setup is (likely) already done.
    # For --type / --sentences-file modes, this overlap doesn't help so
    # we just do setup synchronously.
    if args.interactive and not getattr(args, "no_overlap_setup", False):
        print("[session] starting setup in background; type your first "
              "sentence (we'll wait for setup if needed):")
        setup_thread, setup_result = setup_session_background(args)
        try:
            first_sentence = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            first_sentence = ""
        if not first_sentence or first_sentence.lower() in ("q", "quit", "exit"):
            setup_thread.join(timeout=60)
            print("[session] exiting; setup may not have completed.")
            return 0
        # Wait for setup to finish (usually already done)
        print("[session] waiting for setup to finish...")
        setup_thread.join(timeout=180)
        if "error" in setup_result:
            print(f"[session] setup failed: {setup_result['error']}")
            print(setup_result.get('traceback', ''))
            return 2
        session = setup_result['session']
        try:
            type_one_sentence(session, first_sentence)
        except Exception as e:
            print(f"  ERROR: {e!r}")
            session.log.add(event="rollout_error", err=repr(e))
    else:
        session = setup_session(args)

    # Optional rehearsal(s)
    if args.full_rehearse:
        result = run_full_keyboard_rehearsal(session)
        rate = (result["n_registered"] / max(1, result["n_keys"]))
        if result.get("miss_keys"):
            print(f"[full-rehearse] {len(result['miss_keys'])} key(s) failed "
                  f"even after retry: {result['miss_keys']}")
            if args.preset == "BLITZ":
                print("[full-rehearse] downgrading BLITZ -> ADAPTIVE because "
                      "rehearsal had misses")
                args.preset = "ADAPTIVE"
    elif args.rehearse:
        rate = run_rehearsal(session, word=args.rehearsal_word)
        print(f"[rehearse] {rate*100:.0f}% of keys registered")
        if rate < REHEARSAL_PASS_THRESHOLD and args.preset == "BLITZ":
            print("[rehearse] BLITZ preset failed rehearsal; downgrading to FAST")
            args.preset = "FAST"

    # Optional CaP planner construction (used by interactive mode)
    cap_planner = None
    if args.cap_mode or args.cap_mock:
        from cybertyping.policy.cap_planner import CaPPlanner, MockLLM
        try:
            if args.cap_mock:
                cap_planner = CaPPlanner(llm=MockLLM(), mode="json")
            else:
                cap_planner = CaPPlanner()    # uses ANTHROPIC_API_KEY or GEMINI_API_KEY
            print(f"[cap] planner ready (mode={cap_planner._mode})")
        except Exception as e:
            print(f"[cap] could not initialise planner: {e!r}")
            print(f"[cap] falling back to literal-input mode")

    rollout_results = []
    used_overlap = (args.interactive and not getattr(args, "no_overlap_setup", False))
    if args.keys:
        labels = args.keys.split()
        rollout_results.append(
            type_one_sentence(session, " ".join(labels), labels=labels)
        )
    elif args.type:
        rollout_results.append(type_one_sentence(session, args.type))
    elif args.interactive:
        # When we used overlap-setup, the first sentence has already been
        # typed; continue with the interactive loop for the rest.
        if not used_overlap:
            run_interactive(session, cap_planner=cap_planner)
        else:
            run_interactive(session, cap_planner=cap_planner)
    elif args.sentences_file:
        with open(args.sentences_file) as f:
            sentences = [l.strip() for l in f if l.strip()]
        rollout_results = run_batch(session, sentences)

    # Persist the adaptive state for the NEXT session to pick up.
    try:
        state = serialize_adaptive_state(session.prepared)
        ADAPTIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ADAPTIVE_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[session] adaptive state -> {ADAPTIVE_STATE_PATH}")
    except Exception as e:
        print(f"[session] save-adaptive failed: {e!r}")

    print(session.timer.summary())
    out = REPO_ROOT / "runs" / f"session_{int(time.time())}.json"
    session.log.save(out)
    print(f"[done] log -> {out}")

    if args.json_out and rollout_results:
        summary = {
            "preset": args.preset,
            "n_rollouts": len(rollout_results),
            "total_time_s": sum(r.get("actual_s", 0) for r in rollout_results),
            "total_score_5": sum(r.get("score_5", 0) for r in rollout_results),
            "rollouts": rollout_results,
            "setup_phases_ms": {k: v * 1000 for k, v in session.timer.phases.items()},
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] json-out -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
