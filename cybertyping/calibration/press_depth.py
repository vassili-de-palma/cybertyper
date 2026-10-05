"""Measure the per-session press depth offset.

Contact-descend on a few sample keys, record where the tip actually
makes contact relative to the plane equation, average. Result is a
single offset_m used by fast_typer for open-loop presses.

    offset_m = calibrate_press_depth(robot, chain, keymap,
                                     sample_labels=["f", "j", "space"])
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# Default sample keys: spread across all 3 letter rows + space, with both
# corners (Q, P) and centers (F, J, V) included. Rationale: some keyboards
# (especially curved laptop ones) have a small Z variation between rows or
# between center and edge keys. Sampling 7 keys catches this in one
# calibration pass and lets us fit a low-order plane correction if needed.
#
# If you're short on time, fall back to ("f", "j", "space"), 3 samples.
DEFAULT_SAMPLE_LABELS = (
    "q",     # QWERTY row, far left
    "p",     # QWERTY row, far right
    "f",     # ASDF row, home-row bump (haptic landmark)
    "j",     # ASDF row, home-row bump
    "v",     # ZXCV row, center
    "b",     # ZXCV row, right of center
    "space", # bottom row, large + central
)
FAST_SAMPLE_LABELS = ("f", "j", "space")   # 3-key minimum for speed


@dataclass
class PressDepthResult:
    offset_m: float                  # the constant we use for open-loop presses
    per_key: dict[str, float]        # measured offset per sampled key
    rms_residual_mm: float           # spread of per-key measurements
    n_keys_used: int

    def to_json(self) -> dict:
        return {
            "offset_m": float(self.offset_m),
            "per_key_offset_m": {k: float(v) for k, v in self.per_key.items()},
            "rms_residual_mm": float(self.rms_residual_mm),
            "n_keys_used": int(self.n_keys_used),
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)


def calibrate_press_depth(
    robot, chain, keymap,
    sample_labels: Iterable[str] = DEFAULT_SAMPLE_LABELS,
    approach_height_m: float = 0.05,
    out_path: str | Path | None = "configs/press_depth.json",
) -> PressDepthResult:
    """Touch each sample key with contact-sense, record actual press Z,
    return the per-key offset relative to the homography-predicted plane Z.

    Args:
        robot:               connected SO101Follower.
        chain:               built ikpy chain.
        keymap:              perception.homography.KeyMap with robot_xy + robot_z.
        sample_labels:       which keys to touch. >=2 recommended.
        approach_height_m:   m above plane for pre-touch.
        out_path:            JSON output; pass None to skip saving.

    Returns:
        PressDepthResult.

    Notes:
        - keymap entries MUST have robot_xy and robot_z filled in (i.e. the
          keymap was built with image_to_robot != None at scan time).
        - This routine takes ~6s per sample key, so 3 samples = ~18s of
          session startup. Worth it: every subsequent press saves ~5s.
    """
    from cybertyping import core   # lazy: needs ikpy
    from cybertyping.press.descend import load_contact as tsl  # safe fallback wrapper

    labels = [l for l in sample_labels if l in keymap.entries
              and keymap.entries[l].robot_xy is not None]
    if len(labels) < 2:
        raise RuntimeError(
            f"Need >= 2 valid sample keys; got {labels}. "
            f"Either keymap is missing robot_xy fields or all sample_labels "
            f"are absent from the keymap."
        )

    per_key: dict[str, float] = {}
    for label in labels:
        entry = keymap.entries[label]
        rx, ry = entry.robot_xy
        plane_z = float(entry.robot_z)
        # 1. slew to pre-touch
        pre = np.array([rx, ry, plane_z + approach_height_m])
        obs   = robot.get_observation()
        q_now = np.array([float(obs[f"{n}.pos"]) for n in core.BODY_MOTOR_ORDER])
        q_pre = core.ik.solve_tip(chain, pre, q_now, core.TOOL_TIP_OFFSET_LOCAL_XYZ).q
        traj  = core.slew_segment(q_now, q_pre, 1.6)
        core.stream(robot, traj, gripper_pos=float(obs["gripper.pos"]))

        # 2. contact descent
        tsl.press(
            chain=chain, robot=robot,
            pre_tip_xyz=pre, q_pre=q_pre,
            tip_off=core.TOOL_TIP_OFFSET_LOCAL_XYZ,
            gripper_pos=float(obs["gripper.pos"]),
        )

        # 3. read actual tip Z at contact
        obs    = robot.get_observation()
        q_at   = np.array([float(obs[f"{n}.pos"]) for n in core.BODY_MOTOR_ORDER])
        tip_xyz = core.fk_tip(chain, q_at, core.TOOL_TIP_OFFSET_LOCAL_XYZ)
        offset = float(tip_xyz[2] - plane_z)
        per_key[label] = offset
        print(f"  [press-depth] {label}: tip_z={tip_xyz[2]:+.4f} m  "
              f"plane_z={plane_z:+.4f} m  offset={offset*1000:+.2f} mm")

        # 4. retreat back to pre-touch
        obs   = robot.get_observation()
        q_now = np.array([float(obs[f"{n}.pos"]) for n in core.BODY_MOTOR_ORDER])
        q_back = core.ik.solve_tip(chain, pre, q_now, core.TOOL_TIP_OFFSET_LOCAL_XYZ).q
        traj  = core.slew_segment(q_now, q_back, 1.6)
        core.stream(robot, traj, gripper_pos=float(obs["gripper.pos"]))

    offsets = np.array(list(per_key.values()))
    mean_offset = float(np.mean(offsets))
    rms_residual = float(np.sqrt(np.mean((offsets - mean_offset) ** 2)) * 1000.0)

    result = PressDepthResult(
        offset_m=mean_offset, per_key=per_key,
        rms_residual_mm=rms_residual, n_keys_used=len(labels),
    )

    if out_path:
        result.save(out_path)
        print(f"  [press-depth] saved -> {out_path}")
    print(f"  [press-depth] OFFSET = {mean_offset*1000:+.2f} mm "
          f"(residual RMS {rms_residual:.2f} mm over {len(labels)} keys)")
    return result


def load_press_depth(path: str | Path = "configs/press_depth.json"
                     ) -> float | None:
    """Load the per-session press offset from disk. Returns None if absent."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return float(data["offset_m"])



def _self_test():
    """Verify the math: given noisy per-key measurements of a constant
    offset, the mean+RMS should recover the truth."""
    rng = np.random.default_rng(0)
    true_offset = 0.003   # 3 mm
    samples = [true_offset + rng.normal(0, 0.0005) for _ in range(5)]
    arr = np.array(samples)
    mean = float(np.mean(arr))
    rms_mm = float(np.sqrt(np.mean((arr - mean) ** 2)) * 1000.0)
    assert abs(mean - true_offset) < 0.001
    assert rms_mm < 1.5
    print(f"  recovered offset = {mean*1000:+.2f} mm "
          f"(truth {true_offset*1000:.2f} mm), residual RMS = {rms_mm:.2f} mm")


if __name__ == "__main__":
    _self_test()
