"""Anchor-keys -> full keymap via planar homography.

The geometric prior we exploit:
- The keyboard is planar.
- The US layout has FIXED relative key positions (configs/us_keyboard_layout.json).
- We therefore only need to estimate the 8-DoF homography H_canon_to_image
  that maps unit-pitch canonical coords -> image-pixel coords.

Pipeline:

    1. VLM detects bounding boxes for a small set of anchor keys (e.g. q,p,z,m,space).
    2. We take each anchor's center in BOTH frames (canonical layout, image pixels).
    3. Solve H via DLT + LMEDS RANSAC (robust to one bad anchor).
    4. Reproject every key center from canonical -> image -> robot XY
       (via perception/extrinsics.py).
    5. Cross-validate: ask the VLM to detect a couple of held-out keys and
       check their reprojection error against canonical layout. This is the
       *quantitative* confidence signal you watch on demo day.

The result is a `KeyMap` containing image pixels, table coordinates, and
robot-frame coordinates for every key. It's the central perception output
that the eval scripts consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from .layout import Layout, load_layout
from cybertyping.perception.detect.vlm_anchors import bbox_center



@dataclass
class KeyEntry:
    label: str
    image_xy: tuple[float, float]          # pixel center
    canonical_xy: tuple[float, float]      # unit-pitch reference
    robot_xy: tuple[float, float] | None = None   # filled by extrinsics step
    robot_z: float | None = None
    image_bbox: tuple[int, int, int, int] | None = None  # if it was directly detected


@dataclass
class KeyMap:
    entries: dict[str, KeyEntry]
    homography_canon_to_image: np.ndarray
    anchors_used: list[str]
    rmse_canonical_px: float                # in-sample fit residual (image px)
    crossval: dict[str, float] = field(default_factory=dict)  # held-out reproj error per key (px)

    def __getitem__(self, label: str) -> KeyEntry:
        return self.entries[label]

    def __contains__(self, label: str) -> bool:
        return label in self.entries

    def labels(self) -> list[str]:
        return list(self.entries.keys())

    def to_json(self) -> dict:
        return {
            "homography_canon_to_image": self.homography_canon_to_image.tolist(),
            "anchors_used": self.anchors_used,
            "rmse_canonical_px": float(self.rmse_canonical_px),
            "crossval": {k: float(v) for k, v in self.crossval.items()},
            "keys": {
                label: {
                    "image_xy":     list(e.image_xy),
                    "canonical_xy": list(e.canonical_xy),
                    "robot_xy":     list(e.robot_xy) if e.robot_xy is not None else None,
                    "robot_z":      e.robot_z,
                    "image_bbox":   list(e.image_bbox) if e.image_bbox is not None else None,
                }
                for label, e in self.entries.items()
            },
        }

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)



def solve_homography(
    src_canonical_xy: np.ndarray,    # (N, 2) anchor centers in canonical units
    dst_image_xy:     np.ndarray,    # (N, 2) anchor centers in image pixels
    method: int = cv2.LMEDS,
    ransac_thresh_px: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (H_3x3, inlier_mask_bool_N).

    Uses cv2.findHomography with LMEDS by default. Falls back to a 4-point
    DLT solve if cv2 isn't happy (e.g. exactly 4 points).
    """
    src = np.asarray(src_canonical_xy, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(dst_image_xy,     dtype=np.float64).reshape(-1, 1, 2)
    if src.shape[0] < 4:
        raise ValueError(f"Need >= 4 anchor pairs, got {src.shape[0]}")
    H, mask = cv2.findHomography(src, dst, method=method,
                                 ransacReprojThreshold=ransac_thresh_px)
    if H is None:
        raise RuntimeError("findHomography failed -- check that anchors are not degenerate")
    mask = (mask.ravel() > 0) if mask is not None else np.ones(src.shape[0], dtype=bool)
    return H, mask


def reproject(H: np.ndarray, canonical_xy: np.ndarray) -> np.ndarray:
    """Apply 3x3 homography to (N,2) canonical points -> (N,2) image points."""
    pts = np.asarray(canonical_xy, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out


def fit_rmse(H: np.ndarray, src_canonical: np.ndarray, dst_image: np.ndarray) -> float:
    pred = reproject(H, src_canonical)
    return float(np.sqrt(np.mean(np.sum((pred - dst_image) ** 2, axis=1))))



def build_keymap_from_anchors(
    anchor_bboxes: dict[str, tuple[int, int, int, int]],
    layout: Layout | None = None,
    target_labels: Iterable[str] | None = None,
    image_to_robot: Callable[[float, float], tuple[float, float, float]] | None = None,
) -> KeyMap:
    """Given VLM-detected anchor bboxes, fit the homography and emit a full key map.

    Args:
        anchor_bboxes:  output of `VLMBackend.detect_anchors`.
        layout:         canonical layout (defaults to configs/us_keyboard_layout.json).
        target_labels:  which keys to populate. Default = all letters + space + enter.
        image_to_robot: optional callable (px_x, px_y) -> (robot_x, robot_y, robot_z).
                        If provided, KeyEntry.robot_xy / robot_z are filled in.

    Raises:
        ValueError if fewer than 4 valid anchors are usable.
    """
    layout = layout or load_layout()
    if target_labels is None:
        target_labels = layout.letters() + ["space", "enter"]
    target_labels = list(target_labels)

    # Collect anchor pairs in (canonical, image) space.
    canon_anchors: list[tuple[float, float]] = []
    image_anchors: list[tuple[float, float]] = []
    used_labels:   list[str] = []
    for label, bbox in anchor_bboxes.items():
        if label not in layout.keys:
            continue
        canon_anchors.append(layout.center(label))
        image_anchors.append(bbox_center(bbox))
        used_labels.append(label)

    if len(canon_anchors) < 4:
        raise ValueError(
            f"Need >= 4 valid anchors; got {len(canon_anchors)} "
            f"({used_labels}). Re-scan or expand the anchor set."
        )

    src = np.array(canon_anchors, dtype=np.float64)
    dst = np.array(image_anchors, dtype=np.float64)
    H, mask = solve_homography(src, dst)

    inlier_labels = [used_labels[i] for i, m in enumerate(mask) if m]
    rmse = fit_rmse(H, src[mask], dst[mask])

    # Reproject every target key.
    target_canon = layout.centers(target_labels)
    target_image = reproject(H, target_canon)

    entries: dict[str, KeyEntry] = {}
    for i, label in enumerate(target_labels):
        ix, iy = float(target_image[i, 0]), float(target_image[i, 1])
        cx, cy = float(target_canon[i, 0]), float(target_canon[i, 1])
        robot_xy: tuple[float, float] | None = None
        robot_z:  float | None = None
        if image_to_robot is not None:
            rx, ry, rz = image_to_robot(ix, iy)
            robot_xy = (rx, ry)
            robot_z  = rz
        entries[label] = KeyEntry(
            label=label,
            image_xy=(ix, iy),
            canonical_xy=(cx, cy),
            robot_xy=robot_xy,
            robot_z=robot_z,
            image_bbox=anchor_bboxes.get(label),
        )

    return KeyMap(
        entries=entries,
        homography_canon_to_image=H,
        anchors_used=inlier_labels,
        rmse_canonical_px=rmse,
        crossval={},
    )


def crossvalidate(
    keymap: KeyMap,
    held_out_bboxes: dict[str, tuple[int, int, int, int]],
) -> dict[str, float]:
    """Compute per-key pixel reprojection error for keys the homography did NOT see.

    Updates keymap.crossval in place and returns it. Use this to gate
    "should I rescan?", typical good fit on a webcam keyboard is < 6 px RMSE.
    """
    errors: dict[str, float] = {}
    for label, bbox in held_out_bboxes.items():
        if label not in keymap.entries:
            continue
        pred = np.array(keymap.entries[label].image_xy)
        obs  = np.array(bbox_center(bbox))
        errors[label] = float(np.linalg.norm(pred - obs))
    keymap.crossval = errors
    return errors



def annotate(image_bgr: np.ndarray, keymap: KeyMap,
             anchor_color=(0, 220, 0), other_color=(0, 200, 220)) -> np.ndarray:
    out = image_bgr.copy()
    for label, e in keymap.entries.items():
        cx, cy = int(round(e.image_xy[0])), int(round(e.image_xy[1]))
        is_anchor = label in keymap.anchors_used
        color = anchor_color if is_anchor else other_color
        cv2.drawMarker(out, (cx, cy), color,
                       markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        cv2.putText(out, label.upper(), (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    # Header
    txt = f"anchors={len(keymap.anchors_used)} rmse={keymap.rmse_canonical_px:.2f}px"
    if keymap.crossval:
        worst = max(keymap.crossval.values())
        txt += f"  xval_worst={worst:.2f}px"
    cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    # Self-test: synthesise an "image" of the canonical layout and check
    # we can recover the identity (up to scale & translation).
    layout = load_layout()
    anchors = layout.recommended_anchors()
    canon = layout.centers(anchors)

    # Pretend the camera scales canonical units by 60 px/unit, translates (200, 100),
    # and rotates by 3 degrees + slight perspective.
    s = 60.0
    theta = np.deg2rad(3.0)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    tx, ty = 200.0, 100.0
    image_pts = (canon @ R.T) * s + np.array([tx, ty])
    # Add 0.5 px of noise
    rng = np.random.default_rng(0)
    image_pts += rng.normal(0, 0.5, image_pts.shape)

    bboxes = {
        lbl: (int(image_pts[i, 0] - 10), int(image_pts[i, 1] - 10),
              int(image_pts[i, 0] + 10), int(image_pts[i, 1] + 10))
        for i, lbl in enumerate(anchors)
    }
    km = build_keymap_from_anchors(bboxes)
    print(f"Recovered RMSE: {km.rmse_canonical_px:.3f} px (should be ~0.5)")
    print(f"Anchors used:   {km.anchors_used}")
    print(f"Q image XY:     {km.entries['q'].image_xy}")
