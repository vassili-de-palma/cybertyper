"""Canonical US keyboard layout loader.

The layout is stored in `configs/us_keyboard_layout.json` as unit-pitch
coordinates of every key CENTER. This module exposes typed access:

    layout = load_layout()
    qx, qy = layout.center("q")          # (2.0, 1.5) in pitch units
    pts    = layout.centers(["q","p","z","m","space"])   # Nx2 ndarray
    anchors = layout.recommended_anchors()

These coordinates are NOT in millimeters and NOT in pixels. They are the
canonical reference frame against which the homography is solved. The
absolute scale (mm per pitch) and the image-space orientation are
recovered by anchor detection (perception/homography.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

LAYOUT_PATH = Path(__file__).resolve().parents[3] / "configs" / "us_keyboard_layout.json"


@dataclass(frozen=True)
class Key:
    label: str
    x: float       # canonical pitch units (column center)
    y: float       # canonical pitch units (row center, +y = down)
    type: str      # "letter" | "digit" | "special"
    width: float = 1.0
    height: float = 1.0


class Layout:
    def __init__(self, raw: dict):
        self.raw = raw
        self.keys: dict[str, Key] = {}
        for label, info in raw["keys"].items():
            self.keys[label] = Key(
                label=label,
                x=float(info["x"]),
                y=float(info["y"]),
                type=str(info.get("type", "letter")),
                width=float(info.get("width", 1.0)),
                height=float(info.get("height", 1.0)),
            )

    # ---- canonical-frame access ----------------------------------------------

    def center(self, label: str) -> tuple[float, float]:
        """Center of `label` in canonical pitch units."""
        if label not in self.keys:
            raise KeyError(f"Key '{label}' not in canonical layout. "
                           f"Known: {sorted(self.keys.keys())}")
        k = self.keys[label]
        return (k.x, k.y)

    def centers(self, labels: Iterable[str]) -> np.ndarray:
        """Nx2 array of centers in canonical pitch units."""
        labels = list(labels)
        return np.array([self.center(l) for l in labels], dtype=float)

    def labels(self) -> list[str]:
        return list(self.keys.keys())

    def letters(self) -> list[str]:
        """The 26 a-z letter keys (typing target set)."""
        return [k.label for k in self.keys.values() if k.type == "letter"]

    def recommended_anchors(self) -> list[str]:
        return list(self.raw.get("recommended_anchors", ["q", "p", "z", "m", "space"]))

    # ---- bbox helpers (for visualisation) ------------------------------------

    def bbox_canonical(self, label: str) -> tuple[float, float, float, float]:
        """(x1, y1, x2, y2) of the key in canonical pitch units."""
        k = self.keys[label]
        return (k.x - k.width / 2.0, k.y - k.height / 2.0,
                k.x + k.width / 2.0, k.y + k.height / 2.0)


@lru_cache(maxsize=1)
def load_layout(path: str | Path = LAYOUT_PATH) -> Layout:
    with open(path, "r") as f:
        raw = json.load(f)
    return Layout(raw)


if __name__ == "__main__":
    L = load_layout()
    print(f"Loaded {len(L.keys)} keys from {LAYOUT_PATH.name}")
    print(f"Letters ({len(L.letters())}): {L.letters()}")
    print(f"Recommended anchors: {L.recommended_anchors()}")
    print(f"Q center (canonical): {L.center('q')}")
    print(f"P center (canonical): {L.center('p')}")
    print(f"Space center (canonical): {L.center('space')}")
