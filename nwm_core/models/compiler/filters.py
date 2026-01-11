from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...common.types import BBox


def _bbox_center(b: BBox) -> np.ndarray:
    return np.array([(b.x1 + b.x2) * 0.5, (b.y1 + b.y2) * 0.5], dtype=np.float64)


def _bbox_wh(b: BBox) -> np.ndarray:
    return np.array([max(1.0, b.x2 - b.x1), max(1.0, b.y2 - b.y1)], dtype=np.float64)


@dataclass
class BBoxTrack:
    track_id: int
    bbox: BBox
    v_xy: np.ndarray
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    pos_var: np.ndarray = None  # (2,)
    vel_var: np.ndarray = None  # (2,)

    def __post_init__(self) -> None:
        if self.pos_var is None:
            self.pos_var = np.array([25.0, 25.0], dtype=np.float64)  # px^2
        if self.vel_var is None:
            self.vel_var = np.array([16.0, 16.0], dtype=np.float64)

    def predict(self, dt: float = 1.0) -> None:
        c = _bbox_center(self.bbox)
        c2 = c + self.v_xy * dt
        wh = _bbox_wh(self.bbox)
        self.bbox = BBox(c2[0] - wh[0] / 2, c2[1] - wh[1] / 2, c2[0] + wh[0] / 2, c2[1] + wh[1] / 2)

        self.pos_var = self.pos_var + self.vel_var * (dt * dt) + 10.0
        self.vel_var = self.vel_var + 2.0
        self.age += 1
        self.time_since_update += 1

    def update(self, meas: BBox, alpha: float = 0.75) -> None:
        c = _bbox_center(self.bbox)
        m = _bbox_center(meas)
        v_new = (m - c)
        self.v_xy = alpha * self.v_xy + (1.0 - alpha) * v_new

        self.bbox = meas
        self.hits += 1
        self.time_since_update = 0

        # shrink uncertainty after update
        self.pos_var = np.maximum(self.pos_var * 0.5, 4.0)
        self.vel_var = np.maximum(self.vel_var * 0.7, 2.0)

    def uncertainty(self) -> float:
        # scalar uncertainty for gating/probing
        return float(np.sqrt(self.pos_var.mean()) + 0.5 * np.sqrt(self.vel_var.mean()))
