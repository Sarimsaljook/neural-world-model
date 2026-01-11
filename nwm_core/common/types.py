from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Union

import numpy as np

NDArray = np.ndarray
NDArrayF32 = np.ndarray
NDArrayF64 = np.ndarray

TensorLike = Union["np.ndarray", Any]

@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def iou_xyxy(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area() + b.area() - inter
    return 0.0 if union <= 0.0 else float(inter / union)
