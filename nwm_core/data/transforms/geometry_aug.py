from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class GeometryAugment:
    hflip_p: float = 0.0

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if self.hflip_p > 0 and np.random.rand() < self.hflip_p:
            return frames[:, :, ::-1, :]
        return frames
