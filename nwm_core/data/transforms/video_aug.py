from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

@dataclass
class VideoAugment:
    color_jitter: float = 0.0
    random_gray: float = 0.0

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        x = frames.astype(np.float32)
        if self.color_jitter > 0:
            scale = 1.0 + (np.random.rand() * 2 - 1) * self.color_jitter
            x = np.clip(x * scale, 0, 255)
        if self.random_gray > 0 and np.random.rand() < self.random_gray:
            g = x.mean(axis=-1, keepdims=True)
            x = np.repeat(g, 3, axis=-1)
        return x.astype(frames.dtype)
