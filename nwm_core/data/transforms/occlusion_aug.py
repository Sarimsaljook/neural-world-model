from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class OcclusionAugment:
    enable: bool = False
    max_boxes: int = 2
    max_area_frac: float = 0.2

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if not self.enable:
            return frames
        out = frames.copy()
        t, h, w, c = out.shape
        for _ in range(int(np.random.randint(0, self.max_boxes + 1))):
            area = self.max_area_frac * h * w * np.random.rand()
            bh = int(max(1, np.sqrt(area)))
            bw = int(max(1, area / max(1, bh)))
            y = int(np.random.randint(0, max(1, h - bh)))
            x = int(np.random.randint(0, max(1, w - bw)))
            out[:, y:y+bh, x:x+bw, :] = 0
        return out
