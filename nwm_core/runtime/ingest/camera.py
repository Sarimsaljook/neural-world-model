from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

class CameraStream:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def close(self) -> None:
        self.cap.release()
