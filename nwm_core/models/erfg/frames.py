from __future__ import annotations

import numpy as np
from .state import Frame

def identity_frame(frame_id: str) -> Frame:
    return Frame(frame_id=frame_id, rotation_mat=np.eye(3, dtype=np.float64), translation=np.zeros(3, dtype=np.float64))
