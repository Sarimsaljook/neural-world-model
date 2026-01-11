from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

def load_intrinsics(path: Path) -> CameraIntrinsics:
    d = json.loads(path.read_text(encoding="utf-8"))
    return CameraIntrinsics(
        fx=float(d["fx"]), fy=float(d["fy"]),
        cx=float(d["cx"]), cy=float(d["cy"]),
        width=int(d["width"]), height=int(d["height"])
    )
