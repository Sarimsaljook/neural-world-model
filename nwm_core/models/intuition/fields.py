from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ...common.types import BBox


@dataclass(frozen=True)
class IntuitionConfig:
    tall_ratio: float = 1.4
    speed_ref: float = 12.0
    slip_speed_ref: float = 10.0
    eps: float = 1e-6


def _wh(b: BBox) -> Tuple[float, float]:
    w = max(1.0, b.x2 - b.x1)
    h = max(1.0, b.y2 - b.y1)
    return w, h


def compute_intuition_fields(
    boxes: Dict[str, BBox],
    velocities: Dict[str, np.ndarray],
    relations: List[Dict],
    cfg: IntuitionConfig,
) -> Dict[str, Dict[str, float]]:
    rel_map = {(r["src"], r["dst"], r["type"]) for r in relations}
    out: Dict[str, Dict[str, float]] = {}

    for eid, b in boxes.items():
        w, h = _wh(b)
        v = velocities.get(eid, np.zeros(2, dtype=np.float64))
        speed = float(np.linalg.norm(v))
        aspect = float(h / max(cfg.eps, w))

        # stability for tall objects and speed
        tallness = max(0.0, (aspect - cfg.tall_ratio) / max(cfg.eps, cfg.tall_ratio))
        motion = min(1.0, speed / max(cfg.eps, cfg.speed_ref))
        stability_risk = float(np.clip(0.25 * tallness + 0.75 * motion, 0.0, 1.0))

        # lateral speed when supported_by something
        supported = any((eid, other, "supported_by") in rel_map for other in boxes.keys() if other != eid)
        lateral = abs(float(v[0]))
        slip = min(1.0, lateral / max(cfg.eps, cfg.slip_speed_ref))
        slip_risk = float(np.clip(slip if supported else 0.25 * slip, 0.0, 1.0))

        out[eid] = {
            "stability_risk": stability_risk,
            "slip_risk": slip_risk,
            "speed_px": speed,
            "aspect_hw": aspect,
        }

    return out
