from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ...common.types import BBox, iou_xyxy


@dataclass(frozen=True)
class RelationConfig:
    contact_iou: float = 0.10
    inside_iou: float = 0.90
    support_y_gap: float = 12.0
    attach_persist_frames: int = 20
    attach_motion_eps: float = 6.0


def _contains(a: BBox, b: BBox, margin: float = 4.0) -> bool:
    return (b.x1 >= a.x1 + margin and b.y1 >= a.y1 + margin and b.x2 <= a.x2 - margin and b.y2 <= a.y2 - margin)


def infer_relations(
    boxes: Dict[str, BBox],
    velocities: Dict[str, np.ndarray],
    attach_counters: Dict[Tuple[str, str], int],
    cfg: RelationConfig,
) -> Tuple[List[Dict], Dict[Tuple[str, str], int]]:
    ids = list(boxes.keys())
    rels: List[Dict] = []

    # update attach persistence
    new_attach = dict(attach_counters)

    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j:
                continue
            a, b = ids[i], ids[j]
            ba, bb = boxes[a], boxes[b]
            iou = iou_xyxy(ba, bb)

            if iou >= cfg.contact_iou:
                rels.append({"src": a, "dst": b, "type": "contact", "confidence": float(min(1.0, iou * 2.0))})

            if _contains(ba, bb):
                rels.append({"src": b, "dst": a, "type": "inside", "confidence": 0.9})
                rels.append({"src": a, "dst": b, "type": "contains", "confidence": 0.9})

            x_overlap = max(0.0, min(ba.x2, bb.x2) - max(ba.x1, bb.x1))
            x_union = max(1.0, max(ba.x2, bb.x2) - min(ba.x1, bb.x1))
            x_overlap_frac = x_overlap / x_union

            if abs(bb.y2 - ba.y1) <= cfg.support_y_gap and x_overlap_frac >= 0.2:
                rels.append({"src": a, "dst": b, "type": "supporting", "confidence": float(min(1.0, 0.5 + x_overlap_frac))})
                rels.append({"src": b, "dst": a, "type": "supported_by", "confidence": float(min(1.0, 0.5 + x_overlap_frac))})

            # persistent contact and low relative motion
            key = (a, b)
            rel_motion = np.linalg.norm((velocities.get(a, np.zeros(2)) - velocities.get(b, np.zeros(2))))
            if iou >= cfg.contact_iou and rel_motion <= cfg.attach_motion_eps:
                new_attach[key] = int(new_attach.get(key, 0) + 1)
            else:
                new_attach[key] = max(0, int(new_attach.get(key, 0) - 1))

            if new_attach[key] >= cfg.attach_persist_frames:
                rels.append({"src": a, "dst": b, "type": "attached", "confidence": 0.85})

    return rels, new_attach
