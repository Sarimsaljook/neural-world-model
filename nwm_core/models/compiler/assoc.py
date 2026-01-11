from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ...common.types import BBox, iou_xyxy


@dataclass(frozen=True)
class AssocConfig:
    min_iou: float = 0.15
    max_age: int = 15
    min_hits: int = 2


def greedy_iou_match(dets: List[BBox], trks: List[BBox], min_iou: float) -> List[Tuple[int, int, float]]:
    if not dets or not trks:
        return []
    ious = np.zeros((len(dets), len(trks)), dtype=np.float64)
    for i, d in enumerate(dets):
        for j, t in enumerate(trks):
            ious[i, j] = iou_xyxy(d, t)

    pairs: List[Tuple[int, int, float]] = []
    used_d, used_t = set(), set()
    while True:
        idx = np.unravel_index(np.argmax(ious), ious.shape)
        i, j = int(idx[0]), int(idx[1])
        best = float(ious[i, j])
        if best < min_iou:
            break
        if i in used_d or j in used_t:
            ious[i, j] = -1.0
            continue
        pairs.append((i, j, best))
        used_d.add(i)
        used_t.add(j)
        ious[i, :] = -1.0
        ious[:, j] = -1.0
    return pairs
