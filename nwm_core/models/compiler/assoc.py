from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class AssocMatch:
    track_index: int
    det_index: int
    score: float


def _bbox_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> float:
    x1 = float(max(a[0], b[0]))
    y1 = float(max(a[1], b[1]))
    x2 = float(min(a[2], b[2]))
    y2 = float(min(a[3], b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter + 1e-6
    return float(inter / denom)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sum(a * b).clamp(-1, 1))


def greedy_associate(
    track_embs: torch.Tensor,   # (T,D)
    track_boxes: torch.Tensor,  # (T,4) xyxy normalized
    det_embs: torch.Tensor,     # (N,D)
    det_boxes: torch.Tensor,    # (N,4) xyxy normalized
    min_score: float = 0.25,
    w_emb: float = 0.65,
    w_iou: float = 0.35,
) -> Tuple[Dict[int, int], List[int], List[int], List[AssocMatch]]:
    t = track_embs.shape[0]
    n = det_embs.shape[0]

    if t == 0:
        return {}, list(range(n)), [], []
    if n == 0:
        return {}, [], list(range(t)), []

    scores = torch.empty((t, n), device=track_embs.device, dtype=torch.float32)
    for i in range(t):
        for j in range(n):
            emb = _cos_sim(track_embs[i], det_embs[j])
            iou = _bbox_iou_xyxy(track_boxes[i], det_boxes[j])
            scores[i, j] = w_emb * emb + w_iou * iou

    matches: Dict[int, int] = {}
    used_det = set()
    used_trk = set()
    all_matches: List[AssocMatch] = []

    flat = []
    for i in range(t):
        for j in range(n):
            flat.append((float(scores[i, j]), i, j))
    flat.sort(reverse=True, key=lambda x: x[0])

    for sc, i, j in flat:
        if sc < min_score:
            break
        if i in used_trk or j in used_det:
            continue
        used_trk.add(i)
        used_det.add(j)
        matches[i] = j
        all_matches.append(AssocMatch(track_index=i, det_index=j, score=sc))

    unmatched_dets = [j for j in range(n) if j not in used_det]
    unmatched_trks = [i for i in range(t) if i not in used_trk]
    return matches, unmatched_dets, unmatched_trks, all_matches
