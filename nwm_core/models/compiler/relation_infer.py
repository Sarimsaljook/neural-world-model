from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import torch

def cxcywh_to_xyxy(boxes_cxcywh: torch.Tensor) -> torch.Tensor:

    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    out = torch.stack([x1, y1, x2, y2], dim=-1)
    return out.clamp(0.0, 1.0)

def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 0.0)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 0.0)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)

def infer_pair_relations(
    a_box: torch.Tensor,
    b_box: torch.Tensor,
    a_mask: torch.Tensor,
    b_mask: torch.Tensor,
    depth: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    rel: Dict[str, float] = {}

    a_xyxy = cxcywh_to_xyxy(a_box)
    b_xyxy = cxcywh_to_xyxy(b_box)

    iou = _iou_xyxy(a_xyxy, b_xyxy)
    if iou > 0.01:
        rel["contact"] = min(1.0, iou * 5.0)

    a_bin = a_mask > 0.5
    b_bin = b_mask > 0.5

    inter = (a_bin & b_bin).float().sum()
    if inter > 0:
        frac_a = inter / (a_bin.float().sum() + 1e-6)
        frac_b = inter / (b_bin.float().sum() + 1e-6)

        if depth is not None:
            da = depth[a_bin].median()
            db = depth[b_bin].median()
            if da < db:
                frac_b *= 0.2

        if frac_a > 0.6:
            rel["inside"] = float(frac_a)
        if frac_b > 0.6:
            rel["contains"] = float(frac_b)

    return rel

@dataclass
class RelationInferConfig:
    contact_on: float = 0.65
    contact_off: float = 0.45
    contact_on_frames: int = 2
    contact_off_frames: int = 4

    inside_on: float = 0.70
    inside_off: float = 0.50
    inside_on_frames: int = 3
    inside_off_frames: int = 5

    min_overlap_iou: float = 0.08
    min_inside_frac: float = 0.65

    depth_margin: float = 0.02
    depth_min_valid: float = 1e-6


@dataclass
class RelEvent:
    kind: str
    src: int
    dst: int
    p: float


class _EdgeState:
    __slots__ = ("active", "on_count", "off_count")

    def __init__(self):
        self.active = False
        self.on_count = 0
        self.off_count = 0


class RelationEventStabilizer:
    def __init__(self, cfg: RelationInferConfig):
        self.cfg = cfg
        self.state: Dict[Tuple[str, int, int], _EdgeState] = {}

    def _step(
        self,
        key: Tuple[str, int, int],
        p: float,
        on_th: float,
        off_th: float,
        on_frames: int,
        off_frames: int,
        begin: str,
        end: str,
    ) -> List[RelEvent]:
        st = self.state.setdefault(key, _EdgeState())
        evs: List[RelEvent] = []

        if not st.active:
            if p >= on_th:
                st.on_count += 1
                if st.on_count >= on_frames:
                    st.active = True
                    st.on_count = 0
                    st.off_count = 0
                    _, s, d = key
                    evs.append(RelEvent(begin, s, d, float(p)))
            else:
                st.on_count = 0
        else:
            if p <= off_th:
                st.off_count += 1
                if st.off_count >= off_frames:
                    st.active = False
                    st.on_count = 0
                    st.off_count = 0
                    _, s, d = key
                    evs.append(RelEvent(end, s, d, float(p)))
            else:
                st.off_count = 0

        return evs

    def update(self, scores: Dict[Tuple[str, int, int], float]) -> List[RelEvent]:
        cfg = self.cfg
        evs: List[RelEvent] = []

        for (k, s, d), p in scores.items():
            if k == "contact":
                evs += self._step(
                    (k, s, d), p,
                    cfg.contact_on, cfg.contact_off,
                    cfg.contact_on_frames, cfg.contact_off_frames,
                    "contact_begin", "contact_end",
                )
            elif k == "inside":
                evs += self._step(
                    (k, s, d), p,
                    cfg.inside_on, cfg.inside_off,
                    cfg.inside_on_frames, cfg.inside_off_frames,
                    "inside_begin", "inside_end",
                )
            elif k == "contains":
                evs += self._step(
                    (k, s, d), p,
                    cfg.inside_on, cfg.inside_off,
                    cfg.inside_on_frames, cfg.inside_off_frames,
                    "contains_begin", "contains_end",
                )

        return evs


def _mask_iou_bin(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter = (a & b).sum().float()
    union = (a | b).sum().float().clamp(min=1.0)
    return inter / union


def _median_depth(depth: torch.Tensor, mask_bin: torch.Tensor, min_valid: float) -> float:
    if not torch.any(mask_bin):
        return 0.0
    d = depth[mask_bin]
    d = d[d > min_valid]
    if d.numel() == 0:
        return 0.0
    return float(d.median().item())


class RelationInfer:
    def __init__(self, cfg: Optional[RelationInferConfig] = None):
        self.cfg = cfg or RelationInferConfig()
        self.stabilizer = RelationEventStabilizer(self.cfg)

    @torch.no_grad()
    def infer(
        self,
        masks: torch.Tensor,        # (N,H,W) prob in [0,1]
        boxes_xyxy: torch.Tensor,   # (N,4) norm [0,1], only used for fast rejection if you want later
        depth: torch.Tensor,        # (H,W)
    ) -> Tuple[Dict[Tuple[str, int, int], float], List[RelEvent]]:

        cfg = self.cfg
        N, H, W = masks.shape
        binm = masks > 0.5
        area = binm.flatten(1).sum(dim=1).float().clamp(min=1.0)

        med_depth = torch.tensor(
            [_median_depth(depth, binm[i], cfg.depth_min_valid) for i in range(N)],
            device=masks.device,
            dtype=torch.float32,
        )

        scores: Dict[Tuple[str, int, int], float] = {}

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                inter = (binm[i] & binm[j]).sum().float()
                if inter <= 0:
                    continue

                iou = _mask_iou_bin(binm[i], binm[j])
                if iou < cfg.min_overlap_iou:
                    continue

                p_contact = float(torch.clamp(iou * 2.0, 0.0, 1.0).item())
                scores[("contact", i, j)] = p_contact

                frac_inside = inter / area[j]
                if float(frac_inside.item()) >= cfg.min_inside_frac:
                    dj = float(med_depth[j].item())
                    di = float(med_depth[i].item())
                    if dj > di + cfg.depth_margin:
                        p_inside = float(torch.clamp(frac_inside, 0.0, 1.0).item())
                        scores[("inside", j, i)] = p_inside
                        scores[("contains", i, j)] = p_inside

        events = self.stabilizer.update(scores)
        return scores, events
