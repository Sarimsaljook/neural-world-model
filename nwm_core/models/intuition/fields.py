from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .heads import IntuitionHeads, IntuitionHeadsConfig


@dataclass(frozen=True)
class IntuitionConfig:
    mask_thresh: float = 0.5
    min_area_frac: float = 0.003

    collision_iou_thresh: float = 0.02
    collision_topk: int = 64

    depth_valid_min: float = 1e-4

    heuristic_weight: float = 1.0
    learned_weight: float = 0.0

    feature_dim: int = 256
    learned_heads: bool = False


def _pairwise_iou_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    A = boxes.shape[0]

    xx1 = torch.maximum(x1[:, None], x1[None, :])
    yy1 = torch.maximum(y1[:, None], y1[None, :])
    xx2 = torch.minimum(x2[:, None], x2[None, :])
    yy2 = torch.minimum(y2[:, None], y2[None, :])

    iw = (xx2 - xx1).clamp(min=0.0)
    ih = (yy2 - yy1).clamp(min=0.0)
    inter = iw * ih

    area = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    union = area[:, None] + area[None, :] - inter
    return inter / (union + 1e-6)


def _mask_median_depth(depth: torch.Tensor, mask: torch.Tensor, valid_min: float) -> Tuple[torch.Tensor, torch.Tensor]:
    m = mask > 0.5
    if not torch.any(m):
        return depth.new_tensor(0.0), depth.new_tensor(0.0)
    d = depth[m]
    d = d[d > valid_min]
    if d.numel() == 0:
        return depth.new_tensor(0.0), depth.new_tensor(0.0)
    return d.median(), d.var(unbiased=False)


def _mask_edge_fraction(mask: torch.Tensor) -> torch.Tensor:
    m = (mask > 0.5).float()
    if m.sum() <= 1.0:
        return m.new_tensor(0.0)
    gy = (m[1:, :] - m[:-1, :]).abs()
    gx = (m[:, 1:] - m[:, :-1]).abs()
    edge = gy.sum() + gx.sum()
    return (edge / (m.sum() + 1e-6)).clamp(0.0, 10.0)


def _build_entity_features(det: Dict[str, torch.Tensor], cfg: IntuitionConfig) -> torch.Tensor:
    boxes = det["boxes"]
    masks = det["masks"]
    conf = det["conf"]
    depth = det["depth"]

    N = boxes.shape[0]
    if N == 0:
        return boxes.new_zeros((0, cfg.feature_dim))

    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1).clamp(min=0.0)
    bh = (y2 - y1).clamp(min=0.0)
    area = (bw * bh).clamp(0.0, 1.0)

    d_med = []
    d_var = []
    e_frac = []
    for i in range(N):
        dm, dv = _mask_median_depth(depth, masks[i], cfg.depth_valid_min)
        d_med.append(dm)
        d_var.append(dv)
        e_frac.append(_mask_edge_fraction(masks[i]))

    d_med = torch.stack(d_med, dim=0)
    d_var = torch.stack(d_var, dim=0)
    e_frac = torch.stack(e_frac, dim=0)

    base = torch.stack([conf, cx, cy, bw, bh, area, d_med, d_var, e_frac], dim=-1)  # (N,9)
    base = torch.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)

    Fdim = cfg.feature_dim
    if base.shape[-1] >= Fdim:
        return base[:, :Fdim]
    pad = base.new_zeros((N, Fdim - base.shape[-1]))
    return torch.cat([base, pad], dim=-1)


def _heuristic_fields(det: Dict[str, torch.Tensor], cfg: IntuitionConfig) -> Dict[str, torch.Tensor]:
    boxes = det["boxes"]
    masks = det["masks"]
    conf = det["conf"]
    depth = det["depth"]
    H = int(det["H"].item()) if torch.is_tensor(det["H"]) else int(det["H"])
    W = int(det["W"].item()) if torch.is_tensor(det["W"]) else int(det["W"])

    N = boxes.shape[0]
    if N == 0:
        return {
            "stability": boxes.new_zeros((0,)),
            "slip": boxes.new_zeros((0,)),
            "support": boxes.new_zeros((0,)),
            "collision_pairs": boxes.new_zeros((0, 2), dtype=torch.long),
            "collision": boxes.new_zeros((0,)),
        }

    x1, y1, x2, y2 = boxes.unbind(-1)
    bw = (x2 - x1).clamp(min=0.0)
    bh = (y2 - y1).clamp(min=0.0)
    area = (bw * bh).clamp(0.0, 1.0)

    d_med = []
    d_var = []
    for i in range(N):
        dm, dv = _mask_median_depth(depth, masks[i], cfg.depth_valid_min)
        d_med.append(dm)
        d_var.append(dv)
    d_med = torch.stack(d_med, dim=0)
    d_var = torch.stack(d_var, dim=0)

    tall = (bh / (bw + 1e-6)).clamp(0.0, 10.0)
    thinness = torch.sigmoid((tall - 1.2) * 2.0)
    unstable = (0.35 * thinness + 0.35 * torch.sigmoid(d_var * 6.0) + 0.30 * (1.0 - conf)).clamp(0.0, 1.0)

    slip = (0.45 * torch.sigmoid(d_var * 8.0) + 0.25 * (1.0 - conf) + 0.30 * torch.sigmoid((area - 0.05) * 3.0)).clamp(
        0.0, 1.0
    )

    support = (1.0 - unstable).clamp(0.0, 1.0)

    iou = _pairwise_iou_xyxy(boxes)
    iou.fill_diagonal_(0.0)

    pairs = torch.nonzero(iou > cfg.collision_iou_thresh)
    if pairs.numel() == 0:
        return {
            "stability": unstable,
            "slip": slip,
            "support": support,
            "collision_pairs": boxes.new_zeros((0, 2), dtype=torch.long),
            "collision": boxes.new_zeros((0,)),
        }

    scores = iou[pairs[:, 0], pairs[:, 1]]
    if pairs.shape[0] > cfg.collision_topk:
        topk = torch.topk(scores, k=cfg.collision_topk, largest=True)
        keep = topk.indices
        pairs = pairs[keep]
        scores = scores[keep]

    collision = (scores * 5.0).clamp(0.0, 1.0)

    return {
        "stability": unstable,
        "slip": slip,
        "support": support,
        "collision_pairs": pairs.long(),
        "collision": collision,
    }


class IntuitionFields(nn.Module):

    def __init__(self, cfg: Optional[IntuitionConfig] = None, heads_cfg: Optional[IntuitionHeadsConfig] = None):
        super().__init__()
        self.cfg = cfg or IntuitionConfig()

        self.heads: Optional[IntuitionHeads] = None
        if self.cfg.learned_heads:
            hc = heads_cfg or IntuitionHeadsConfig(in_dim=self.cfg.feature_dim)
            self.heads = IntuitionHeads(hc)

    @torch.no_grad()
    def forward(
        self,
        det: Dict[str, torch.Tensor],
        relations: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        heur = _heuristic_fields(det, self.cfg)

        if self.heads is None or self.cfg.learned_weight <= 0.0:
            return heur

        ent_feats = _build_entity_features(det, self.cfg)
        pair_idx = heur["collision_pairs"]
        preds = self.heads(ent_feats, pair_index=pair_idx)

        out = dict(heur)

        if "stability_logits" in preds:
            out["stability"] = (
                self.cfg.heuristic_weight * heur["stability"] + self.cfg.learned_weight * torch.sigmoid(preds["stability_logits"])
            ).clamp(0.0, 1.0)

        if "slip_logits" in preds:
            out["slip"] = (
                self.cfg.heuristic_weight * heur["slip"] + self.cfg.learned_weight * torch.sigmoid(preds["slip_logits"])
            ).clamp(0.0, 1.0)

        if "support_logits" in preds:
            out["support"] = (
                self.cfg.heuristic_weight * heur["support"] + self.cfg.learned_weight * torch.sigmoid(preds["support_logits"])
            ).clamp(0.0, 1.0)

        if "collision_logits" in preds:
            out["collision"] = (
                self.cfg.heuristic_weight * heur["collision"] + self.cfg.learned_weight * torch.sigmoid(preds["collision_logits"])
            ).clamp(0.0, 1.0)

        return out

    @torch.no_grad()
    def to_python(self, fields: Dict[str, torch.Tensor]) -> Dict:
        out: Dict = {}
        for k, v in fields.items():
            if torch.is_tensor(v):
                out[k] = v.detach().cpu().tolist()
            else:
                out[k] = v
        return out
