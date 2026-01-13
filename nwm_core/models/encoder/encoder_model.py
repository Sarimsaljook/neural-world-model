from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DinoV2Backbone
from .heads.instseg import Mask2FormerInstSeg, Mask2FormerConfig
from .heads.depth import DepthHead
from .heads.flow import FlowHead
from .heads.uncertainty import UncertaintyHead
from .heads.keypoints import KeypointHead


def _pad_to_multiple(x: torch.Tensor, m: int) -> torch.Tensor:
    b, c, h, w = x.shape
    ph = (m - (h % m)) % m
    pw = (m - (w % m)) % m
    if ph == 0 and pw == 0:
        return x
    return F.pad(x, (0, pw, 0, ph), mode="replicate")


class EvidenceEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        dino_name: str = "dinov2_vitb14",
        use_keypoints: bool = False,
        instseg_cfg: Optional[Mask2FormerConfig] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone = DinoV2Backbone(model_name=dino_name)

        self.instseg = Mask2FormerInstSeg(instseg_cfg or Mask2FormerConfig())
        self.depth = DepthHead(dim=self.backbone.out_dim)
        self.flow = FlowHead(dim=self.backbone.out_dim)
        self.uncertainty = UncertaintyHead(feat_dim=self.backbone.out_dim)

        self.use_keypoints = bool(use_keypoints)
        self.keypoints = KeypointHead(dim=self.backbone.out_dim) if self.use_keypoints else None

    @torch.no_grad()
    def forward(self, frame_t: torch.Tensor, prev_frame: Optional[torch.Tensor] = None) -> Dict:
        feat_map, pad_meta = self.backbone.forward_map(frame_t)  # (B,D,gh,gw) + meta

        inst = self.instseg(frame_t, feat_map, class_count=self.num_classes)
        depth = self.depth(feat_map)

        if prev_frame is not None:
            prev_feat, _ = self.backbone.forward_map(prev_frame)
            flow = self.flow(feat_map, prev_feat)
        else:
            flow = None

        uncert = self.uncertainty(feat_map, inst)

        out = {
            "instances": inst,
            "depth": depth,
            "flow": flow,
            "pixel_uncertainty": uncert["pixel"],
            "pad_meta": pad_meta,
        }
        if uncert["instance"] is not None:
            out["instance_uncertainty"] = uncert["instance"]
        return out
