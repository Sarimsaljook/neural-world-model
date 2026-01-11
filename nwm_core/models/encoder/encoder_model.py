from __future__ import annotations

import torch
from torch import nn

from .backbone import SimpleBackbone
from .heads.instseg import InstSegHead
from .heads.depth import DepthHead
from .heads.flow import FlowHead
from .heads.uncertainty import UncertaintyHead

class EncoderModel(nn.Module):
    def __init__(self, num_classes: int = 120, feat_dim: int = 256) -> None:
        super().__init__()
        self.backbone = SimpleBackbone(in_ch=3, dim=feat_dim)
        self.instseg = InstSegHead(feat_dim, num_classes)
        self.depth = DepthHead(feat_dim)
        self.flow = FlowHead(feat_dim)
        self.unc = UncertaintyHead(feat_dim)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone(x)
        out = {}
        out.update(self.instseg(feat))
        out["depth"] = self.depth(feat)
        out["flow"] = self.flow(feat)
        out["log_var"] = self.unc(feat)
        out["feat"] = feat
        return out
