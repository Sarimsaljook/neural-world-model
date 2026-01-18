from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PadMeta:
    orig_hw: Tuple[int, int]
    padded_hw: Tuple[int, int]
    pad_bottom: int
    pad_right: int
    patch: int
    resized_hw: Tuple[int, int]


def pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, PadMeta]:
    _, _, H0, W0 = x.shape
    Hp = (H0 + multiple - 1) // multiple * multiple
    Wp = (W0 + multiple - 1) // multiple * multiple
    pb = Hp - H0
    pr = Wp - W0

    if pb == 0 and pr == 0:
        meta = PadMeta(
            orig_hw=(H0, W0),
            padded_hw=(H0, W0),
            pad_bottom=0,
            pad_right=0,
            patch=multiple,
            resized_hw=(H0 // multiple, W0 // multiple),
        )
        return x, meta

    x_pad = F.pad(x, (0, pr, 0, pb), mode="replicate")
    meta = PadMeta(
        orig_hw=(H0, W0),
        padded_hw=(Hp, Wp),
        pad_bottom=pb,
        pad_right=pr,
        patch=multiple,
        resized_hw=(Hp // multiple, Wp // multiple),
    )
    return x_pad, meta


class DinoV2Backbone(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.out_dim = self.model.embed_dim
        self.patch = self.model.patch_size

    @torch.no_grad()
    def forward_map(self, x: torch.Tensor) -> Tuple[torch.Tensor, PadMeta]:
        x_pad, meta = pad_to_multiple(x, self.patch)

        tokens = self.model.get_intermediate_layers(x_pad, n=1)[0]
        tok = tokens[:, 1:, :]
        B, N, D = tok.shape

        # infer grid from N
        gh = int(math.floor(math.sqrt(N)))
        gw = int(math.ceil(N / gh))

        if gh * gw != N:
            pad = gh * gw - N
            tok = torch.cat([tok, tok[:, -1:, :].expand(B, pad, D)], dim=1)

        feat = tok.transpose(1, 2).contiguous().view(B, D, gh, gw)

        meta = PadMeta(
            orig_hw=meta.orig_hw,
            padded_hw=meta.padded_hw,
            pad_bottom=meta.pad_bottom,
            pad_right=meta.pad_right,
            patch=meta.patch,
            resized_hw=(gh, gw),
        )
        return feat, meta
