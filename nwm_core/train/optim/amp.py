from __future__ import annotations

from dataclasses import dataclass
import torch

@dataclass
class AmpConfig:
    enable: bool = True
    dtype: str = "bf16"

    def autocast(self):
        if not self.enable:
            return torch.autocast("cuda", enabled=False)
        dt = torch.bfloat16 if self.dtype.lower() == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dt)
