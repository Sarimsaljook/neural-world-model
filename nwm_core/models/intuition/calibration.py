from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """
    Temperature scaling for logits calibration.
    Fit on a held-out validation set: minimize NLL.
    """

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(float(init_temp)).log())

    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(min=1e-3, max=100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature()

    @torch.no_grad()
    def calibrate_binary(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        steps: int = 200,
        lr: float = 0.05,
    ) -> float:
        self.train()
        opt = torch.optim.LBFGS([self.log_temp], lr=lr, max_iter=steps)

        targets = targets.float()

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(self.forward(logits), targets)
            loss.backward()
            return loss

        opt.step(closure)
        self.eval()
        return float(self.temperature().item())


@dataclass(frozen=True)
class ExpectedCalibrationError:
    bins: int = 15

    @torch.no_grad()
    def binary_ece(self, probs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, torch.Tensor]:
        probs = probs.detach().float().clamp(0.0, 1.0)
        targets = targets.detach().float()

        edges = torch.linspace(0.0, 1.0, self.bins + 1, device=probs.device)
        ece = torch.zeros((), device=probs.device)
        per_bin = torch.zeros((self.bins, 3), device=probs.device)

        for b in range(self.bins):
            lo = edges[b]
            hi = edges[b + 1]
            m = (probs >= lo) & (probs < hi) if b < self.bins - 1 else (probs >= lo) & (probs <= hi)
            cnt = m.float().sum().clamp(min=1.0)
            acc = targets[m].mean() if m.any() else torch.zeros((), device=probs.device)
            conf = probs[m].mean() if m.any() else torch.zeros((), device=probs.device)
            frac = m.float().mean()
            ece = ece + frac * (acc - conf).abs()

            per_bin[b, 0] = acc
            per_bin[b, 1] = conf
            per_bin[b, 2] = frac

        return float(ece.item()), per_bin
