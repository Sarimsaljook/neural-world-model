from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))

def gaussian_nll(x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    # elementwise NLL
    return 0.5 * (math.log(2.0 * math.pi) + log_var + (x - mean) ** 2 * torch.exp(-log_var))

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    conf = probs
    pred = (probs >= 0.5).astype(np.int64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi)
        if not np.any(m):
            continue
        acc = float(np.mean(pred[m] == labels[m]))
        c = float(np.mean(conf[m]))
        ece_val += float(np.mean(m)) * abs(acc - c)
    return float(ece_val)

def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return (a * b).sum(dim=-1)
