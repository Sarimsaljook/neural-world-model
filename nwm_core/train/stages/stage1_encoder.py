from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nwm_core.train.checkpoints.model_zoo import build_core
from nwm_core.train.optim import AMP, WarmupCosine, clip_grad
from nwm_core.train.stages._data import build_datapipe


# --- 1. Define the Adapter ---
class TrainableAdapter(nn.Module):
    """
    Learns to project frozen visual embeddings into the NWM latent space.
    """

    def __init__(self, input_dim=768, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderWithAdapter(nn.Module):
    def __init__(self, frozen_encoder: nn.Module, adapter: nn.Module):
        super().__init__()
        self.frozen_encoder = frozen_encoder
        self.adapter = adapter

    def train(self, mode=True):
        self.frozen_encoder.eval()
        self.adapter.train(mode)
        return self

    def forward(self, x: torch.Tensor, prev: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        with torch.no_grad():
            out = self.frozen_encoder(x, prev)

        # Apply Adapter to Embeddings
        inst = out.get("instances", {})
        if "embeddings" in inst and isinstance(inst["embeddings"], torch.Tensor):
            raw_emb = inst["embeddings"]
            if raw_emb.numel() > 0:
                # [B, N, D] -> Adapter -> [B, N, D]
                inst["embeddings"] = self.adapter(raw_emb)

        return out


def _frame_batch(video: torch.Tensor, t: int) -> torch.Tensor:
    return video[:, t]


def _safe_get(d: Dict[str, Any], k: str, default=None):
    return d.get(k, default) if isinstance(d, dict) else default


class Stage1Encoder:
    def __init__(self, device, data_cfg, stage_cfg, workdir: Path, logger, ckpt):
        self.device = device
        self.data_cfg = data_cfg
        self.cfg = stage_cfg
        self.workdir = workdir
        self.logger = logger
        self.ckpt = ckpt

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[{self.__class__.__name__}] Initializing...")

        core = build_core(self.device, cfg=self.cfg.get("model", {}))
        frozen_encoder = core["encoder"].eval()

        adapter = TrainableAdapter().to(self.device)
        encoder = EncoderWithAdapter(frozen_encoder, adapter).to(self.device)

        print(
            f"[{self.__class__.__name__}] Encoder is Frozen. Training Adapter (Params: {sum(p.numel() for p in adapter.parameters()):,})")

        pipe = build_datapipe(self.data_cfg, self.cfg)
        it = pipe.train_iter()

        lr = float(self.cfg.get("lr", 3e-4))
        wd = float(self.cfg.get("weight_decay", 0.05))

        # Optimize ONLY the Adapter
        opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=wd)

        total_steps = int(self.cfg.get("total_steps", 50000))
        sched = WarmupCosine(base_lr=lr, warmup_steps=int(self.cfg.get("warmup_steps", 2000)), total_steps=total_steps,
                             min_lr=float(self.cfg.get("min_lr", 3e-6)))

        amp = AMP(enabled=bool(self.cfg.get("amp", True)))
        grad_max = float(self.cfg.get("grad_clip", 1.0))

        step = 0
        t0 = time.time()

        while step < total_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = pipe.train_iter()
                batch = next(it)

            video = batch["video"].to(self.device, non_blocking=True)  # (B,T,3,H,W)

            x0 = _frame_batch(video, 0)
            x1 = _frame_batch(video, 1)

            with amp.autocast():
                # Forward passes
                out0 = encoder(x0, None)
                out1 = encoder(x1, x0)

                inst0 = _safe_get(out0, "instances", {})
                inst1 = _safe_get(out1, "instances", {})

                e0 = _safe_get(inst0, "embeddings", None)
                e1 = _safe_get(inst1, "embeddings", None)

                loss = torch.tensor(0.0, device=self.device)
                valid_loss = False

                # Embedding Loss
                if isinstance(e0, torch.Tensor) and isinstance(e1, torch.Tensor) and e0.numel() > 0 and e1.numel() > 0:
                    q = min(e0.shape[1], e1.shape[1], int(self.cfg.get("topk", 16)))
                    a = F.normalize(e0[:, :q], dim=-1)
                    b = F.normalize(e1[:, :q], dim=-1)

                    # maximize similarity between tracked instances across frames
                    loss_emb = (1.0 - (a * b).sum(dim=-1)).mean()
                    loss = loss + (float(self.cfg.get("w_emb", 1.0)) * loss_emb)
                    valid_loss = True

            if not valid_loss:
                if step % 50 == 0:
                    print(f"Step {step}: Warning - No valid loss. Skipping update.")
                step += 1
                opt.zero_grad(set_to_none=True)
                continue

            # Optimization
            opt.zero_grad(set_to_none=True)
            amp.backward(loss)
            amp.unscale_(opt)
            gn = clip_grad(adapter, grad_max)
            for pg in opt.param_groups:
                pg["lr"] = sched.lr(step)
            amp.step(opt)

            if step % int(self.cfg.get("log_every", 50)) == 0:
                dt = time.time() - t0
                loss_val = loss.item()
                self.logger.log({
                    "stage": "stage1_encoder",
                    "step": step,
                    "loss": loss_val,
                    "grad_norm": float(gn),
                    "lr": sched.lr(step),
                    "sec": dt
                })
                t0 = time.time()

            if step % int(self.cfg.get("ckpt_every", 2000)) == 0 and step > 0:
                self.ckpt.save(f"stage1_adapter_step{step:07d}", {"adapter": adapter.state_dict(), "step": step})

            step += 1

        encoder.eval()
        state["encoder"] = encoder
        state["compiler"] = core["compiler"]
        return state