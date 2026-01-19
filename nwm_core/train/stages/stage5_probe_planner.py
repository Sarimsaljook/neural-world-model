from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from nwm_core.models.planning.probing import ProbingPolicy
from nwm_core.models.planning.mpc import MicroMPC
from nwm_core.models.planning.policy_distill import DistillConfig, DistillationBuffer, DistillationPolicyHead

from nwm_core.train.optim import AMP, WarmupCosine, clip_grad
from nwm_core.train.stages._data import build_datapipe


def _frame(video: torch.Tensor, t: int) -> torch.Tensor:
    return video[:, t]


class Stage5ProbePlanner:
    def __init__(self, device, data_cfg, stage_cfg, workdir: Path, logger, ckpt):
        self.device = device
        self.data_cfg = data_cfg
        self.cfg = stage_cfg
        self.workdir = workdir
        self.logger = logger
        self.ckpt = ckpt

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        encoder = state["encoder"].to(self.device).eval()
        compiler = state["compiler"]
        intuition = state.get("intuition", None)

        prober = ProbingPolicy().to(self.device)
        mpc = MicroMPC()

        # Distill scaffold: train a small head to imitate “good” probes/actions from heuristic labels
        dcfg = DistillConfig(device=str(self.device), embed_dim=int(self.cfg.get("embed_dim", 256)), max_items=int(self.cfg.get("max_items", 50000)))
        buf = DistillationBuffer(dcfg)
        head = DistillationPolicyHead(in_dim=dcfg.embed_dim, out_dim=int(self.cfg.get("out_dim", 32))).to(self.device)

        params = [p for p in prober.parameters() if p.requires_grad] + [p for p in head.parameters() if p.requires_grad]

        pipe = build_datapipe(self.data_cfg, self.cfg)
        it = pipe.train_iter()

        lr = float(self.cfg.get("lr", 2e-4))
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(self.cfg.get("weight_decay", 0.02)))
        sched = WarmupCosine(base_lr=lr, warmup_steps=int(self.cfg.get("warmup_steps", 1000)), total_steps=int(self.cfg.get("total_steps", 15000)), min_lr=float(self.cfg.get("min_lr", 1e-6)))
        amp = AMP(enabled=bool(self.cfg.get("amp", True)))

        step = 0
        total_steps = int(self.cfg.get("total_steps", 15000))
        t0 = time.time()

        while step < total_steps:
            batch = next(it)
            video = batch["video"].to(self.device, non_blocking=True)

            x0 = _frame(video, 0)
            x1 = _frame(video, 1)

            with torch.no_grad():
                ev0 = encoder(x0, None)
                ev1 = encoder(x1, x0)

            w1 = compiler.step(ev1, dt=1.0 / 15.0)
            erfg = w1["erfg"]
            events = w1.get("events", [])

            intu = {}
            if intuition is not None:
                with torch.no_grad():
                    intu = intuition.step(erfg, dt=1.0 / 15.0, events=events, evidence=ev1) or {}

            # heuristic “teacher” probe/action (what you already do in demo)
            with torch.no_grad():
                probe = prober.suggest(erfg, intu, events=events, goal=None) or {}
                act = mpc.propose_action(erfg, 0.0, events, intuition=intu, goal=None) or {}

            # distill target: compress probe+action into a discrete id vector (simple but stable)
            target = torch.zeros((int(self.cfg.get("out_dim", 32)),), device=self.device)
            target[0] = 1.0 if probe else 0.0
            target[1] = 1.0 if act else 0.0

            # input: pick a stable embedding from evidence instances if present
            inst = ev1.get("instances", {}) if isinstance(ev1, dict) else {}
            embs = inst.get("embeddings", None)
            if isinstance(embs, torch.Tensor) and embs.numel() > 0:
                x_in = embs[:, 0, :dcfg.embed_dim].detach()
            else:
                x_in = torch.zeros((video.shape[0], dcfg.embed_dim), device=self.device)

            # buffer for online training stability
            buf.add(x_in.mean(dim=0), target)

            X, Y = buf.sample(int(self.cfg.get("batch_buf", 64)))
            if X.numel() == 0:
                step += 1
                continue

            X = X.to(self.device)
            Y = Y.to(self.device)

            with amp.autocast():
                pred = head(X)
                loss = F.mse_loss(pred, Y)

            opt.zero_grad(set_to_none=True)
            amp.backward(loss)
            amp.unscale_(opt)
            gn = clip_grad(head, float(self.cfg.get("grad_clip", 1.0)))
            for pg in opt.param_groups:
                pg["lr"] = sched.lr(step)
            amp.step(opt)

            if step % int(self.cfg.get("log_every", 50)) == 0:
                self.logger.log({"stage": "stage5_probe_planner", "step": step, "loss": float(loss.item()), "grad_norm": float(gn), "sec": time.time() - t0})

            if step % int(self.cfg.get("ckpt_every", 2000)) == 0 and step > 0:
                self.ckpt.save(f"stage5_probe_planner_step{step:07d}", {"prober": prober.state_dict(), "distill_head": head.state_dict(), "step": step})

            step += 1

        state["prober"] = prober.eval()
        state["mpc"] = mpc
        state["distill"] = head.eval()
        return state
