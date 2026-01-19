from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import torch

from nwm_core.models.memory.episodic import EpisodicMemory
from nwm_core.models.memory.semantic import SemanticMemory
from nwm_core.models.memory.spatial import SpatialMemory
from nwm_core.models.memory.rule_memory import RuleMemory
from nwm_core.models.memory.consolidation import MemoryConsolidator

from nwm_core.train.stages._data import build_datapipe


def _frame(video: torch.Tensor, t: int) -> torch.Tensor:
    return video[:, t]


class Stage6Memory:
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

        episodic = EpisodicMemory()
        semantic = SemanticMemory()
        spatial = SpatialMemory()
        rulemem = RuleMemory()
        consolidator = MemoryConsolidator(episodic, semantic, spatial, rulemem)

        pipe = build_datapipe(self.data_cfg, self.cfg)
        it = pipe.train_iter()

        steps = int(self.cfg.get("steps", 5000))
        t0 = time.time()
        for step in range(steps):
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

            episodic.add(erfg, events, intu)
            semantic.update_from_erfg(erfg)
            spatial.update_from_erfg(erfg)
            rulemem.update(erfg, events)

            consolidator.step(erfg, events)

            if step % int(self.cfg.get("log_every", 200)) == 0:
                self.logger.log({"stage": "stage6_memory", "step": step, "sec": time.time() - t0})

        state["episodic"] = episodic
        state["semantic"] = semantic
        state["spatial"] = spatial
        state["rulemem"] = rulemem
        state["consolidator"] = consolidator
        return state
