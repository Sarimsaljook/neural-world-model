from __future__ import annotations

from typing import Any, Dict
from ...common.logging import get_logger

log = get_logger("nwm.train.stage3_mechanisms")

def run(cfg: Dict[str, Any]) -> None:
    # Fully wired training loops get added after you lock model implementations.
    # This stage runner is compatibility-ready: it validates config keys and exits cleanly.
    train = cfg.get("train", {})
    max_steps = int(train.get("max_steps", 0))
    if max_steps <= 0:
        raise ValueError("train.max_steps must be > 0")
    log.info("Stage %s starting (max_steps=%d)", "stage3_mechanisms", max_steps)
    # Add: build dataloaders -> build models -> optimize -> save checkpoints.
    log.info("Stage %s complete", "stage3_mechanisms")
