from __future__ import annotations

from typing import Any, Dict
from ...common.logging import get_logger

log = get_logger("nwm.train.stage6_memory")

def run(cfg: Dict[str, Any]) -> None:
    train = cfg.get("train", {})
    max_steps = int(train.get("max_steps", 0))
    if max_steps <= 0:
        raise ValueError("train.max_steps must be > 0")
    log.info("Stage %s starting (max_steps=%d)", "stage6_memory", max_steps)
    log.info("Stage %s complete", "stage6_memory")
