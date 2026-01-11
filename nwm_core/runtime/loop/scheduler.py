from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ...common.timing import ClockSpec, MultiRateScheduler

@dataclass
class RuntimeClocks:
    sensory_hz: float = 30.0
    micro_hz: float = 10.0
    event_hz: float = 2.0
    intent_hz: float = 0.2
    narrative_hz: float = 1.0 / 60.0

    def build(self) -> MultiRateScheduler:
        clocks: Dict[str, ClockSpec] = {
            "sensory": ClockSpec("sensory", self.sensory_hz),
            "micro": ClockSpec("micro", self.micro_hz),
            "event": ClockSpec("event", self.event_hz),
            "intent": ClockSpec("intent", self.intent_hz),
            "narrative": ClockSpec("narrative", self.narrative_hz),
        }
        return MultiRateScheduler(clocks)
