from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

@dataclass(frozen=True)
class ClockSpec:
    name: str
    hz: float

    @property
    def period_s(self) -> float:
        return 1.0 / self.hz

class MultiRateScheduler:
    def __init__(self, clocks: Dict[str, ClockSpec]) -> None:
        self._clocks = clocks
        self._last: Dict[str, float] = {k: 0.0 for k in clocks.keys()}

    def should_tick(self, clock_name: str, now_s: Optional[float] = None) -> bool:
        if now_s is None:
            now_s = time.perf_counter()
        spec = self._clocks[clock_name]
        last = self._last[clock_name]
        return (now_s - last) >= spec.period_s

    def mark_tick(self, clock_name: str, now_s: Optional[float] = None) -> None:
        if now_s is None:
            now_s = time.perf_counter()
        self._last[clock_name] = now_s

    def tick_if_due(self, clock_name: str, fn: Callable[[], None], now_s: Optional[float] = None) -> bool:
        if self.should_tick(clock_name, now_s=now_s):
            fn()
            self.mark_tick(clock_name, now_s=now_s)
            return True
        return False
