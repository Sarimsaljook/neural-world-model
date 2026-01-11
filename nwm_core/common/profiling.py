from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Dict, Iterator

from .logging import get_logger

log = get_logger("nwm.profiling")

@dataclass
class Timer:
    name: str
    start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start

class Profiler:
    def __init__(self) -> None:
        self._times: Dict[str, float] = {}

    @contextlib.contextmanager
    def section(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._times[name] = self._times.get(name, 0.0) + dt

    def snapshot(self, reset: bool = True) -> Dict[str, float]:
        out = dict(self._times)
        if reset:
            self._times.clear()
        return out
