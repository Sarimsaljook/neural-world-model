from __future__ import annotations

import time
from dataclasses import dataclass

@dataclass
class TimeSync:
    def now_ns(self) -> int:
        return int(time.time() * 1e9)
