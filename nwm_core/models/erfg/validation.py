from __future__ import annotations

from typing import Tuple
from .state import ERFGState

def validate_erfg(s: ERFGState) -> Tuple[bool, str]:
    if not s.hypotheses:
        return False, "No hypotheses"
    if s.timestamp_ns < 0:
        return False, "Negative timestamp"
    for h in s.hypotheses:
        if h.weight < 0:
            return False, "Negative hypothesis weight"
    return True, "ok"
