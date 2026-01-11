from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class MechanismPseudoLabeler:
    def infer(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []
