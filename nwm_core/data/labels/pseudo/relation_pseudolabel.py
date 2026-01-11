from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class RelationPseudoLabeler:
    def infer(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        # return empty when unknown
        return []
