from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class SimGroundTruth:
    def extract(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "state": sample.get("state"),
            "actions": sample.get("actions")
        }
