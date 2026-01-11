from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class RuleMemory:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"env": {}, "rules": {}}, indent=2), encoding="utf-8")

    def load(self) -> Dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, d: Dict) -> None:
        self.path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    def update_rule(self, rule: str, delta: float) -> None:
        d = self.load()
        rules = d.setdefault("rules", {})
        rules[rule] = float(rules.get(rule, 0.0) + delta)
        self.save(d)

    def score(self, rule: str) -> float:
        d = self.load()
        return float(d.get("rules", {}).get(rule, 0.0))
