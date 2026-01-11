from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class SemanticMemory:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"relations": {}, "types": {}}), encoding="utf-8")

    def _load(self) -> Dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, d: Dict) -> None:
        self.path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    def update_relations(self, rels: List[Dict]) -> None:
        d = self._load()
        rel_stats = d.setdefault("relations", {})
        for r in rels:
            k = f'{r["type"]}'
            rel_stats[k] = int(rel_stats.get(k, 0) + 1)
        self._save(d)

    def relation_prior(self, rel_type: str) -> float:
        d = self._load()
        rel_stats = d.get("relations", {})
        total = sum(int(v) for v in rel_stats.values()) + 1
        return float(rel_stats.get(rel_type, 0) + 1) / float(total)
