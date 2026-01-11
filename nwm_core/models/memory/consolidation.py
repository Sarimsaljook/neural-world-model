from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .semantic import SemanticMemory
from .rule_memory import RuleMemory


@dataclass
class ConsolidationConfig:
    attach_bonus: float = 0.02
    support_bonus: float = 0.01
    contact_bonus: float = 0.005


def consolidate_from_relations(
    semantic: SemanticMemory,
    rules: RuleMemory,
    rels: List[Dict],
    cfg: ConsolidationConfig = ConsolidationConfig(),
) -> None:
    semantic.update_relations(rels)
    for r in rels:
        t = r["type"]
        if t == "attached":
            rules.update_rule("attach_persistence", cfg.attach_bonus)
        elif t == "supporting":
            rules.update_rule("support_common", cfg.support_bonus)
        elif t == "contact":
            rules.update_rule("contact_common", cfg.contact_bonus)
