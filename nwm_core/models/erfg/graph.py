from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .state import ERFGState, EntityNode, RelationEdge

@dataclass
class ERFGGraph:
    state: ERFGState

    def entities(self) -> Dict[str, EntityNode]:
        # active hypothesis = max weight
        h = max(self.state.hypotheses, key=lambda x: x.weight)
        return h.entities

    def relations(self) -> List[RelationEdge]:
        h = max(self.state.hypotheses, key=lambda x: x.weight)
        return h.relations
