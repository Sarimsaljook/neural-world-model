from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

@dataclass
class ContactDiscovery:
    dist_thresh: float = 0.03

    def infer_contacts(self, positions: Dict[str, List[Tuple[float,float,float]]]) -> List[Dict[str, Any]]:
        # positions: entity_id -> list of xyz over time
        keys = list(positions.keys())
        out: List[Dict[str, Any]] = []
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a, b = keys[i], keys[j]
                pa, pb = positions[a], positions[b]
                n = min(len(pa), len(pb))
                for t in range(n):
                    ax, ay, az = pa[t]
                    bx, by, bz = pb[t]
                    dx, dy, dz = ax-bx, ay-by, az-bz
                    d2 = dx*dx + dy*dy + dz*dz
                    if d2 <= self.dist_thresh * self.dist_thresh:
                        out.append({"t": t, "a": a, "b": b, "type": "contact"})
                        break
        return out
