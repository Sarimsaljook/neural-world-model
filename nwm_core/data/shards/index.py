from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class ShardIndex:
    shards: List[Path]

    @classmethod
    def from_glob(cls, glob_pattern: str) -> "ShardIndex":
        p = Path(".")
        shards = sorted(p.glob(glob_pattern))
        return cls(shards=shards)
