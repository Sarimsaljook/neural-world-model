from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

import numpy as np

@dataclass
class RoboticsDataset:
    episodes: list[Path]

    def __iter__(self) -> Iterator[Dict]:
        for p in self.episodes:
            d = np.load(p, allow_pickle=True)
            out = {k: d[k].item() if d[k].dtype == object else d[k] for k in d.files}
            out["key"] = p.stem
            yield out
