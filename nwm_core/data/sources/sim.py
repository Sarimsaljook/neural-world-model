from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

import numpy as np
from io import BytesIO

from ..shards.reader import ShardReader

@dataclass
class SimDataset:
    shards: list[Path]

    def __iter__(self) -> Iterator[Dict]:
        for sp in self.shards:
            with ShardReader(sp) as r:
                for key, sample in r.iter_samples():
                    out: Dict = {"key": key}
                    if "rgb.npy" in sample:
                        out["rgb"] = np.load(BytesIO(sample["rgb.npy"]))
                    if "state.json" in sample:
                        out["state"] = json.loads(sample["state.json"].decode("utf-8"))
                    if "actions.npy" in sample:
                        out["actions"] = np.load(BytesIO(sample["actions.npy"]))
                    yield out
