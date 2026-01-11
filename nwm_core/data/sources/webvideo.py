from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from ..shards.reader import ShardReader

@dataclass
class WebVideoDataset:
    shards: list[Path]

    def __iter__(self) -> Iterator[Dict]:
        for sp in self.shards:
            with ShardReader(sp) as r:
                for key, sample in r.iter_samples():
                    # expected: rgb.npy (T,H,W,3), meta.json optional
                    out: Dict = {"key": key}
                    if "rgb.npy" in sample:
                        out["rgb"] = np.load(BytesIO(sample["rgb.npy"]))  # type: ignore
                    if "meta.json" in sample:
                        out["meta"] = json.loads(sample["meta.json"].decode("utf-8"))
                    yield out

from io import BytesIO
