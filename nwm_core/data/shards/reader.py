from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

@dataclass
class ShardReader:
    path: Path

    def __enter__(self) -> "ShardReader":
        self._tf = tarfile.open(str(self.path), "r:*")
        self._members = self._tf.getmembers()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tf.close()

    def iter_samples(self) -> Iterator[Tuple[str, Dict[str, bytes]]]:
        cur_key: Optional[str] = None
        cur: Dict[str, bytes] = {}
        for m in self._members:
            if not m.isfile():
                continue
            name = m.name
            if "." not in name:
                continue
            key, suffix = name.split(".", 1)
            f = self._tf.extractfile(m)
            if f is None:
                continue
            data = f.read()
            if cur_key is None:
                cur_key = key
            if key != cur_key:
                yield cur_key, cur
                cur_key = key
                cur = {}
            cur[suffix] = data
        if cur_key is not None:
            yield cur_key, cur
