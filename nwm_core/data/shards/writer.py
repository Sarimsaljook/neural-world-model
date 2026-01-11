from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple, Union

import numpy as np

@dataclass
class ShardWriter:
    out_path: Path
    mode: str = "w"

    def __post_init__(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "ShardWriter":
        self._tf = tarfile.open(str(self.out_path), self.mode)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tf.close()

    def write(self, key: str, files: Dict[str, bytes]) -> None:
        for suffix, data in files.items():
            name = f"{key}.{suffix}"
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            self._tf.addfile(ti, io.BytesIO(data))
