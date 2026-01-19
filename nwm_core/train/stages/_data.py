from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import torch

from nwm_core.data.shards.reader import ReaderCfg, iter_samples


@dataclass(frozen=True)
class DataPipe:
    train_glob: str
    val_glob: str
    reader: ReaderCfg
    batch_size: int = 8
    num_batches_per_epoch: int = 2000

    def _batcher(self, it: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        bs = int(self.batch_size)
        while True:
            batch = []
            for _ in range(bs):
                batch.append(next(it))
            # stack video: list[(T,3,H,W)] -> (B,T,3,H,W)
            v = torch.stack([x["video"] for x in batch], dim=0)
            out = {
                "video": v,
                "template_id": torch.tensor([x["template_id"] for x in batch], dtype=torch.long),
                "template": [x["template"] for x in batch],
                "placeholders": [x["placeholders"] for x in batch],
                "id": [x["id"] for x in batch],
                "raw": [x["raw"] for x in batch],
            }
            yield out

    def train_iter(self) -> Iterator[Dict[str, Any]]:
        it = iter_samples(self.train_glob, self.reader, infinite=True)
        return self._batcher(it)

    def val_iter(self) -> Iterator[Dict[str, Any]]:
        it = iter_samples(self.val_glob, self.reader, infinite=True)
        return self._batcher(it)


def build_datapipe(data_cfg: Dict[str, Any], stage_cfg: Dict[str, Any]) -> DataPipe:
    root = data_cfg["root"]
    shards_glob = data_cfg["shards_glob"]
    val_glob = data_cfg["val_shards_glob"]

    train_glob = shards_glob.replace("${configs.data.ssv2.root}", root).replace("${paths.data_root}", "")
    val_glob = val_glob.replace("${configs.data.ssv2.root}", root).replace("${paths.data_root}", "")

    r = data_cfg.get("sampling", {}) or {}
    aug = data_cfg.get("augment", {}) or {}

    reader = ReaderCfg(
        clip_len=int(r.get("clip_len", 16)),
        stride=int(r.get("stride", 2)),
        fps_target=int(r.get("fps_target", 15)),
        resize_short=int(stage_cfg.get("data", {}).get("resize_short", 256)),
        max_size=int(stage_cfg.get("data", {}).get("max_size", 384)),
        shuffle_buffer=int((data_cfg.get("streaming", {}) or {}).get("shuffle_buffer", 4096)),
        seed=int(stage_cfg.get("seed", 1337)),
    )

    return DataPipe(
        train_glob=train_glob,
        val_glob=val_glob,
        reader=reader,
        batch_size=int(stage_cfg.get("batch_size", 8)),
        num_batches_per_epoch=int(stage_cfg.get("steps_per_epoch", 2000)),
    )
