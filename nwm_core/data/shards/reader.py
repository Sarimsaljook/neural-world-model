from __future__ import annotations

import json
import random
import tarfile
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, Iterator, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F


@dataclass(frozen=True)
class ReaderCfg:
    clip_len: int = 16
    stride: int = 2
    fps_target: int = 15
    resize_short: int = 256
    max_size: int = 384
    shuffle_buffer: int = 4096
    seed: int = 1337


def _list_shards(glob_pat: str) -> List[str]:
    xs = sorted(glob(glob_pat))
    if not xs:
        raise FileNotFoundError(f"No shards match: {glob_pat}")
    return xs


def _decode_webm_bytes(webm: bytes) -> List[np.ndarray]:
    tmp = np.frombuffer(webm, dtype=np.uint8)
    cap = cv2.VideoCapture()
    # OpenCV cannot open from memory directly; use tempfile-like trick via VideoCapture API is not supported.
    # So we use cv2.imdecode only for images; for video we must write to RAM-backed file.
    # On Windows, simplest is NamedTemporaryFile; but to avoid disk seeks, we keep it small and sequential.
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(webm)
        name = f.name

    cap.open(name)
    frames: List[np.ndarray] = []
    ok = True
    while ok:
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(fr)
    cap.release()
    try:
        os.remove(name)
    except Exception:
        pass
    return frames


def _resize_keep_ar(frame_bgr: np.ndarray, short: int, max_size: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    s = short / float(min(h, w))
    nh = int(round(h * s))
    nw = int(round(w * s))
    if max(nh, nw) > max_size:
        s2 = max_size / float(max(nh, nw))
        nh = int(round(nh * s2))
        nw = int(round(nw * s2))
    return cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def _sample_clip(frames: List[np.ndarray], cfg: ReaderCfg) -> List[np.ndarray]:
    if not frames:
        return []
    T = len(frames)
    need = cfg.clip_len * cfg.stride
    if T < need:
        # loop pad
        out = []
        idx = 0
        for _ in range(cfg.clip_len):
            out.append(frames[idx % T])
            idx += cfg.stride
        return out
    start = random.randint(0, T - need)
    out = []
    idx = start
    for _ in range(cfg.clip_len):
        out.append(frames[idx])
        idx += cfg.stride
    return out


def _to_tensor(frames_bgr: List[np.ndarray], cfg: ReaderCfg) -> torch.Tensor:
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    x = np.stack(rgb, axis=0)
    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()  # [T, 3, H, W]

    x = F.resize(x, cfg.resize_short)
    x = F.center_crop(x, (cfg.resize_short, cfg.max_size))

    return x.float() / 255.0


def iter_samples(shards_glob: str, cfg: ReaderCfg, infinite: bool = True) -> Iterator[Dict[str, Any]]:
    rng = random.Random(cfg.seed)
    shards = _list_shards(shards_glob)
    order = list(range(len(shards)))

    def _stream_tar(path: str) -> Iterator[Tuple[str, bytes]]:
        with tarfile.open(path, "r:*") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                yield m.name, f.read()

    buffer: List[Dict[str, Any]] = []

    while True:
        rng.shuffle(order)
        for si in order:
            tar_path = shards[si]
            cur: Dict[str, Dict[str, bytes]] = {}

            for name, data in _stream_tar(tar_path):
                # name: "000000123.webm" or "000000123.json"
                if "." not in name:
                    continue
                key, ext = name.rsplit(".", 1)
                if key not in cur:
                    cur[key] = {}
                cur[key][ext] = data

                if "webm" in cur[key] and "json" in cur[key]:
                    meta = json.loads(cur[key]["json"].decode("utf-8"))
                    frames = _decode_webm_bytes(cur[key]["webm"])
                    clip = _sample_clip(frames, cfg)
                    if not clip:
                        continue
                    vid = _to_tensor(clip, cfg)

                    sample = {
                        "video": vid,  # (T,3,H,W) float
                        "template_id": int(meta.get("template_id", -1)),
                        "template": meta.get("template", ""),
                        "label": meta.get("label", ""),
                        "placeholders": meta.get("placeholders", []),
                        "id": meta.get("id", ""),
                        "raw": meta,
                    }

                    if cfg.shuffle_buffer > 0:
                        buffer.append(sample)
                        if len(buffer) >= cfg.shuffle_buffer:
                            rng.shuffle(buffer)
                            while buffer:
                                yield buffer.pop()
                    else:
                        yield sample

                    cur.pop(key, None)

        if not infinite:
            break

    while buffer:
        yield buffer.pop()
