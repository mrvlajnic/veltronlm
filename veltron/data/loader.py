"""Memory-mapped dataset reading and batch assembly.

The loader is deliberately simple: shards are ``uint16``/``uint32`` files written by
:func:`veltron.data.pipeline.write_shards`, memory-mapped with ``numpy`` and sliced into
fixed-length windows. That keeps the resident set proportional to the batch rather than to
the corpus, which is what allows a 12GB card to train on a corpus larger than RAM.

Sequences are drawn at random offsets *within* a shard rather than only at shard
boundaries, so no sequence is ever a concatenation of two unrelated documents unless the
document boundaries were crossed deliberately during packing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class ShardInfo:
    path: Path
    tokens: int
    sha256: str

    @property
    def name(self) -> str:
        return self.path.name


def read_index(split_dir: str | Path, prefix: str = "train") -> tuple[list[ShardInfo], int, str]:
    split_dir = Path(split_dir)
    idx_path = split_dir / f"{prefix}-index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"missing shard index: {idx_path}")
    payload = json.loads(idx_path.read_text(encoding="utf-8"))
    shards = [ShardInfo(split_dir / s["path"], s["tokens"], s.get("sha256", "")) for s in payload["shards"]]
    return shards, int(payload["total_tokens"]), payload.get("dtype", "uint32")


class PackedDataset(Dataset):
    """Random fixed-length windows over memory-mapped token shards.

    Args:
        split_dir: directory containing ``<prefix>-index.json`` and the shards.
        seq_len: training sequence length in tokens.
        prefix: ``train`` or ``val``.
        seed: base RNG seed; :meth:`set_epoch` re-derives per-epoch shuffling.
        limit_tokens: optional cap, used by evaluation to bound runtime.
    """

    def __init__(
        self,
        split_dir: str | Path,
        seq_len: int,
        prefix: str = "train",
        seed: int = 1234,
        limit_tokens: int | None = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.seq_len = seq_len
        self.prefix = prefix
        self.seed = seed
        self.epoch = 0

        shards, total, dtype = read_index(split_dir, prefix)
        if dtype not in ("uint16", "uint32"):
            raise ValueError(f"unsupported shard dtype {dtype!r}; expected uint16 or uint32")
        self.np_dtype = np.uint16 if dtype == "uint16" else np.uint32
        self.shards = shards

        # Cross-check the recorded dtype against geometry. A mismatch would reinterpret
        # every token id and surface later as an out-of-bounds embedding lookup, which
        # reads like a model bug rather than a data bug, so it is caught here.
        for s in shards:
            if s.tokens == 0:
                continue
            width = s.path.stat().st_size / s.tokens
            expected = 4.0 if dtype == "uint32" else 2.0
            if abs(width - expected) > 0.05:
                raise ValueError(
                    f"shard {s.name} holds {s.tokens} tokens in {s.path.stat().st_size} bytes "
                    f"({width:.2f} bytes/token) but the index declares {dtype} "
                    f"({expected:.0f} bytes/token). Re-run scripts/repair_shard_index.py."
                )
        self.total_tokens = min(total, limit_tokens) if limit_tokens else total
        self._arrays: dict[int, np.ndarray] = {}
        self._cumulative: list[int] = []
        run = 0
        for s in shards:
            run += max(0, s.tokens - seq_len + 1)
            self._cumulative.append(run)
        self.n_windows = run

        if self.n_windows == 0:
            raise ValueError(
                f"no full {seq_len}-token windows in split {prefix!r}: "
                f"{len(shards)} shards, {total} tokens"
            )

    def _load(self, idx: int) -> np.ndarray:
        arr = self._arrays.get(idx)
        if arr is None:
            arr = np.memmap(self.shards[idx].path, dtype=self.np_dtype, mode="r")
            self._arrays[idx] = arr
        return arr

    def _locate(self, window: int) -> tuple[int, int]:
        lo, hi = 0, len(self._cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if window < self._cumulative[mid]:
                hi = mid
            else:
                lo = mid + 1
        shard = lo
        offset_in_shard = window - (self._cumulative[shard - 1] if shard else 0)
        return shard, offset_in_shard

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        shard_i, off = self._locate(int(i))
        arr = self._load(shard_i)
        chunk = np.asarray(arr[off : off + self.seq_len + 1])
        if chunk.size < self.seq_len + 1:
            chunk = np.pad(chunk, (0, self.seq_len + 1 - chunk.size))
        t = torch.from_numpy(chunk.astype(np.int64))
        return {"input_ids": t[:-1].contiguous(), "labels": t[1:].contiguous()}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def stats(self) -> dict[str, Any]:
        return {
            "split_dir": str(self.split_dir),
            "prefix": self.prefix,
            "shards": len(self.shards),
            "total_tokens": self.total_tokens,
            "seq_len": self.seq_len,
            "windows": self.n_windows,
            "dtype": self.np_dtype.__name__,
        }


def collate(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


class TokenShardLoader:
    """Convenience wrapper producing a DataLoader with reproducible shuffling.

    Shuffling is done over *window indices* with a per-epoch generator. Two consecutive
    epochs therefore visit windows in different orders, while a given (seed, epoch) pair
    always produces the same order -- which is what makes a resumed run reproduce the
    uninterrupted one.
    """

    def __init__(self, dataset: PackedDataset, batch_size: int, shuffle: bool = True,
                 num_workers: int = 0, seed: int = 1234, drop_last: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
            drop_last=drop_last,
            generator=gen if shuffle else None,
            pin_memory=False,
        )
        self._gen = gen

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        # Re-seed the sampler so each epoch's permutation is a fresh, reproducible draw.
        state = torch.get_rng_state()
        torch.set_rng_state(state)
        self._gen.manual_seed(self.seed + epoch)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)

    def tokens_per_batch(self) -> int:
        return self.batch_size * self.dataset.seq_len


def build_dataloaders(
    dataset_root: str | Path,
    seq_len: int,
    micro_batch_size: int,
    seed: int = 1234,
    num_workers: int = 0,
    val_limit_tokens: int | None = None,
) -> tuple[TokenShardLoader, TokenShardLoader, PackedDataset, PackedDataset]:
    """Construct train and validation loaders from a built dataset directory."""
    root = Path(dataset_root)
    train_ds = PackedDataset(root / "train", seq_len, prefix="train", seed=seed)
    val_ds = PackedDataset(root / "val", seq_len, prefix="val", seed=seed, limit_tokens=val_limit_tokens)
    train_loader = TokenShardLoader(train_ds, micro_batch_size, shuffle=True,
                                    num_workers=num_workers, seed=seed)
    val_loader = TokenShardLoader(val_ds, micro_batch_size, shuffle=False,
                                  num_workers=0, seed=seed, drop_last=False)
    log.info("train: %s", train_ds.stats())
    log.info("val:   %s", val_ds.stats())
    return train_loader, val_loader, train_ds, val_ds


__all__ = [
    "PackedDataset",
    "TokenShardLoader",
    "ShardInfo",
    "read_index",
    "build_dataloaders",
    "collate",
]
