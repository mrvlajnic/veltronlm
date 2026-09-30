"""Checkpointing, resume and corruption recovery.

A checkpoint is a directory, not a file:

    checkpoints/step-0001200/
        model.safetensors          weights (+ optimizer.pt, scheduler.pt, rng.pt, trainer.json)
        metadata.json              step, epoch, tokens_seen, wall_clock, dataset/tokenizer hashes
        COMPLETE                  written last; its absence means the checkpoint is torn

Writing ``COMPLETE`` last and validating on load is what lets :func:`latest_valid`
distinguish a finished checkpoint from one interrupted mid-write. A resumed run then
falls back to the previous good step instead of dying on a truncated file.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..model.io import load_safetensors, save_safetensors
from ..utils.hashing import sha256_json
from ..utils.logging_utils import get_logger

log = get_logger(__name__)

COMPLETE_MARKER = "COMPLETE"
OPT_FILE = "optimizer.pt"
SCHED_FILE = "scheduler.pt"
RNG_FILE = "rng.pt"
META_FILE = "metadata.json"


def git_commit(root: str | Path = ".") -> str:
    """Best-effort git revision. Returns ``"unavailable"`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True,
            text=True, timeout=10, check=False,
        )
        rev = (out.stdout or "").strip()
        return rev[:12] if rev else "unavailable"
    except Exception:
        return "unavailable"


@dataclass
class CheckpointMetadata:
    step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    tokens_per_second: float = 0.0
    wall_clock_seconds: float = 0.0
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    val_perplexity: float = float("nan")
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    tokens_per_step: int = 0
    micro_batch_size: int = 0
    grad_accum_steps: int = 0
    dataset_version: str = ""
    dataset_manifest_hash: str = ""
    tokenizer_version: str = ""
    tokenizer_sha256: str = ""
    model_config: dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    created_at: str = ""
    trainer_config: dict[str, Any] = field(default_factory=dict)
    peak_memory_bytes: int = 0
    device: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CheckpointManager:
    """Writes, validates, rotates and resumes checkpoints."""

    def __init__(
        self,
        root: str | Path,
        save_every: int = 500,
        keep_last: int = 5,
        keep_best: int = 2,
        monitor: str = "val_loss",
        monitor_mode: str = "min",
        atomic: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.atomic = atomic
        self.best_value: float | None = None
        self.best_step: int = -1
        self.written: list[Path] = []
        self.failure_budget: dict[str, int] = {
            "oom": 0, "nan_loss": 0, "corrupt_checkpoint": 0,
            "gradient_explosion": 0, "io_error": 0,
        }

    # ------------------------------------------------------------------ paths
    def dir_for(self, step: int) -> Path:
        return self.root / f"step-{step:08d}"

    def list_checkpoints(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(
            (p for p in self.root.glob("step-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[-1]),
        )

    def is_valid(self, path: Path) -> bool:
        """A checkpoint is valid iff it has weights and the COMPLETE marker."""
        return (
            path.is_dir()
            and (path / COMPLETE_MARKER).exists()
            and any(path.glob("*.safetensors"))
        )

    def latest_valid(self) -> Path | None:
        for p in reversed(self.list_checkpoints()):
            if self.is_valid(p):
                return p
            log.warning("checkpoint %s is incomplete or torn; skipping", p.name)
        return None

    def best_valid(self) -> Path | None:
        best: Path | None = None
        best_val = None
        for p in self.list_checkpoints():
            if not self.is_valid(p):
                continue
            meta = self.read_metadata(p)
            if meta is None:
                continue
            val = meta.get(self.monitor, None)
            if val is None or not np.isfinite(val):
                continue
            if best_val is None or (val < best_val if self.monitor_mode == "min" else val > best_val):
                best_val = val
                best = p
        return best

    # ------------------------------------------------------------------- read
    @staticmethod
    def read_metadata(path: Path) -> dict[str, Any] | None:
        p = Path(path) / META_FILE
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.error("metadata unreadable in %s", path)
            return None

    # ------------------------------------------------------------------ write
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any,
        metadata: CheckpointMetadata,
        scaler: Any = None,
        save_rng: bool = True,
        extra_state: dict[str, Any] | None = None,
    ) -> Path:
        """Write a checkpoint atomically (staging directory + rename + COMPLETE marker)."""
        target = self.dir_for(metadata.step)
        staging = self.root / f".staging-{metadata.step:08d}"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        try:
            save_safetensors(model, staging)

            if optimizer is not None:
                torch.save(optimizer.state_dict(), staging / OPT_FILE)
            if scheduler is not None:
                torch.save(scheduler.state_dict(), staging / SCHED_FILE)
            if scaler is not None:
                torch.save(scaler.state_dict(), staging / "scaler.pt")
            if save_rng:
                torch.save(
                    {
                        "torch": torch.get_rng_state(),
                        "numpy": np.random.get_state(),
                        "python": random.getstate(),
                    },
                    staging / RNG_FILE,
                )
            if extra_state:
                torch.save(extra_state, staging / "extra.pt")

            metadata.git_commit = metadata.git_commit or git_commit()
            metadata.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            payload = metadata.as_dict()
            payload["checkpoint_hash"] = sha256_json(payload)
            (staging / META_FILE).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

            # COMPLETE last: until this file exists the checkpoint is not resumable.
            (staging / COMPLETE_MARKER).write_text("ok", encoding="utf-8")

            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            staging.rename(target)
            self.written.append(target)
            log.info("checkpoint saved: %s (step=%d tokens=%s val_loss=%s)",
                     target.name, metadata.step, f"{metadata.tokens_seen:,}",
                     f"{metadata.val_loss:.4f}" if np.isfinite(metadata.val_loss) else "n/a")

        except Exception as exc:
            self.failure_budget["io_error"] += 1
            shutil.rmtree(staging, ignore_errors=True)
            log.error("failed to write checkpoint at step %d: %s", metadata.step, exc)
            raise
        return target

    # ------------------------------------------------------------------- load
    def load(
        self,
        model: torch.nn.Module,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        load_rng: bool = True,
        strict: bool = True,
    ) -> CheckpointMetadata | None:
        """Restore a checkpoint into ``model``/``optimizer``/``scheduler``.

        Missing or corrupt auxiliary state is tolerated with a warning rather than raised,
        so a run resumes on the strongest available signal. The *weights* must load or
        the exception propagates.
        """
        path = Path(path)
        if not self.is_valid(path):
            self.failure_budget["corrupt_checkpoint"] += 1
            raise RuntimeError(f"checkpoint {path} is not a complete, resumable checkpoint")

        load_safetensors(model, path)

        if optimizer is not None:
            f = path / OPT_FILE
            if f.exists():
                try:
                    optimizer.load_state_dict(torch.load(f, map_location="cpu", weights_only=False))
                except Exception as exc:
                    log.warning("optimizer state unrestorable (%s); continuing with fresh state", exc)
                    self.failure_budget["corrupt_checkpoint"] += 1
            else:
                log.warning("no optimizer state in %s; resuming weights only", path.name)

        if scheduler is not None and (path / SCHED_FILE).exists():
            try:
                scheduler.load_state_dict(torch.load(path / SCHED_FILE, map_location="cpu", weights_only=False))
            except Exception as exc:
                log.warning("scheduler state unrestorable: %s", exc)

        if scaler is not None and (path / "scaler.pt").exists():
            try:
                scaler.load_state_dict(torch.load(path / "scaler.pt", map_location="cpu", weights_only=False))
            except Exception as exc:
                log.warning("scaler state unrestorable: %s", exc)

        if load_rng and (path / RNG_FILE).exists():
            try:
                blob = torch.load(path / RNG_FILE, map_location="cpu", weights_only=False)
                torch.set_rng_state(blob["torch"])
                np.random.set_state(blob["numpy"])
                random.setstate(blob["python"])
            except Exception as exc:
                log.warning("RNG state unrestorable: %s", exc)

        meta = self.read_metadata(path)
        if meta is None and strict:
            raise RuntimeError(f"checkpoint {path} has no readable metadata")
        if meta is not None:
            self.best_value = meta.get(self.monitor, self.best_value)
            self.best_step = int(meta.get("step", -1))
            log.info("resumed from %s: step=%s tokens=%s",
                     path.name, meta.get("step"), meta.get("tokens_seen"))
        return meta

    # -------------------------------------------------------------- rotation
    def rotate(self) -> dict[str, Any]:
        """Delete old checkpoints beyond ``keep_last`` / ``keep_best``.

        Deleting the best-model checkpoint is explicitly avoided even when it falls outside
        the keep-last window, because the best weights are usually what a later run needs
        to continue from.
        """
        valid = [p for p in self.list_checkpoints() if self.is_valid(p)]
        metas = {p: self.read_metadata(p) or {} for p in valid}
        best_path = self.best_valid()

        keep: set[Path] = set(valid[-self.keep_last :]) if self.keep_last > 0 else set()
        if self.monitor in metas.get(best_path, {}) if best_path else False:
            keep.add(best_path)  # type: ignore[arg-type]

        # Also keep the top-N by monitor value, not just the most recent best.
        scored = [(p, m[self.monitor]) for p, m in metas.items()
                  if self.monitor in m and np.isfinite(m[self.monitor])]
        scored.sort(key=lambda kv: kv[1], reverse=(self.monitor_mode == "max"))
        for p, _ in scored[: self.keep_best]:
            keep.add(p)

        removed = []
        for p in valid:
            if p not in keep:
                shutil.rmtree(p, ignore_errors=True)
                removed.append(p.name)
        # Sweep abandoned staging directories from interrupted writes.
        for p in self.root.glob(".staging-*"):
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p.name + "(staging)")

        if removed:
            log.info("rotated %d checkpoints: %s", len(removed), ", ".join(removed[:8]))
        return {"kept": [p.name for p in sorted(keep)], "removed": removed,
                "total_before": len(valid), "total_after": len(keep)}

    # -------------------------------------------------------------- manifest
    def write_manifest(self, extra: dict[str, Any] | None = None) -> Path:
        rows = []
        for p in self.list_checkpoints():
            meta = self.read_metadata(p) or {}
            size = sum(f.stat().st_size for f in p.glob("*.safetensors"))
            rows.append(
                {
                    "name": p.name,
                    "step": meta.get("step"),
                    "tokens_seen": meta.get("tokens_seen"),
                    "train_loss": meta.get("train_loss"),
                    "val_loss": meta.get("val_loss"),
                    "val_perplexity": meta.get("val_perplexity"),
                    "tokens_per_second": meta.get("tokens_per_second"),
                    "bytes": size,
                    "complete": self.is_valid(p),
                    "checkpoint_hash": meta.get("checkpoint_hash"),
                }
            )
        payload = {"checkpoints": rows, "root": str(self.root)}
        if extra:
            payload.update(extra)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def disk_usage_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())


def verify_checkpoint(path: str | Path) -> dict[str, Any]:
    """Independently validate a checkpoint directory and report what is present."""
    p = Path(path)
    result: dict[str, Any] = {
        "path": str(p),
        "exists": p.is_dir(),
        "complete_marker": (p / COMPLETE_MARKER).exists(),
        "metadata": (p / META_FILE).exists(),
        "weights": sorted(f.name for f in p.glob("*.safetensors")),
        "has_optimizer": (p / OPT_FILE).exists(),
        "has_scheduler": (p / SCHED_FILE).exists(),
        "has_rng": (p / RNG_FILE).exists(),
        "bytes": 0,
        "errors": [],
    }
    if not result["exists"]:
        result["errors"].append("directory does not exist")
        return result
    result["bytes"] = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    if not result["complete_marker"]:
        result["errors"].append("no COMPLETE marker: write was interrupted")
    if not result["weights"]:
        result["errors"].append("no safetensors weights")
    else:
        try:
            from safetensors import safe_open

            with safe_open(str(next(p.glob("*.safetensors"))), framework="pt") as f:
                keys = list(f.keys())
            result["tensor_count"] = len(keys)
            if not keys:
                result["errors"].append("weight file contains zero tensors")
        except Exception as exc:
            result["errors"].append(f"weights unreadable: {exc}")
    meta = CheckpointManager.read_metadata(p)
    if meta is None:
        if result["metadata"]:
            result["errors"].append("metadata is not valid JSON")
    else:
        result["meta"] = {k: meta.get(k) for k in
                          ("step", "tokens_seen", "train_loss", "val_loss", "val_perplexity",
                           "dataset_version", "tokenizer_version", "git_commit", "checkpoint_hash")}
    result["valid"] = not result["errors"]
    return result


__all__ = [
    "CheckpointManager",
    "CheckpointMetadata",
    "verify_checkpoint",
    "git_commit",
    "COMPLETE_MARKER",
]
