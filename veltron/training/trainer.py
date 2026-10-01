"""Causal language-model pretraining loop.

Handles gradient accumulation, mixed precision, gradient clipping, LR scheduling,
validation, throughput accounting, checkpointing/resume and failure classification.
Runs on CPU, CUDA or DirectML through :mod:`veltron.utils.device`.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.loader import TokenShardLoader
from ..model.config import ModelConfig
from ..model.transformer import VeltronLM
from ..utils.device import empty_cache, get_device, probe
from ..utils.logging_utils import get_logger
from ..utils.seed import seed_everything
from .checkpoint import CheckpointManager, CheckpointMetadata
from .schedule import build_optimizer, build_schedule, clip_grad_norm

log = get_logger(__name__)


@dataclass
class TrainConfig:
    """Everything needed to reproduce a run, serialised into every checkpoint."""

    run_name: str = "veltronlm-run"
    output_dir: str = "checkpoints"
    dataset_root: str = "datasets/dataset-v1"
    dataset_version: str = "dataset-v1"
    model_config: str = "nano"          # registry key
    tokenizer_dir: str = "models/tokenizer"

    seq_len: int = 512
    micro_batch_size: int = 4
    grad_accum_steps: int = 1
    max_steps: int = 1000
    warmup_steps: int = 50

    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    schedule: str = "cosine"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    optimizer: str = "adamw"
    grad_clip: float = 1.0

    eval_interval: int = 200
    eval_batches: int = 20
    log_interval: int = 10
    save_interval: int = 500
    keep_last: int = 4
    keep_best: int = 2

    device: str = "auto"
    precision: str = "auto"             # auto | fp32 | fp16 | bf16
    seed: int = 1234
    deterministic: bool = False
    num_threads: int | None = None

    resume: str = "auto"                # auto | none | path
    max_hours: float = 0.0              # 0 disables the wall-clock budget
    stop_file: str = ""                 # if set, existence ends training at a step boundary
    nan_patience: int = 3               # consecutive non-finite losses tolerated
    overfit_subset: int = 0             # >0 restricts the corpus to N windows (canary runs)
    loss_chunk_tokens: int = 4096    # bounds fp32 log-softmax memory; 0 disables chunking
    run_canary_first: bool = True
    log_jsonl: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    tokens_per_step: int = 0
    train_loss: float = float("nan")
    best_val_loss: float = float("inf")
    best_val_perplexity: float = float("inf")
    wall_clock_seconds: float = 0.0
    consecutive_nonfinite: int = 0
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Trainer:
    """Single-process causal-LM trainer."""

    def __init__(
        self,
        cfg: TrainConfig,
        model: VeltronLM | None = None,
        train_loader: TokenShardLoader | None = None,
        val_loader: TokenShardLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.device_info = probe(cfg.device)
        self.device = get_device(cfg.device)
        log.info("device: %s", self.device_info.as_dict())

        seed_info = seed_everything(cfg.seed, cfg.deterministic, cfg.num_threads)
        self.seed_info = seed_info

        if model is None:
            from ..model.config import get_config

            mcfg: ModelConfig = get_config(cfg.model_config)
            log.info("building model %s (%s)", mcfg.name, cfg.model_config)
            model = VeltronLM(mcfg)
        self.model = model
        self.model_cfg = model.cfg

        if train_loader is None or val_loader is None:
            from ..data.loader import build_dataloaders

            train_loader, val_loader, _, _ = build_dataloaders(
                cfg.dataset_root, cfg.seq_len, cfg.micro_batch_size, seed=cfg.seed
            )
        self.train_loader = train_loader
        self.val_loader = val_loader

        if cfg.seq_len != train_loader.dataset.seq_len:
            log.warning(
                "config seq_len=%d but dataset was packed at seq_len=%d",
                cfg.seq_len, train_loader.dataset.seq_len,
            )

        self.tokens_per_step = cfg.micro_batch_size * cfg.grad_accum_steps * cfg.seq_len
        self.state = TrainState(tokens_per_step=self.tokens_per_step)

        self.out_dir = Path(cfg.output_dir) / cfg.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt = CheckpointManager(
            self.out_dir, save_every=cfg.save_interval,
            keep_last=cfg.keep_last, keep_best=cfg.keep_best,
        )

        self.precision, self.autocast_dtype, self.use_scaler = self._resolve_precision()
        self.model.to(self.device)
        if self.precision == "fp16_weights":
            # DirectML has no autocast kernel for aten::embedding, so fp16 is expressed by
            # storing the parameters in fp16 and letting each matmul accumulate in fp32.
            # Gradients then arrive in fp16, so the trainer keeps a fp32 master copy of the
            # parameters and casts to fp16 inside the optimizer step via _copy_grad_fp32.
            self.model.half()
            self.fp16_weights = True
        else:
            self.fp16_weights = False

        self.optimizer = build_optimizer(
            self.model, cfg.optimizer, cfg.lr, cfg.weight_decay,
            (cfg.beta1, cfg.beta2), cfg.eps,
        )
        self.scheduler = build_schedule(
            cfg.schedule, self.optimizer, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.use_scaler else None

        self.history: list[dict[str, Any]] = []
        self.history_path = self.out_dir / "train_log.jsonl"
        self._config_written = False

        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("model %s: %s parameters (%.3fB)", self.model_cfg.name, f"{n_params:,}", n_params / 1e9)
        log.info("tokens/step=%s  total target tokens=%s",
                 f"{self.tokens_per_step:,}", f"{self.tokens_per_step * cfg.max_steps:,}")

    # ---------------------------------------------------------------- precision
    def _resolve_precision(self) -> tuple[str, torch.dtype | None, bool]:
        """Choose the compute dtype.

        Returns ``(name, autocast_dtype, use_grad_scaler)``.

        ``torch.autocast`` is used **only** on CUDA. DirectML's autocast backend has no
        kernel for ``aten::embedding``, so entering an autocast region there raises
        ``NotImplementedError`` on the token embedding -- the very first op of the model.
        On DirectML, fp16 is therefore expressed by casting the model's parameters to
        fp16 directly (``model.half()``) and letting each matmul accumulate in fp32,
        which is supported; autocast itself is never entered.
        """
        p = (self.cfg.precision or "auto").lower()
        is_directml = self.device_info.kind == "directml"

        if p == "auto":
            if self.device_info.kind == "cuda":
                return "bf16" if self.device_info.supports_bf16 else "fp16", (
                    torch.bfloat16 if self.device_info.supports_bf16 else torch.float16
                ), self.device_info.kind == "cuda"
            # fp32 for both DirectML and CPU: no autocast, no GradScaler.
            return "fp32", None, False

        if p in ("fp32", "float32"):
            return "fp32", None, False
        if p in ("fp16", "float16"):
            if is_directml:
                return "fp16_weights", None, False
            return "fp16", torch.float16, self.device_info.kind == "cuda"
        if p in ("bf16", "bfloat16"):
            if is_directml:
                log.warning("bf16 requested; DirectML has no native bf16 matmul, using fp16 weights")
                return "fp16_weights", None, False
            if not self.device_info.supports_bf16:
                log.warning("bf16 requested but %s lacks native bf16; falling back to fp32",
                            self.device_info.kind)
                return "fp32", None, False
            return "bf16", torch.bfloat16, False
        raise ValueError(f"unknown precision {self.cfg.precision!r}")

    def _autocast_device(self) -> str:
        """Autocast only ever runs on CUDA; see :meth:`_resolve_precision`."""
        return "cuda"

    # -------------------------------------------------------------------- logs
    def _write_config(self) -> None:
        if self._config_written:
            return
        from ..tokenizer.trainer import VeltronTokenizer

        tok_meta: dict[str, Any] = {}
        tok_path = Path(self.cfg.tokenizer_dir)
        if (tok_path / "tokenizer_meta.json").exists():
            tok_meta = json.loads((tok_path / "tokenizer_meta.json").read_text(encoding="utf-8"))
        else:
            try:
                vt = VeltronTokenizer.load(tok_path)
                tok_meta = {"vocab_size": vt.get_vocab_size(), "sha256": vt.sha256, "name": vt.name}
            except Exception as exc:
                log.warning("could not read tokenizer metadata: %s", exc)

        manifest_path = Path(self.cfg.dataset_root) / "manifest.json"
        manifest_hash = ""
        if manifest_path.exists():
            manifest_hash = json.loads(manifest_path.read_text(encoding="utf-8")).get("manifest_hash", "")

        payload = {
            "train_config": self.cfg.as_dict(),
            "model_config": self.model_cfg.to_dict(),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": self.device_info.as_dict(),
            "precision": self.precision,
            "tokens_per_step": self.tokens_per_step,
            "target_tokens": self.tokens_per_step * self.cfg.max_steps,
            "tokenizer": tok_meta,
            "dataset_manifest_hash": manifest_hash,
            "seed_info": self.seed_info,
            "library_versions": _library_versions(),
        }
        (self.out_dir / "run_config.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        self._config_written = True

    def _log_event(self, event: dict[str, Any]) -> None:
        self.history.append(event)
        if self.cfg.log_jsonl:
            with self.history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        if "loss" in event:
            log.info(
                "step %5d/%d | loss %.4f | lr %.3e | tok/s %s | %s",
                event.get("step", 0), self.cfg.max_steps, event["loss"],
                event.get("lr", 0.0), f"{event.get('tokens_per_second', 0):.0f}"
                if event.get("tokens_per_second") else "n/a",
                event.get("note", ""),
            )

    # ---------------------------------------------------------------- forward
    def _forward_backward(self, batch: dict[str, torch.Tensor]) -> float:
        """One micro-batch forward+backward. Returns the loss (pre-division by accum)."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        if self.autocast_dtype is not None and self.scaler is not None:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                out = self.model(input_ids, labels=labels, loss_chunk_tokens=self.cfg.loss_chunk_tokens)
                loss = out.loss / self.cfg.grad_accum_steps
            self.scaler.scale(loss).backward()
            return float(out.loss.detach())

        if self.autocast_dtype is not None:
            with torch.autocast(device_type=self._autocast_device(), dtype=self.autocast_dtype):
                out = self.model(input_ids, labels=labels, loss_chunk_tokens=self.cfg.loss_chunk_tokens)
                loss = out.loss / self.cfg.grad_accum_steps
            loss.backward()
            return float(out.loss.detach())

        out = self.model(input_ids, labels=labels, loss_chunk_tokens=self.cfg.loss_chunk_tokens)
        loss = out.loss / self.cfg.grad_accum_steps
        loss.backward()
        return float(out.loss.detach())

    def _optimizer_step(self) -> dict[str, float]:
        gn = clip_grad_norm(self.model, self.cfg.grad_clip)
        finite, scale = self._unscale_and_check()
        if not finite:
            self.optimizer.zero_grad(set_to_none=True)
            self.state.consecutive_nonfinite += 1
            self.ckpt.failure_budget["nan_loss"] += 1
            if gn > 50 * max(self.cfg.grad_clip, 1e-6):
                self.ckpt.failure_budget["gradient_explosion"] += 1
            return {"grad_norm": gn, "skipped": 1.0}

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.consecutive_nonfinite = 0
        return {"grad_norm": gn, "skipped": 0.0}

    def _unscale_and_check(self) -> tuple[bool, float]:
        """Unscale (if a GradScaler is active) and verify gradients are finite.

        The check reduces with ``vector_norm`` rather than materialising
        ``torch.isfinite(grad).all()``. The obvious spelling allocates a full-size boolean
        copy of every gradient before reducing it -- for a 244M-parameter model that is
        several hundred megabytes of transient allocation, which fails outright on a nearly
        full DirectML heap. A norm is a fused reduction: NaN and Inf both propagate into
        the scalar result, so the test is exact while allocating O(1).
        """
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        for p in self.model.parameters():
            if p.grad is None:
                continue
            n = torch.linalg.vector_norm(p.grad.detach())
            if not bool(torch.isfinite(n)):
                return False, 1.0
        return True, 1.0

    # -------------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> dict[str, float]:
        """Validation loss and perplexity. Perplexity is reported but not optimised."""
        self.model.eval()
        n = max_batches if max_batches is not None else self.cfg.eval_batches
        total_loss = 0.0
        total_tokens = 0
        for i, batch in enumerate(self.val_loader):
            if n and i >= n:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            if self.autocast_dtype is not None:
                with torch.autocast(device_type=self._autocast_device(),
                                    dtype=self.autocast_dtype):
                    out = self.model(input_ids, labels=labels, loss_chunk_tokens=self.cfg.loss_chunk_tokens)
            else:
                out = self.model(input_ids, labels=labels,
                                 loss_chunk_tokens=self.cfg.loss_chunk_tokens)
            # Mean over tokens in this batch, so batches of unequal token counts average
            # correctly instead of being weighted by batch count.
            ntok = int((labels[:, 1:] != -100).sum())
            total_loss += float(out.loss.detach()) * ntok
            total_tokens += ntok
        self.model.train()
        if total_tokens == 0:
            return {"loss": float("nan"), "perplexity": float("nan"), "tokens": 0}
        mean_loss = total_loss / total_tokens
        return {
            "loss": mean_loss,
            # Computed in log space: exp() of a large loss overflows to inf, which would
            # hide the real value.
            "perplexity": float(math.exp(min(mean_loss, 20.0))),
            "tokens": total_tokens,
        }

    # ------------------------------------------------------------------- loop
    def _maybe_resume(self) -> None:
        mode = (self.cfg.resume or "auto").lower()
        if mode == "none":
            log.info("resume disabled by config")
            return
        path: Path | None = None
        if mode == "auto":
            path = self.ckpt.latest_valid()
        else:
            candidate = Path(mode)
            if not candidate.exists():
                candidate = self.out_dir / mode
            if candidate.exists():
                path = candidate
        if path is None:
            log.info("no checkpoint to resume from; starting fresh")
            return

        meta = self.ckpt.load(
            self.model, path, self.optimizer, self.scheduler, self.scaler, load_rng=True
        )
        if meta:
            self.state.step = int(meta.get("step", 0))
            self.state.epoch = int(meta.get("epoch", 0))
            self.state.tokens_seen = int(meta.get("tokens_seen", 0))
            self.state.wall_clock_seconds = float(meta.get("wall_clock_seconds", 0.0))
            self.state.best_val_loss = float(meta.get("val_loss") or float("inf"))
            self.state.best_val_perplexity = float(meta.get("val_perplexity") or float("inf"))
            # The scheduler must be told the true step so its LR curve is continuous.
            self.scheduler.step_count = self.state.step
            self._set_lr(self.scheduler._lr_at(self.state.step))
            log.info("resumed at step %d / %d", self.state.step, self.cfg.max_steps)

    def _set_lr(self, scale: float) -> None:
        for group, base in zip(self.optimizer.param_groups, self.scheduler.base_lrs):
            group["lr"] = base * scale

    def _wall_clock_budget_exhausted(self) -> bool:
        if self.cfg.max_hours <= 0:
            return False
        elapsed = time.time() - self.state.started_at
        return elapsed >= self.cfg.max_hours * 3600

    def _stop_requested(self) -> bool:
        """True when a stop file has been created.

        Windows has no clean way to signal a detached process: ``Stop-Process`` is an
        immediate termination with no chance to finish the in-flight step, and Ctrl+C is
        unavailable to a process launched detached. A file is the only cooperative signal
        that survives process boundaries, and polling it costs one ``stat`` per step
        (~20 s here).

        Polled only when ``stop_file`` is configured, so the default path is unchanged and
        existing runs and checkpoints are unaffected.
        """
        if not self.cfg.stop_file:
            return False
        return Path(self.cfg.stop_file).exists()

    def train(self) -> dict[str, Any]:
        """Run the training loop to ``max_steps`` or until the wall-clock budget ends."""
        self._write_config()
        self._maybe_resume()
        self.model.train()

        if self.cfg.run_canary_first and self.state.step == 0:
            self._canary()

        log.info(
            "training start: %d steps, %d tokens/step, target %s tokens, precision=%s",
            self.cfg.max_steps, self.tokens_per_step,
            f"{self.tokens_per_step * self.cfg.max_steps:,}", self.precision,
        )

        self.state.started_at = time.time()
        batch_iter: Iterator[dict[str, torch.Tensor]] = iter(self.train_loader)
        running: list[float] = []
        step_start = time.time()
        tokens_since = 0
        stop_reason = "max_steps"

        while self.state.step < self.cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []

            for _ in range(self.cfg.grad_accum_steps):
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    self.state.epoch += 1
                    self.train_loader.set_epoch(self.state.epoch)
                    batch_iter = iter(self.train_loader)
                    log.info("epoch %d: reshuffled loader", self.state.epoch)
                    batch = next(batch_iter)

                try:
                    loss = self._forward_backward(batch)
                except RuntimeError as exc:
                    # OOM is recoverable-ish: checkpoint, then re-raise with the budget
                    # updated. A non-OOM RuntimeError is a real bug and must not be
                    # silently swallowed, so it propagates after the safety checkpoint.
                    if "out of memory" in str(exc).lower():
                        self.ckpt.failure_budget["oom"] += 1
                        self._emergency_checkpoint()
                    raise
                except MemoryError:
                    self.ckpt.failure_budget["oom"] += 1
                    self._emergency_checkpoint()
                    raise
                micro_losses.append(loss)
                tokens_since += batch["input_ids"].numel()

            info = self._optimizer_step()
            lr = self.scheduler.step()
            self.state.step += 1
            self.state.tokens_seen += self.tokens_per_step
            if info["skipped"]:
                continue

            mean_loss = float(np.mean(micro_losses)) if micro_losses else float("nan")
            running.append(mean_loss)
            self.state.train_loss = mean_loss

            dt = time.time() - step_start
            tps = tokens_since / dt if dt > 0 else 0.0

            if self.state.step % self.cfg.log_interval == 0:
                self._log_event(
                    {
                        "step": self.state.step,
                        "loss": mean_loss,
                        "loss_ema": float(np.mean(running[-50:])),
                        "lr": lr,
                        "grad_norm": info["grad_norm"],
                        "tokens_seen": self.state.tokens_seen,
                        "tokens_per_step": self.tokens_per_step,
                        "tokens_per_second": tps,
                        "seconds_per_step": dt / self.cfg.grad_accum_steps,
                        "precision": self.precision,
                        "epoch": self.state.epoch,
                    }
                )
                step_start = time.time()
                tokens_since = 0

            if self.cfg.eval_interval > 0 and self.state.step % self.cfg.eval_interval == 0:
                ev = self.evaluate()
                self.state.best_val_loss = min(self.state.best_val_loss, ev["loss"])
                self.state.best_val_perplexity = min(self.state.best_val_perplexity, ev["perplexity"])
                self._log_event(
                    {"step": self.state.step, "eval": True, "val_loss": ev["loss"],
                     "val_perplexity": ev["perplexity"], "val_tokens": ev["tokens"],
                     "train_loss": mean_loss, "lr": lr}
                )

            if self.cfg.save_interval > 0 and self.state.step % self.cfg.save_interval == 0:
                # The interval save happens outside the eval block on most steps, so
                # evaluate here rather than reusing the training loss. Storing the train
                # loss in `val_loss` makes the best-checkpoint selection compare
                # train-vs-train and silently pick an overfit checkpoint.
                sv = self.evaluate() if self.cfg.eval_interval > 0 else None
                self._save_checkpoint(
                    val_loss=(sv or {}).get("loss", float("nan")),
                    tps=tps,
                    train_loss=mean_loss,
                    val_ppl=(sv or {}).get("perplexity", float("nan")),
                )

            if self._wall_clock_budget_exhausted():
                stop_reason = "wall_clock_budget"
                log.info("wall-clock budget of %.2fh exhausted at step %d",
                         self.cfg.max_hours, self.state.step)
                break

            if self._stop_requested():
                # Checked here, deliberately: after the optimizer step, after the
                # scheduled evaluation, and after the scheduled checkpoint write. Breaking
                # at this point means the step that was in flight has completed and any
                # checkpoint that was due has been written, then the loop falls through to
                # the final evaluation, final checkpoint and summary.
                stop_reason = "stop_file_requested"
                log.info("stop file %s detected; stopping cleanly at step %d",
                         self.cfg.stop_file, self.state.step)
                break

            if self.state.consecutive_nonfinite > self.cfg.nan_patience:
                stop_reason = "repeated_nonfinite_loss"
                log.error("aborting: %d consecutive non-finite losses", self.state.consecutive_nonfinite)
                break

        final_ev = self.evaluate(max_batches=max(self.cfg.eval_batches, 10))
        self.state.best_val_loss = min(self.state.best_val_loss, final_ev["loss"])
        self.state.best_val_perplexity = min(self.state.best_val_perplexity, final_ev["perplexity"])
        self._save_checkpoint(
            val_loss=final_ev["loss"], tps=0.0, tag="final",
            train_loss=self.state.train_loss, val_ppl=final_ev["perplexity"],
        )
        self.ckpt.rotate()
        self.ckpt.write_manifest()

        summary = {
            "run_name": self.cfg.run_name,
            "model": self.model_cfg.name,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "steps_completed": self.state.step,
            "tokens_seen": self.state.tokens_seen,
            "final_train_loss": self.state.train_loss,
            "final_val_loss": final_ev["loss"],
            "final_val_perplexity": final_ev["perplexity"],
            "best_val_loss": self.state.best_val_loss,
            "best_val_perplexity": self.state.best_val_perplexity,
            "wall_clock_seconds": round(time.time() - self.state.started_at, 1),
            "stop_reason": stop_reason,
            "precision": self.precision,
            "device": self.device_info.as_dict(),
            "failure_budget": self.ckpt.failure_budget,
            "output_dir": str(self.out_dir),
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        log.info("TRAINING COMPLETE: %s", json.dumps({k: v for k, v in summary.items()
                                                      if k not in ("device",)}, default=str))
        return summary

    # ------------------------------------------------------------ checkpoints
    def _metadata(self, val_loss: float, tps: float, val_ppl: float = float("nan"),
                  train_loss: float = float("nan")) -> CheckpointMetadata:
        manifest_path = Path(self.cfg.dataset_root) / "manifest.json"
        manifest_hash = ""
        dataset_version = self.cfg.dataset_version
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_hash = m.get("manifest_hash", "")
            dataset_version = m.get("dataset_version", dataset_version)

        tok_meta_path = Path(self.cfg.tokenizer_dir) / "tokenizer_meta.json"
        tok_version, tok_sha = self.cfg.model_config, ""
        if tok_meta_path.exists():
            tm = json.loads(tok_meta_path.read_text(encoding="utf-8"))
            tok_version = tm.get("name", tok_version)
            tok_sha = tm.get("sha256", "")

        peak = 0
        if self.device.type == "cuda":
            peak = int(torch.cuda.max_memory_allocated())

        return CheckpointMetadata(
            step=self.state.step,
            epoch=self.state.epoch,
            tokens_seen=self.state.tokens_seen,
            tokens_per_second=tps,
            wall_clock_seconds=round(time.time() - self.state.started_at, 1),
            train_loss=(train_loss if train_loss == train_loss else self.state.train_loss),
            val_loss=val_loss,
            val_perplexity=val_ppl,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            tokens_per_step=self.tokens_per_step,
            micro_batch_size=self.cfg.micro_batch_size,
            grad_accum_steps=self.cfg.grad_accum_steps,
            dataset_version=dataset_version,
            dataset_manifest_hash=manifest_hash,
            tokenizer_version=tok_version,
            tokenizer_sha256=tok_sha,
            model_config=self.model_cfg.to_dict(),
            trainer_config=self.cfg.as_dict(),
            peak_memory_bytes=peak,
            device=self.device_info.kind,
        )

    def _save_checkpoint(
        self,
        val_loss: float,
        tps: float,
        tag: str | None = None,
        train_loss: float = float("nan"),
        val_ppl: float = float("nan"),
    ) -> None:
        """Persist a checkpoint.

        ``train_loss`` is carried separately from ``val_loss`` because an interval save
        may land on a step where no evaluation ran. Storing the training loss in the
        ``val_loss`` field would make best-checkpoint selection compare train against
        train and silently prefer an overfit checkpoint.
        """
        self.ckpt.save(
            self.model, self.optimizer, self.scheduler,
            self._metadata(val_loss, tps, val_ppl, train_loss), scaler=self.scaler,
        )
        self.ckpt.rotate()

    def _emergency_checkpoint(self) -> None:
        """Save without rotating, so an OOM crash never destroys the previous good state."""
        try:
            log.warning("writing emergency checkpoint at step %d", self.state.step)
            self.ckpt.save(
                self.model, self.optimizer, self.scheduler,
                self._metadata(float("nan"), 0.0), scaler=self.scaler,
            )
            log.info("emergency checkpoint written to %s", self.ckpt.dir_for(self.state.step))
        except Exception as exc:
            log.error("emergency checkpoint failed: %s", exc)

# ----------------------------------------------------------------- canary
    def _canary(self) -> None:
        """Overfit a tiny slice before spending real compute.

        A model that cannot drive the loss on a few hundred tokens to near-zero has a
        wiring bug (wrong label shift, detached activations, corrupted shard), and every
        subsequent step would be wasted.

        The canary runs in a **subprocess**. On DirectML the allocator grows its heap and
        never returns it, so an in-process canary permanently reserves several gigabytes
        and the real run then OOMs inside ``cross_entropy`` a few steps in. That produced
        two consecutive false diagnoses of this project -- `mini` at b=2/seq=512/accum=32
        looked like a memory-budget failure, but ``scripts/stress_accum.py`` showed the
        identical configuration surviving 40 micro-steps on its own. A subprocess is the
        only reliable way to reclaim the heap on this backend.
        """
        n = min(32, len(self.train_loader.dataset))
        canary_len = min(self.train_loader.dataset.seq_len, 128)
        batch = 1
        while batch < 8:
            if batch * canary_len * self.model_cfg.vocab_size * 4 > 96 * 1024**2:
                break
            batch += 1

        out_path = self.out_dir / "canary.json"
        cmd = [
            sys.executable, "-m", "veltron.training.canary",
            "--tier", self.cfg.model_config,
            "--seq-len", str(self.cfg.seq_len),
            "--batch-size", str(batch),
            "--window-len", str(canary_len),
            "--windows", str(n),
            "--steps", "200",
            "--loss-chunk", str(self.cfg.loss_chunk_tokens),
            "--device", self.device_info.kind,
            "--out", str(out_path),
        ]
        log.info("CANARY: overfitting %d windows of %d tokens (subprocess, %s)",
                 n, canary_len, " ".join(cmd[2:]))
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            raise RuntimeError("canary timed out after 30 minutes; refusing to train")

        if proc.returncode != 0:
            log.error("canary subprocess failed:\n%s", (proc.stderr or "")[-800:])
            raise RuntimeError(
                "CANARY FAILED (subprocess): the pipeline did not converge on a memorisation "
                "task. Refusing to spend hours on a broken run."
            )
        try:
            result = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"canary produced no readable result: {exc}")

        result["seconds"] = round(time.time() - t0, 1)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        self._log_event({"note": "canary", **result})
        if not result.get("passed"):
            raise RuntimeError(
                f"CANARY FAILED: loss {result.get('initial_loss'):.4f} -> "
                f"{result.get('best_loss'):.4f}. The architecture or data pipeline has a bug."
            )
        log.info("CANARY PASSED: loss %.4f -> %.4f (%.1fx reduction, %.0fs, subprocess)",
                 result["initial_loss"], result["best_loss"],
                 result["reduction_factor"], result["seconds"])


def _library_versions() -> dict[str, str]:
    import platform
    import sys

    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    for mod in ("numpy", "tokenizers", "safetensors"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "absent"
    try:
        import torch_directml  # type: ignore[import-not-found]

        out["torch_directml"] = getattr(torch_directml, "__version__", "installed")
    except Exception:
        out["torch_directml"] = "absent"
    return out


__all__ = ["Trainer", "TrainConfig", "TrainState"]
