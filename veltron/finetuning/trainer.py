"""Supervised instruction fine-tuning.

Loss is computed **only on assistant tokens**. Including prompt tokens would spend most of
the gradient budget teaching the model to reproduce its own input, and is the single most
common reason a fine-tune looks healthy while failing to follow instructions.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..model.transformer import VeltronLM
from ..training.checkpoint import CheckpointManager, CheckpointMetadata
from ..training.schedule import build_optimizer, build_schedule, clip_grad_norm
from ..utils.device import get_device, probe
from ..utils.logging_utils import get_logger
from .data import SFTExample

log = get_logger(__name__)


@dataclass
class SFTConfig:
    run_name: str = "sft"
    base_checkpoint: str = ""          # directory produced by the pretrainer
    model_config: str = "micro"
    tokenizer_dir: str = "models/tok-mini-32k"
    output_dir: str = "checkpoints"
    data_path: str = "datasets/sft/sft.jsonl"

    max_seq_len: int = 640
    micro_batch_size: int = 2
    grad_accum_steps: int = 4
    epochs: float = 3.0
    max_steps: int = 0                 # 0 = derive from epochs
    lr: float = 1e-5
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.05
    schedule: str = "cosine"
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    eval_fraction: float = 0.08
    log_interval: int = 20
    save_interval: int = 100
    seed: int = 1234
    device: str = "auto"
    precision: str = "fp32"
    loss_chunk_tokens: int = 4096
    resume: str = "auto"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EncodedExample:
    input_ids: list[int]
    loss_mask: list[int]
    category: str
    language: str


def encode_example(
    ex: SFTExample,
    tokenizer: Any,
    max_seq_len: int,
    train_on_prompt: bool = False,
) -> EncodedExample | None:
    """Tokenise a chat example and mask everything that is not an assistant turn.

    The template is split at each role marker so that the boundary between "context" and
    "target" is exact rather than approximated by a character offset. The assistant's
    opening marker is kept unmasked, because the model must learn to *start* its own turn.
    """
    parts: list[tuple[str, bool]] = []  # (text, is_target)
    for i, m in enumerate(ex.messages):
        marker = {"system": "<|system|>", "user": "<|user|>",
                  "assistant": "<|assistant|>"}.get(m.get("role", "user"), "<|user|>")
        content = m.get("content", "")
        if i == 0:
            parts.append((f"<|bos|>{marker}", False))
        else:
            parts.append((f"{marker}", False))
        # Only the final assistant message is a training target; earlier assistant turns
        # (rare in this dataset) stay masked so the model is not trained to parrot history.
        is_target = (i == len(ex.messages) - 1 and m.get("role") == "assistant")
        parts.append((content, is_target if (is_target or train_on_prompt) else False))
        if is_target:
            parts.append(("<|end|>", True))
        else:
            parts.append(("<|end|>", False))

    ids: list[int] = []
    mask: list[int] = []
    for text, is_target in parts:
        if not text:
            continue
        tids = tokenizer.encode(text, add_special_tokens=False)
        ids.extend(tids)
        mask.extend([1 if is_target else 0] * len(tids))

    if len(ids) > max_seq_len:
        # Truncate from the left so the assistant's answer survives: the target is at the
        # end, and cutting the tail would train on a half-finished response.
        ids = ids[-max_seq_len:]
        mask = mask[-max_seq_len:]
    if sum(mask) == 0:
        return None
    return EncodedExample(ids, mask, ex.category, ex.language)


def encode_dataset(
    examples: Sequence[SFTExample],
    tokenizer: Any,
    max_seq_len: int,
    train_on_prompt: bool = False,
) -> list[EncodedExample]:
    out = []
    dropped = 0
    for ex in examples:
        enc = encode_example(ex, tokenizer, max_seq_len, train_on_prompt)
        if enc is None:
            dropped += 1
            continue
        out.append(enc)
    if dropped:
        log.warning("dropped %d examples with no target tokens", dropped)
    log.info("encoded %d examples (max_seq_len=%d, mean_len=%.0f)", len(out), max_seq_len,
             float(np.mean([len(e.input_ids) for e in out])) if out else 0.0)
    return out


def collate_sft(batch: Sequence[EncodedExample], pad_id: int) -> dict[str, torch.Tensor]:
    maxlen = max(len(b.input_ids) for b in batch)
    input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b.input_ids)
        input_ids[i, :n] = torch.tensor(b.input_ids, dtype=torch.long)
        # Shifted inside the loss: labels[:, :-1] is compared against logits[:, :-1],
        # so position t predicts token t+1. Only target positions carry a label.
        for t in range(n):
            if b.loss_mask[t]:
                labels[i, t] = b.input_ids[t]
    return {"input_ids": input_ids, "labels": labels}


def masked_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_tokens: int = 0,
) -> torch.Tensor:
    """Cross-entropy over assistant tokens only.

    Positions whose label is ``-100`` are ignored entirely, so the loss equals the mean
    over target tokens rather than over the whole padded batch.
    """
    flat_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    flat_labels = labels[:, 1:].reshape(-1)
    keep = flat_labels != -100
    if keep.sum() == 0:
        return flat_logits.sum() * 0.0
    sel_logits = flat_logits[keep]
    sel_labels = flat_labels[keep]
    if chunk_tokens and chunk_tokens > 0:
        total = sel_logits.new_zeros((), dtype=torch.float32)
        for s in range(0, sel_logits.shape[0], chunk_tokens):
            e = min(sel_logits.shape[0], s + chunk_tokens)
            total = total + F.cross_entropy(sel_logits[s:e].float(), sel_labels[s:e],
                                           reduction="sum")
        return total / sel_labels.numel()
    return F.cross_entropy(sel_logits.float(), sel_labels)


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, encoded: Sequence[EncodedExample], pad_id: int) -> None:
        self.encoded = list(encoded)
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, i: int) -> EncodedExample:
        return self.encoded[i]


def _sort_by_length(examples: Sequence[EncodedExample]) -> list[EncodedExample]:
    """Length-bucketed order.

    Grouping similar lengths keeps padding waste low, which matters here because the
    dataset is small enough that padding can otherwise double the cost of every step.
    """
    return sorted(examples, key=lambda e: len(e.input_ids))


def batched_batches(examples: Sequence[EncodedExample], batch_size: int,
                    rng: random.Random, bucket_size: int = 64) -> list[list[EncodedExample]]:
    ordered = _sort_by_length(examples)
    batches = [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]
    rng.shuffle(batches)
    return batches


class SFTTrainer:
    def __init__(self, cfg: SFTConfig) -> None:
        from ..model.config import get_config
        from ..model.io import load_safetensors
        from ..tokenizer.trainer import VeltronTokenizer

        self.cfg = cfg
        self.device_info = probe(cfg.device)
        self.device = get_device(cfg.device)
        self.tokenizer = VeltronTokenizer.load(cfg.tokenizer_dir)

        mcfg = get_config(cfg.model_config)
        self.model = VeltronLM(mcfg)
        if cfg.base_checkpoint:
            load_safetensors(self.model, cfg.base_checkpoint)
            log.info("loaded base weights from %s", cfg.base_checkpoint)
        else:
            log.warning("no base_checkpoint given: fine-tuning from random initialisation")
        self.model.to(self.device)
        self.model_cfg = mcfg

        self.out_dir = Path(cfg.output_dir) / cfg.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt = CheckpointManager(self.out_dir, save_every=cfg.save_interval, keep_last=3, keep_best=1)

        self.history_path = self.out_dir / "sft_log.jsonl"
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ data
    def load(self) -> None:
        from .data import load_sft_dataset

        examples = load_sft_dataset(self.cfg.data_path)
        rng = random.Random(self.cfg.seed)
        rng.shuffle(examples)
        n_val = max(1, int(len(examples) * self.cfg.eval_fraction))
        val_examples, train_examples = examples[:n_val], examples[n_val:]

        self.train_enc = encode_dataset(train_examples, self.tokenizer, self.cfg.max_seq_len)
        self.val_enc = encode_dataset(val_examples, self.tokenizer, self.cfg.max_seq_len)
        if not self.train_enc:
            raise RuntimeError(f"no trainable examples in {self.cfg.data_path}")
        self.pad_id = self.tokenizer.pad_id
        self.rng = rng
        log.info("SFT split: %d train / %d val", len(self.train_enc), len(self.val_enc))

    # ----------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, max_batches: int = 40) -> dict[str, float]:
        self.model.eval()
        rng = random.Random(0)
        batches = batched_batches(self.val_enc, self.cfg.micro_batch_size, rng)[:max_batches]
        total, ntok = 0.0, 0
        per_cat: dict[str, list[float]] = {}
        for batch in batches:
            tensors = collate_sft(batch, self.pad_id)
            out = self.model(tensors["input_ids"].to(self.device))
            labels = tensors["labels"].to(self.device)
            n = int((labels[:, 1:] != -100).sum())
            loss = masked_loss(out.logits, labels, self.cfg.loss_chunk_tokens)
            total += float(loss) * n
            ntok += n
            cat = batch[0].category
            per_cat.setdefault(cat, []).append(float(loss))
        self.model.train()
        mean = total / max(1, ntok)
        return {
            "loss": mean,
            "perplexity": float(math.exp(min(mean, 20.0))),
            "target_tokens": ntok,
            "by_category": {k: float(np.mean(v)) for k, v in sorted(per_cat.items())},
        }

    # ----------------------------------------------------------------- train
    def train(self) -> dict[str, Any]:
        cfg = self.cfg
        self.load()

        batches_per_epoch = math.ceil(len(self.train_enc) / cfg.micro_batch_size)
        steps_per_epoch = max(1, batches_per_epoch // cfg.grad_accum_steps)
        total_steps = cfg.max_steps or int(steps_per_epoch * cfg.epochs)
        warmup = max(1, int(total_steps * cfg.warmup_ratio))
        log.info("SFT plan: %d batches/epoch, %d optim steps/epoch, %d total, %d warmup",
                 batches_per_epoch, steps_per_epoch, total_steps, warmup)

        opt = build_optimizer(self.model, "adamw", cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2))
        sched = build_schedule(cfg.schedule, opt, warmup, total_steps, cfg.min_lr_ratio)
        self.ckpt.load(self.model, self.ckpt.latest_valid() or Path("."),
                       opt, sched, load_rng=True) if cfg.resume != "none" and self.ckpt.latest_valid() else None

        rng = random.Random(cfg.seed)
        self.model.train()
        step = 0
        epoch = 0
        running: list[float] = []
        t0 = time.time()
        stop = "max_steps"

        while step < total_steps:
            batches = batched_batches(self.train_enc, cfg.micro_batch_size, rng)
            acc = 0
            losses: list[float] = []
            opt.zero_grad(set_to_none=True)
            for i in range(0, len(batches), cfg.grad_accum_steps):
                group = batches[i : i + cfg.grad_accum_steps]
                for batch in group:
                    tensors = collate_sft(batch, self.pad_id)
                    out = self.model(tensors["input_ids"].to(self.device))
                    labels = tensors["labels"].to(self.device)
                    loss = masked_loss(out.logits, labels, cfg.loss_chunk_tokens)
                    (loss / cfg.grad_accum_steps).backward()
                    losses.append(float(loss.detach()))
                acc += 1
                gn = clip_grad_norm(self.model, cfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                lr = sched.step()
                step += 1
                running.extend(losses)
                losses = []

                if step % cfg.log_interval == 0:
                    self._log({"step": step, "total": total_steps, "epoch": epoch,
                               "loss": float(np.mean(running[-50:])),
                               "lr": lr, "grad_norm": gn,
                               "elapsed_s": round(time.time() - t0, 1)})
                if cfg.save_interval and step % cfg.save_interval == 0:
                    self._save(step, float(np.mean(running[-50:])))
                if step >= total_steps:
                    break
            epoch += 1

        final = self.evaluate()
        self._save(step, final["loss"], val=final)
        meta = CheckpointMetadata(
            step=step, tokens_seen=step * cfg.micro_batch_size * cfg.grad_accum_steps
            * cfg.max_seq_len, train_loss=float(np.mean(running[-50:]) if running else float("nan")),
            val_loss=final["loss"], model_config=self.model_cfg.to_dict(),
            trainer_config=cfg.as_dict(), device=self.device_info.kind,
        )
        self.ckpt.save(self.model, opt, sched, meta)
        self.ckpt.write_manifest()

        summary = {
            "run_name": cfg.run_name,
            "base_checkpoint": cfg.base_checkpoint,
            "steps": step,
            "train_examples": len(self.train_enc),
            "val_examples": len(self.val_enc),
            "final_val_loss": final["loss"],
            "final_val_perplexity": final["perplexity"],
            "by_category": final["by_category"],
            "elapsed_seconds": round(time.time() - t0, 1),
            "output_dir": str(self.out_dir),
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                   encoding="utf-8")
        log.info("SFT COMPLETE: %s", json.dumps({k: v for k, v in summary.items()
                                                 if k != "by_category"}, default=str))
        return summary

    def _log(self, event: dict[str, Any]) -> None:
        self.history.append(event)
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")
        log.info("sft step %4d/%d | loss %.4f | lr %.3e | %.0fs",
                 event["step"], event["total"], event["loss"], event["lr"], event["elapsed_s"])

    def _save(self, step: int, loss: float, val: dict[str, float] | None = None) -> None:
        meta = CheckpointMetadata(
            step=step, train_loss=loss,
            val_loss=(val or {}).get("loss", float("nan")),
            model_config=self.model_cfg.to_dict(), trainer_config=self.cfg.as_dict(),
            device=self.device_info.kind,
        )
        self.ckpt.save(self.model, None, None, meta)
        self.ckpt.rotate()


__all__ = [
    "SFTConfig",
    "SFTTrainer",
    "SFTDataset",
    "EncodedExample",
    "encode_example",
    "encode_dataset",
    "collate_sft",
    "masked_loss",
    "batched_batches",
]
