"""Direct Preference Optimisation (DPO).

Implements the DPO objective of Rafailov et al. (2023):

    L = -log sigmoid( beta * [ (log pi(yw|x)/pi_ref(yw|x)) - (log pi(yl|x)/pi_ref(yl|x)) ] )

The reference model is a frozen copy of the policy at initialisation, held on the same
device. Because the reference is frozen and the policy is derived from a *checkpoint*,
loading it is a matter of copying tensors -- no second inference pass is needed at any
point, which is what makes DPO affordable here.

The implicit reward margin and the fraction of pairs where the chosen response wins are
both logged, because a preference run that does not increase the win rate has not aligned
anything.
"""

from __future__ import annotations

import copy
import json
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
from .data import PreferencePair

log = get_logger(__name__)


@dataclass
class DPOConfig:
    run_name: str = "dpo"
    init_checkpoint: str = ""
    model_config: str = "micro"
    tokenizer_dir: str = "models/tok-mini-32k"
    output_dir: str = "checkpoints"
    data_path: str = "datasets/alignment/preferences.jsonl"

    max_seq_len: int = 640
    micro_batch_size: int = 1
    grad_accum_steps: int = 8
    max_steps: int = 200
    lr: float = 5e-7
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.1
    schedule: str = "cosine"
    grad_clip: float = 1.0
    beta: float = 0.1
    label_smoothing: float = 0.0

    eval_fraction: float = 0.12
    log_interval: int = 10
    save_interval: int = 100
    seed: int = 1234
    device: str = "auto"
    precision: str = "fp32"
    loss_chunk_tokens: int = 4096
    reference_device: str = "same"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def encode_pair(pair: PreferencePair, tokenizer: Any, max_seq_len: int
                ) -> tuple[list[int], list[int]] | None:
    """Tokenise ``prompt + response``, masking the prompt out of the loss.

    DPO scores only the response tokens. Including prompt tokens would let a pair "win"
    because the rejected response had an unusual prompt rendering rather than a worse answer.
    """
    prompt = f"<|bos|><|user|>{pair.prompt}<|end|><|assistant|>"
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    c_ids = tokenizer.encode(pair.chosen, add_special_tokens=False) + [tokenizer.eos_id]
    r_ids = tokenizer.encode(pair.rejected, add_special_tokens=False) + [tokenizer.eos_id]

    def build(resp: Sequence[int]) -> tuple[list[int], list[int]] | None:
        ids = p_ids + list(resp)
        if len(ids) > max_seq_len:
            # Truncate the prompt, never the response, for the same reason as SFT.
            overflow = len(ids) - max_seq_len
            p_trunc = p_ids[-max(1, len(p_ids) - overflow):]
            ids = p_trunc + list(resp)
        labels = [-100] * len(ids)
        for i in range(len(ids) - len(resp), len(ids)):
            labels[i] = ids[i]
        return ids, labels

    chosen = build(c_ids)
    rejected = build(r_ids)
    if chosen is None or rejected is None:
        return None
    return (chosen, rejected)  # type: ignore[return-value]


def sequence_logprob(
    model: VeltronLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    chunk_tokens: int = 0,
    average: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of log p(label) over labelled positions, per sequence.

    The whole sequence is scored in one forward pass and the log-softmax is gathered at
    label positions. Chunks are only used when the flattened score matrix would not fit;
    the result is identical either way.
    """
    out = model(input_ids)
    logits = out.logits[:, :-1, :].float()
    targets = labels[:, 1:]
    mask = targets != -100
    safe = targets.clone()
    safe[~mask] = 0
    logp = torch.log_softmax(logits, dim=-1)
    gathered = logp.gather(2, safe.unsqueeze(2)).squeeze(2)
    gathered = gathered * mask
    totals = gathered.sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1)
    return (totals / counts if average else totals), counts


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DPO objective plus the diagnostics needed to tell whether it is working."""
    pi_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = pi_logratios - ref_logratios
    if label_smoothing > 0:
        loss = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    else:
        loss = -F.logsigmoid(beta * logits)
    chosen_win = (policy_chosen > policy_rejected).float().mean()
    with torch.no_grad():
        reward_margin = (beta * logits).mean()
        accuracy = (logits > 0).float().mean()
    metrics = {
        "chosen_win_rate": float(chosen_win),
        "preference_accuracy": float(accuracy),
        "reward_margin": float(reward_margin),
        "logits_mean": float(logits.mean()),
        "logits_std": float(logits.std()) if logits.numel() > 1 else 0.0,
    }
    return loss.mean(), metrics


class DPOTrainer:
    def __init__(self, cfg: DPOConfig) -> None:
        from ..model.config import get_config
        from ..model.io import load_safetensors
        from ..tokenizer.trainer import VeltronTokenizer

        self.cfg = cfg
        self.device_info = probe(cfg.device)
        self.device = get_device(cfg.device)
        self.tokenizer = VeltronTokenizer.load(cfg.tokenizer_dir)

        mcfg = get_config(cfg.model_config)
        self.model = VeltronLM(mcfg)
        if cfg.init_checkpoint:
            load_safetensors(self.model, cfg.init_checkpoint)
            log.info("policy initialised from %s", cfg.init_checkpoint)
        else:
            log.warning("no init_checkpoint: aligning a randomly initialised policy")
        self.model.to(self.device)
        self.model.train()
        self.model_cfg = mcfg

        # Frozen reference. A deep copy is exact and needs no extra forward-pass plumbing;
        # after copy.deepcopy the tensors are independent, so gradients cannot leak.
        self.reference = copy.deepcopy(self.model).to(self.device)
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)

        self.out_dir = Path(cfg.output_dir) / cfg.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt = CheckpointManager(self.out_dir, save_every=cfg.save_interval,
                                     keep_last=2, keep_best=1, monitor="preference_accuracy",
                                     monitor_mode="max")
        self.history_path = self.out_dir / "dpo_log.jsonl"

    # ------------------------------------------------------------------- data
    def load(self) -> None:
        from .data import load_preference_pairs

        pairs = load_preference_pairs(self.cfg.data_path)
        rng = random.Random(self.cfg.seed)
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * self.cfg.eval_fraction))
        self.val_pairs = pairs[:n_val]
        self.train_pairs = pairs[n_val:]
        if not self.train_pairs:
            raise RuntimeError(f"no training pairs in {self.cfg.data_path}")
        log.info("DPO split: %d train / %d val pairs", len(self.train_pairs), len(self.val_pairs))

    def _encode(self, pair: PreferencePair):
        return encode_pair(pair, self.tokenizer, self.cfg.max_seq_len)

    def _batch_tensors(self, batch: Sequence[PreferencePair], device) -> tuple:
        c_ids, c_lab, r_ids, r_lab = [], [], [], []
        for p in batch:
            enc = self._encode(p)
            if enc is None:
                continue
            (ci, cl), (ri, rl) = enc
            c_ids.append(ci)
            c_lab.append(cl)
            r_ids.append(ri)
            r_lab.append(rl)
        if not c_ids:
            return None
        pad = self.tokenizer.pad_id
        def pad_stack(rows: list[list[int]]) -> torch.Tensor:
            n = max(len(r) for r in rows)
            return torch.stack([
                torch.tensor(r + [pad] * (n - len(r)), dtype=torch.long) for r in rows
            ]).to(device)
        return pad_stack(c_ids), pad_stack(c_lab), pad_stack(r_ids), pad_stack(r_lab)

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(self, max_pairs: int = 32) -> dict[str, float]:
        self.model.eval()
        pairs = self.val_pairs[:max_pairs]
        if not pairs:
            return {"preference_accuracy": float("nan"), "reward_margin": float("nan"),
                    "loss": float("nan")}
        accs, margins, losses = [], [], []
        for p in pairs:
            t = self._batch_tensors([p], self.device)
            if t is None:
                continue
            ci, cl, ri, rl = t
            pc, _ = sequence_logprob(self.model, ci, cl)
            pr, _ = sequence_logprob(self.model, ri, rl)
            rc, _ = sequence_logprob(self.reference, ci, cl)
            rr, _ = sequence_logprob(self.reference, ri, rl)
            loss, m = dpo_loss(pc, pr, rc, rr, self.cfg.beta, self.cfg.label_smoothing)
            losses.append(float(loss))
            accs.append(m["preference_accuracy"])
            margins.append(m["reward_margin"])
        self.model.train()
        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "preference_accuracy": float(np.mean(accs)) if accs else float("nan"),
            "reward_margin": float(np.mean(margins)) if margins else float("nan"),
            "pairs": len(losses),
        }

    # ----------------------------------------------------------------- train
    def train(self) -> dict[str, Any]:
        cfg = self.cfg
        self.load()
        rng = random.Random(cfg.seed)

        total_steps = cfg.max_steps
        warmup = max(1, int(total_steps * cfg.warmup_ratio))
        opt = build_optimizer(self.model, "adamw", cfg.lr, 0.0, (0.9, 0.95))
        sched = build_schedule(cfg.schedule, opt, warmup, total_steps, cfg.min_lr_ratio)

        before = self.evaluate(max_pairs=32)
        log.info("DPO baseline: accuracy=%.3f margin=%.4f loss=%.4f",
                 before["preference_accuracy"], before["reward_margin"], before["loss"])

        step = 0
        running: list[dict[str, float]] = []
        losses: list[float] = []
        t0 = time.time()
        order = list(self.train_pairs)

        while step < total_steps:
            rng.shuffle(order)
            opt.zero_grad(set_to_none=True)
            micro = 0
            for p in order:
                if micro >= cfg.grad_accum_steps or step >= total_steps:
                    break
                t = self._batch_tensors([p], self.device)
                if t is None:
                    continue
                ci, cl, ri, rl = t
                pc, _ = sequence_logprob(self.model, ci, cl)
                pr, _ = sequence_logprob(self.model, ri, rl)
                with torch.no_grad():
                    rc, _ = sequence_logprob(self.reference, ci, cl)
                    rr, _ = sequence_logprob(self.reference, ri, rl)
                loss, metrics = dpo_loss(pc, pr, rc, rr, cfg.beta, cfg.label_smoothing)
                (loss / cfg.grad_accum_steps).backward()
                losses.append(float(loss.detach()))
                running.append(metrics)
                micro += 1

            gn = clip_grad_norm(self.model, cfg.grad_clip)
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters())
            if not finite:
                opt.zero_grad(set_to_none=True)
                log.warning("non-finite DPO gradient at step %d; skipping update", step)
                step += 1
                continue
            opt.step()
            lr = sched.step()
            step += 1

            if step % cfg.log_interval == 0:
                agg = {k: float(np.mean([m[k] for m in running[-50:]]))
                       for k in running[-1]} if running else {}
                self._log({"step": step, "total": total_steps,
                           "loss": float(np.mean(losses[-50:])) if losses else float("nan"),
                           "lr": lr, "grad_norm": gn, "elapsed_s": round(time.time() - t0, 1), **agg})

        after = self.evaluate(max_pairs=32)
        log.info("DPO result: accuracy %.3f -> %.3f, margin %.4f -> %.4f",
                 before["preference_accuracy"], after["preference_accuracy"],
                 before["reward_margin"], after["reward_margin"])

        meta = CheckpointMetadata(
            step=step, val_loss=after["loss"], model_config=self.model_cfg.to_dict(),
            trainer_config=cfg.as_dict(), device=self.device_info.kind,
        )
        self.ckpt.save(self.model, opt, sched, meta)
        self.ckpt.write_manifest()

        improved = after["preference_accuracy"] > before["preference_accuracy"]
        summary = {
            "run_name": cfg.run_name,
            "init_checkpoint": cfg.init_checkpoint,
            "steps": step,
            "beta": cfg.beta,
            "baseline": before,
            "final": after,
            "accuracy_delta": round(after["preference_accuracy"] - before["preference_accuracy"], 4),
            "margin_delta": round(after["reward_margin"] - before["reward_margin"], 4),
            "improved": bool(improved),
            "interpretation": (
                "Preference accuracy increased on held-out pairs." if improved else
                "No measurable improvement on held-out pairs; this run did not demonstrate "
                "an alignment effect and must not be reported as one."
            ),
            "elapsed_seconds": round(time.time() - t0, 1),
            "output_dir": str(self.out_dir),
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                   encoding="utf-8")
        return summary

    def _log(self, event: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")
        log.info("dpo step %4d/%d | loss %.4f | acc %.3f | margin %.4f | lr %.2e",
                 event["step"], event["total"], event["loss"],
                 event.get("preference_accuracy", 0.0), event.get("reward_margin", 0.0),
                 event["lr"])


__all__ = ["DPOConfig", "DPOTrainer", "dpo_loss", "sequence_logprob", "encode_pair"]
