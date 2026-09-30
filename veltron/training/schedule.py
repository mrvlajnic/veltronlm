"""Learning-rate schedules and optimiser construction."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from ..utils.logging_utils import get_logger

log = get_logger(__name__)


class WarmupCosineSchedule:
    """Linear warmup then cosine decay to ``min_lr_ratio`` of the peak rate.

    The schedule is defined as a function of *total training steps*, not wall-clock or
    token count, so a run that is preempted and resumed lands on exactly the same rate it
    would have used without the interruption.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.step_count = 0
        self.base_lrs = [g.get("lr", 0.0) for g in optimizer.param_groups]
        self._set(last_epoch + 1)

    def _lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            # Warmup from 1/warmup to 1.0 linearly; step 0 is a no-op to avoid a
            # zero-learning-rate first update.
            frac = (step + 1) / max(1, self.warmup_steps)
            return frac
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _set(self, step: int) -> None:
        scale = self._lr_at(step)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * scale
        self.step_count = step

    def step(self) -> float:
        self._set(self.step_count + 1)
        return self.optimizer.param_groups[0]["lr"]

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"step_count": self.step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.base_lrs = state["base_lrs"]
        self._set(int(state["step_count"]))


class WarmupStableSchedule:
    """Linear warmup then a flat rate.

    Used for fine-tuning, where decaying the rate across a short SFT run mostly wastes
    the last third of the data.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int,
                 total_steps: int, min_lr_ratio: float = 1.0, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = total_steps
        self.step_count = 0
        self.base_lrs = [g.get("lr", 0.0) for g in optimizer.param_groups]
        self._set(last_epoch + 1)

    def _set(self, step: int) -> None:
        frac = min(1.0, (step + 1) / max(1, self.warmup_steps)) if self.warmup_steps else 1.0
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * frac
        self.step_count = step

    def step(self) -> float:
        self._set(self.step_count + 1)
        return self.optimizer.param_groups[0]["lr"]

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"step_count": self.step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.base_lrs = state["base_lrs"]
        self._set(int(state["step_count"]))


SCHEDULES: dict[str, Callable[..., Any]] = {
    "cosine": WarmupCosineSchedule,
    "stable": WarmupStableSchedule,
}


def build_schedule(name: str, optimizer: torch.optim.Optimizer, warmup_steps: int,
                   total_steps: int, min_lr_ratio: float = 0.1,
                   last_epoch: int = -1) -> Any:
    key = name.lower()
    if key not in SCHEDULES:
        raise ValueError(f"unknown schedule {name!r}; available: {sorted(SCHEDULES)}")
    return SCHEDULES[key](optimizer, warmup_steps, total_steps, min_lr_ratio, last_epoch)


def param_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    no_decay_norms_and_biases: bool = True,
    embedding_weight_decay: float | None = None,
) -> list[dict[str, Any]]:
    """Split parameters into decayed and non-decayed groups.

    Decaying the RMSNorm gains and the embedding is harmful: both are direction-sensitive
    vectors whose magnitude carries meaning, so shrinking them toward zero degrades the
    model without acting as a useful regulariser.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    emb_decay: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            no_decay.append(p)
        elif "embed" in name or name.startswith("lm_head"):
            emb_decay.append(p)
        else:
            decay.append(p)

    groups: list[dict[str, Any]] = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if emb_decay:
        groups.append(
            {"params": emb_decay,
             "weight_decay": weight_decay if embedding_weight_decay is None else embedding_weight_decay}
        )
    return groups


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    embedding_weight_decay: float | None = None,
) -> torch.optim.Optimizer:
    groups = param_groups(model, weight_decay, embedding_weight_decay=embedding_weight_decay)
    key = name.lower()
    if key == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if key == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=betas, eps=eps)
    if key == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9)
    raise ValueError(f"unknown optimizer {name!r}")


def clip_grad_norm(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Clip gradients and return the pre-clip norm.

    The pre-clip norm is logged rather than just the post-clip value because a rising
    pre-clip norm is the earliest reliable signal that a run is about to diverge.
    """
    if max_norm <= 0:
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), norm_type)
        return float(total)
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)
    return float(total)


__all__ = [
    "WarmupCosineSchedule",
    "WarmupStableSchedule",
    "build_schedule",
    "build_optimizer",
    "param_groups",
    "clip_grad_norm",
    "SCHEDULES",
]
