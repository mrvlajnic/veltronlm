"""Training package: schedules, optimiser construction, checkpointing, trainer."""

from .checkpoint import CheckpointManager, CheckpointMetadata, verify_checkpoint
from .schedule import (
    WarmupCosineSchedule,
    WarmupStableSchedule,
    build_optimizer,
    build_schedule,
    clip_grad_norm,
    param_groups,
)
from .trainer import TrainConfig, Trainer, TrainState

__all__ = [
    "CheckpointManager",
    "CheckpointMetadata",
    "verify_checkpoint",
    "WarmupCosineSchedule",
    "WarmupStableSchedule",
    "build_optimizer",
    "build_schedule",
    "clip_grad_norm",
    "param_groups",
    "Trainer",
    "TrainConfig",
    "TrainState",
]
