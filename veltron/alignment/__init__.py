"""Preference alignment package."""

from .data import (
    FAILURE_MODES,
    PreferencePair,
    build_dataset_pairs,
    build_preference_pairs,
    load_preference_pairs,
    save_preference_pairs,
)
from .trainer import DPOConfig, DPOTrainer, dpo_loss, encode_pair, sequence_logprob

__all__ = [
    "PreferencePair",
    "FAILURE_MODES",
    "build_preference_pairs",
    "build_dataset_pairs",
    "save_preference_pairs",
    "load_preference_pairs",
    "DPOConfig",
    "DPOTrainer",
    "dpo_loss",
    "sequence_logprob",
    "encode_pair",
]
