"""Evaluation framework."""

from .datasets import (
    ADVERSARIAL_ITEMS,
    LONG_CONTEXT_ITEMS,
    SUITES,
    SUPPORT_ITEMS,
    EvalItem,
    all_items,
    load_suite,
    save_suites,
    suite_stats,
)
from .metrics import (
    SupportMetrics,
    aggregate_support,
    concision,
    distinct_n,
    kendall_tau_rank,
    lexical_overlap,
    lm_metrics,
    repetition_rate,
    score_support_item,
)
from .runner import EvalReport, Evaluator, build_long_context_prompts, load_validation_texts

__all__ = [
    "EvalItem",
    "SUITES",
    "SUPPORT_ITEMS",
    "ADVERSARIAL_ITEMS",
    "LONG_CONTEXT_ITEMS",
    "all_items",
    "load_suite",
    "suite_stats",
    "save_suites",
    "SupportMetrics",
    "lm_metrics",
    "score_support_item",
    "aggregate_support",
    "repetition_rate",
    "distinct_n",
    "concision",
    "lexical_overlap",
    "kendall_tau_rank",
    "Evaluator",
    "EvalReport",
    "build_long_context_prompts",
    "load_validation_texts",
]
