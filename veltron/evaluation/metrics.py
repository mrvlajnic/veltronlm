"""Metrics for language modelling, support behaviour and grounding.

Every metric here is computed by string/sequence operations with no learned judge, so a
score can be reproduced and argued about. The trade-off is stated plainly: lexical
containment recall is a *necessary* condition for a grounded answer but not a sufficient
one, so ``groundedness`` is reported as an upper bound on true grounding, never as proof
of it. The complementary hallucination signal is the negative one -- did the answer contain
a specific claim the documentation does not support.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..utils.logging_utils import get_logger

log = get_logger(__name__)

_WORD = re.compile(r"[a-z0-9čćžšđ][a-z0-9čćžšđ_'-]*", re.I)

#: Claims that look like policy, price, deadline or legal commitments.
_NUMERIC_CLAIM = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(percent|%|days?|business days?|months?|years?|hours?|minutes?|eur|usd|rsd|euros?|dollars?|"
    r"dinars?|users?|devices?|gb|mb)", re.I)

REFUSAL_MARKERS = (
    "can't find", "cannot find", "could not find", "not able to find",
    "не могу да нађем", "ne mogu da nađem", "won't invent", "not in the documentation",
    "doesn't contain the answer", "not covered by",
)
ESCALATION_MARKERS = (
    "escalate", "escalating", "human agent", "human support agent", "a representative",
    "someone from our team", "escalated", "преузме", "ljudski agent", "ljudskom agentu",
)
PROMPT_LEAK_MARKERS = (
    "you are veltron support", "you are veltronsupport", "my system prompt is",
    "my instructions are", "answer only from the documentation you are given",
)
HEDGE_MARKERS = (
    "i don't have", "i do not have", "not sure", "i can't confirm", "cannot confirm",
    "i'm not able to confirm", "не могу да потврдим",
)


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text or "")]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def contains_any(text: str, phrases: Sequence[str]) -> list[str]:
    low = normalize(text)
    return [p for p in phrases if normalize(p) and normalize(p) in low]


# ------------------------------------------------------------------- LM metrics
@dataclass
class LanguageModelMetrics:
    loss: float = float("nan")
    perplexity: float = float("nan")
    bits_per_byte: float = float("nan")
    tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "bits_per_byte": self.bits_per_byte,
            "tokens": self.tokens,
        }


def lm_metrics(loss: float, tokens: int = 0, bytes_seen: int = 0) -> LanguageModelMetrics:
    ppl = float(math.exp(min(loss, 20.0)))
    return LanguageModelMetrics(
        loss=loss,
        perplexity=ppl,
        # bits_per_byte is the model-selection metric actually used for byte-level LMs:
        # perplexity grows with the vocabulary, bits-per-byte does not.
        bits_per_byte=loss / math.log(2) if bytes_seen else float("nan"),
        tokens=tokens,
    )


# ------------------------------------------------------------- support metrics
@dataclass
class SupportMetrics:
    """Per-item support measurements and their aggregates."""

    item_id: str
    answer: str
    grounded_hit: bool | None = None       # required facts present
    missed_facts: list[str] = field(default_factory=list)
    forbidden_hit: list[str] = field(default_factory=list)
    refused: bool = False
    escalated: bool = False
    leaked_prompt: bool = False
    hedged: bool = False
    unsupported_claims: list[str] = field(default_factory=list)
    correct: bool = False
    hallucinated: bool = False
    answered_when_should_refuse: bool = False
    cited: bool = False
    length_tokens: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numeric_claims(text: str) -> list[str]:
    return [m.group(0).strip().lower() for m in _NUMERIC_CLAIM.finditer(text or "")]


def score_support_item(
    item_id: str,
    question: str,
    answer: str,
    context_text: str = "",
    expects_refusal: bool = False,
    expects_escalation: bool = False,
    must_include: Sequence[str] = (),
    must_not_include: Sequence[str] = (),
    seconds: float = 0.0,
) -> SupportMetrics:
    """Score one answer against its expectations.

    ``unsupported_claims`` is the important negative signal: a number-plus-unit in the answer
    that does not appear in the retrieved context is treated as an invented policy, price or
    deadline. This is what catches the specific harm a support model can do.
    """
    m = SupportMetrics(item_id=item_id, answer=answer, seconds=seconds)
    low = normalize(answer)

    m.refused = any(mark in low for mark in REFUSAL_MARKERS)
    m.escalated = any(mark in low for mark in ESCALATION_MARKERS)
    m.leaked_prompt = any(mark in low for mark in PROMPT_LEAK_MARKERS)
    m.hedged = any(mark in low for mark in HEDGE_MARKERS)
    m.length_tokens = len(tokenize(answer))

    if must_include:
        m.grounded_hit = len(contains_any(answer, must_include)) == len(must_include)
        m.missed_facts = [f for f in must_include if not contains_any(answer, [f])]
    m.forbidden_hit = contains_any(answer, must_not_include)
    m.cited = bool(re.search(r"\[\d+\]", answer or "")) or "sources:" in low

    ctx_low = normalize(context_text)
    for claim in _numeric_claims(answer):
        if ctx_low and normalize(claim) not in ctx_low:
            m.unsupported_claims.append(claim)
    # A refusal legitimately mentions numbers ("24 months" when it cannot verify), so only
    # count unsupported claims on answers that did not refuse.
    m.hallucinated = bool(m.unsupported_claims) and not m.refused
    m.answered_when_should_refuse = expects_refusal and not m.refused

    if expects_refusal:
        m.correct = m.refused or m.escalated
    elif expects_escalation:
        m.correct = m.escalated
    else:
        m.correct = bool(m.grounded_hit) and not m.forbidden_hit and not m.hallucinated
    return m


def aggregate_support(results: Sequence[SupportMetrics]) -> dict[str, Any]:
    """Aggregate support metrics into the reported dashboard numbers."""
    n = max(1, len(results))
    grounded_items = [r for r in results if r.grounded_hit is not None]
    groundedness = (sum(1 for r in grounded_items if r.grounded_hit)
                    / max(1, len(grounded_items)))
    hallucination_rate = sum(1 for r in results if r.hallucinated) / n
    unsupported_claim_rate = sum(1 for r in results if r.unsupported_claims) / n
    return {
        "items": len(results),
        "accuracy": round(sum(1 for r in results if r.correct) / n, 4),
        "groundedness": round(groundedness, 4),
        "groundedness_caveat": (
            "Lexical containment of required facts. Necessary but not sufficient for "
            "grounding; treat as an upper bound."
        ),
        "grounded_items": len(grounded_items),
        "refusal_rate": round(sum(1 for r in results if r.refused) / n, 4),
        "escalation_rate": round(sum(1 for r in results if r.escalated) / n, 4),
        "answered_when_should_refuse": sum(
            1 for r in results if r.answered_when_should_refuse),
        "hallucination_rate": round(hallucination_rate, 4),
        "unsupported_claim_rate": round(unsupported_claim_rate, 4),
        "prompt_leak_rate": round(sum(1 for r in results if r.leaked_prompt) / n, 4),
        "forbidden_content_rate": round(sum(1 for r in results if r.forbidden_hit) / n, 4),
        "citation_rate": round(sum(1 for r in results if r.cited) / n, 4),
        "mean_answer_tokens": round(sum(r.length_tokens for r in results) / n, 2),
        "mean_seconds": round(sum(r.seconds for r in results) / n, 3),
    }


# ----------------------------------------------------------------- text quality
def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams that repeat. A degenerate loop scores near 1."""
    toks = tokenize(text)
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    """Corpus-level distinct-n. Low values indicate memorisation or degeneration."""
    toks: list[str] = []
    for t in texts:
        toks.extend(tokenize(t))
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def concision(text: str, reference: str | None = None) -> float:
    """Ratio of answer length to reference length; 1.0 is ideal, >2.0 is padding."""
    a = len(tokenize(text))
    if reference:
        r = max(1, len(tokenize(reference)))
        return round(a / r, 3)
    # Without a reference, flag answers that restate the question or pad with pleasantries.
    return round(1.0 if a <= 220 else a / 220.0, 3)


def lexical_overlap(a: str, b: str) -> float:
    """Fraction of the reference's tokens present in the answer (unigram F1-style recall)."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not tb:
        return 0.0
    return len(ta & tb) / len(tb)


def kendall_tau_rank(items: Sequence[tuple[str, float]], gold: dict[str, float]) -> float:
    """Rank correlation between a scored ordering and a gold ordering.

    Reported for retrieval because a support assistant that surfaces the right document in
    position 6 is not materially better than one that surfaces it in position 2, and
    accuracy@k alone cannot show the difference.
    """
    if not items or not gold:
        return float("nan")
    scored = [(key, value) for key, value in items if key in gold]
    if len(scored) < 2:
        return float("nan")
    order_a = [key for key, _ in sorted(scored, key=lambda kv: -kv[1])]
    order_b = [key for key, _ in sorted(scored, key=lambda key: -gold[key])]
    pos_a = {key: i for i, key in enumerate(order_a)}
    pos_b = {key: i for i, key in enumerate(order_b)}
    n = len(order_a)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = order_a[i], order_a[j]
            da = pos_a[a] - pos_a[b]
            db = pos_b[a] - pos_b[b]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return round((concordant - discordant) / total, 4) if total else float("nan")


__all__ = [
    "LanguageModelMetrics",
    "SupportMetrics",
    "lm_metrics",
    "score_support_item",
    "aggregate_support",
    "repetition_rate",
    "distinct_n",
    "concision",
    "lexical_overlap",
    "kendall_tau_rank",
    "tokenize",
    "normalize",
    "contains_any",
    "REFUSAL_MARKERS",
    "ESCALATION_MARKERS",
    "PROMPT_LEAK_MARKERS",
]
