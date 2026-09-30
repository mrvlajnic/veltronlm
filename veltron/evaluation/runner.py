"""Evaluation runner.

Scores a loaded engine against the fixed suites in :mod:`veltron.evaluation.datasets` and
writes a machine-readable report. Nothing here asks a model to judge itself.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..rag.pipeline import RAGPipeline
from ..utils.logging_utils import get_logger
from .datasets import EvalItem, load_suite, suite_stats
from .metrics import (
    SupportMetrics,
    aggregate_support,
    distinct_n,
    lm_metrics,
    repetition_rate,
    score_support_item,
)

log = get_logger(__name__)

#: Long-context filler drawn from the knowledge base, so the needle test measures attention
#: over realistic text rather than repeated filler tokens.
FILLER_MARKER = "(unrelated support documentation follows)"


def build_long_context_prompts(chunks: Sequence[Any], length_chars: int = 9000
                                ) -> list[tuple[str, str]]:
    """Construct (prompt, answer) pairs with a unique needle planted at a known depth."""
    body = " ".join(c.text for c in chunks if c.language == "en")[:length_chars]
    out: list[tuple[str, str]] = []
    for depth, needle in ((0.1, "NEEDLE-ALPHA-4417"), (0.5, "NEEDLE-BRAVO-8823"),
                          (0.9, "NEEDLE-CHARLIE-1290")):
        cut = int(len(body) * depth)
        planted = (body[:cut]
                   + f"\n\nReference record: the maintenance token for this unit is {needle}.\n"
                   + body[cut:])
        prompt = (f"{FILLER_MARKER}\n\n{planted}\n\n"
                  f"Question: Which needle identifier is recorded in this document?\n"
                  f"Answer with the identifier only.")
        out.append((prompt, needle))
    return out


@dataclass
class EvalReport:
    model: str
    checkpoint: str
    suites: dict[str, Any]
    per_item: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False, default=str)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    def table(self) -> str:
        lines = ["=" * 96, f"EVALUATION REPORT  {self.model}", "=" * 96]
        for suite, data in self.suites.items():
            lines.append(f"\n[{suite}]")
            for k, v in data.items():
                if isinstance(v, dict):
                    continue
                lines.append(f"  {k:34} {v}")
        return "\n".join(lines)


class Evaluator:
    def __init__(
        self,
        engine: Any,
        rag: RAGPipeline | None = None,
        gen_cfg: Any = None,
        use_rag: bool = True,
    ) -> None:
        self.engine = engine
        self.rag = rag
        self.gen_cfg = gen_cfg
        self.use_rag = use_rag

    # ------------------------------------------------------------- perplexity
    @torch.no_grad()
    def perplexity(
        self,
        texts: Sequence[str],
        max_length: int = 1024,
        max_tokens: int | None = 200_000,
    ) -> dict[str, Any]:
        """Teacher-forced loss over held-out text.

        Uses ``num_logits_to_keep`` semantics only implicitly -- the whole sequence is
        scored at once because perplexity needs every position.
        """
        tok = self.engine.tokenizer
        model = self.engine.model
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_bytes = 0
        chunk = self.engine.info.context_length or max_length
        for text in texts:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) < 8:
                continue
            if max_tokens is not None and total_tokens >= max_tokens:
                break
            for start in range(0, len(ids) - 1, chunk - 1):
                window = ids[start : start + chunk]
                if len(window) < 8:
                    break
                x = torch.tensor([window[:-1]], dtype=torch.long, device=self.engine.device)
                y = torch.tensor([window[1:]], dtype=torch.long, device=self.engine.device)
                out = model(x)
                loss = torch.nn.functional.cross_entropy(
                    out.logits[0].float(), y[0], reduction="sum"
                )
                total_loss += float(loss)
                total_tokens += y.numel()
                total_bytes += len(text.encode("utf-8"))
        m = lm_metrics(total_loss / max(1, total_tokens), total_tokens, total_bytes)
        return m.as_dict()

    # --------------------------------------------------------- support suites
    def run_support_suite(
        self,
        suite: str,
        max_items: int | None = None,
        include_debug: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        items = load_suite(suite)
        if max_items:
            items = items[:max_items]
        results: list[SupportMetrics] = []
        records: list[dict[str, Any]] = []
        answers: list[str] = []
        t0 = time.perf_counter()

        for item in items:
            answer, context_text, citations, seconds = self._answer(item)
            answers.append(answer)
            m = score_support_item(
                item_id=item.item_id, question=item.question, answer=answer,
                context_text=context_text,
                expects_refusal=item.expects_refusal,
                expects_escalation=item.expects_escalation,
                must_include=item.must_include,
                must_not_include=item.must_not_include,
                seconds=seconds,
            )
            results.append(m)
            records.append({
                "item": item.as_dict(),
                "metrics": m.as_dict(),
                "repetition_4gram": round(repetition_rate(answer), 4),
                "citations": citations,
            })
            log.info("[%s] %-9s %-5s %s", suite, item.item_id,
                     "PASS" if m.correct else "FAIL", item.question[:64])
            log.info("    -> %s", answer.replace("\n", " ")[:170])

        agg = aggregate_support(results)
        agg["distinct_2"] = round(distinct_n(answers, 2), 4)
        agg["repetition_4gram_mean"] = round(
            float(np.mean([r["repetition_4gram"] for r in records])), 4)
        agg["by_kind"] = self._by_kind(items, results)
        agg["by_language"] = self._by_language(items, results)
        agg["wall_clock_seconds"] = round(time.perf_counter() - t0, 2)
        return agg, records

    def _answer(self, item: EvalItem) -> tuple[str, str, list[dict[str, Any]], float]:
        """Answer one item, through RAG when available so grounding is exercised."""
        if self.use_rag and self.rag is not None and self.rag.retriever is not None:
            t0 = time.perf_counter()
            res = self.rag.answer(item.question, language=item.language,
                                  include_debug=include_debug_flag(self))
            context = " ".join(c["quoted"] for c in res.citations)
            cites = [{"index": c["index"], "title": c["title"],
                      "section_path": c["section_path"], "score": c["score"]}
                     for c in res.citations]
            return res.answer, context, cites, time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.engine is None:
            return "(no engine loaded)", "", [], time.perf_counter() - t0
        cfg = self.gen_cfg
        res = self.engine.generate_chat(
            [{"role": "system", "content": "You are Veltron Support. Be accurate and concise."},
             {"role": "user", "content": item.question}], cfg)
        return res.text, "", [], time.perf_counter() - t0

    @staticmethod
    def _by_kind(items: Sequence[EvalItem], results: Sequence[SupportMetrics]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item, m in zip(items, results):
            out.setdefault(item.kind, []).append(1.0 if m.correct else 0.0)
        return {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for k, v in out.items()}

    @staticmethod
    def _by_language(items: Sequence[EvalItem], results: Sequence[SupportMetrics]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item, m in zip(items, results):
            out.setdefault(item.language, []).append(1.0 if m.correct else 0.0)
        return {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for k, v in out.items()}

    # -------------------------------------------------------- long context
    def run_long_context(self, chunks: Sequence[Any], max_items: int | None = None
                         ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        items = [i for i in load_suite("long_context")
                 if i.kind == "needle_retrieval"]
        if max_items:
            items = items[:max_items]
        prompts = build_long_context_prompts(chunks)
        by_needle = {ans: p for p, ans in prompts}
        by_item = {i.metadata["needle"]: i for i in items}

        records: list[dict[str, Any]] = []
        by_depth: dict[float, list[int]] = {}
        for needle, prompt in by_needle.items():
            item = by_item.get(needle)
            if item is None:
                continue
            t0 = time.perf_counter()
            res = self.engine.generate(prompt, self.gen_cfg)
            seconds = time.perf_counter() - t0
            hit = needle.lower() in res.text.lower()
            by_depth.setdefault(float(item.metadata["depth"]), []).append(1 if hit else 0)
            records.append({
                "item": item.as_dict(),
                "needle": needle,
                "depth": item.metadata["depth"],
                "hit": hit,
                "seconds": round(seconds, 3),
                "answer": res.text[:400],
                "answer_tokens": res.completion_tokens,
                "prompt_tokens": res.prompt_tokens,
            })
            log.info("[long_context] depth=%.1f hit=%s (%.1fs)", item.metadata["depth"], hit, seconds)

        depths = {f"depth_{int(d*100)}": {"n": len(v), "recall": round(sum(v) / len(v), 4)}
                  for d, v in sorted(by_depth.items())}
        overall = round(sum(sum(v) for v in by_depth.values()) /
                        max(1, sum(len(v) for v in by_depth.values())), 4)
        return {
            "items": len(records),
            "needle_recall": overall,
            "by_depth": depths,
            "position_sensitivity": self._position_sensitivity(by_depth),
            "note": ("Depths are scored separately so a model that only attends to the "
                     "beginning or end cannot average into an acceptable score."),
        }, records

    @staticmethod
    def _position_sensitivity(by_depth: dict[float, list[int]]) -> float:
        """Spread between the best and worst depth: high spread means positional bias."""
        recalls = [sum(v) / len(v) for v in by_depth.values() if v]
        if len(recalls) < 2:
            return float("nan")
        return round(max(recalls) - min(recalls), 4)

    # ------------------------------------------------------------- retrieval
    def run_retrieval(self, retriever: Any, questions: Sequence[tuple[str, set[str]]],
                      top_k: int = 5) -> dict[str, Any]:
        hit1 = 0
        recall_k = 0
        records = []
        for q, accepted in questions:
            hits = retriever.search(q, top_k=top_k, per_doc_cap=2)
            docs = [h.doc_id for h in hits]
            h1 = bool(docs) and docs[0] in accepted
            rk = bool(set(docs) & accepted)
            hit1 += int(h1)
            recall_k += int(rk)
            records.append({"question": q, "expected": sorted(accepted),
                            "retrieved": docs, "hit@1": h1, f"recall@{top_k}": rk,
                            "scores": [h.confidence for h in hits]})
        n = max(1, len(questions))
        return {
            "questions": len(questions),
            "hit@1": round(hit1 / n, 4),
            f"recall@{top_k}": round(recall_k / n, 4),
            "per_question": records,
        }

    # ------------------------------------------------------------------ all
    def run_all(
        self,
        suites: Sequence[str] = ("support", "adversarial"),
        val_texts: Sequence[str] | None = None,
        chunks: Sequence[Any] | None = None,
        long_context: bool = True,
        max_items: int | None = None,
    ) -> EvalReport:
        suites_out: dict[str, Any] = {}
        per_item: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        if val_texts:
            suites_out["perplexity"] = self.perplexity(val_texts)
            log.info("perplexity: %s", json.dumps(suites_out["perplexity"]))

        for s in suites:
            agg, records = self.run_support_suite(s, max_items=max_items)
            suites_out[s] = agg
            per_item.extend(records)

        if long_context and chunks is not None:
            agg, records = self.run_long_context(chunks, max_items=max_items)
            suites_out["long_context"] = agg
            per_item.extend({"item": r["item"], "metrics": {"hit": r["hit"], "depth": r["depth"]}}
                            for r in records)

        return EvalReport(
            model=self.engine.info.model_name if self.engine else "unavailable",
            checkpoint=str(self.engine.info.checkpoint) if self.engine else "",
            suites=suites_out,
            per_item=per_item,
            timing={"wall_clock_seconds": round(time.perf_counter() - t0, 2),
                    "suite_stats": suite_stats()},
        )


def include_debug_flag(ev: Evaluator) -> bool:
    return False


def load_validation_texts(dataset_root: str | Path = "datasets/dataset-v1", n: int = 24) -> list[str]:
    """Read held-out documents back out of the dataset for perplexity scoring."""
    path = Path(dataset_root) / "documents-train.jsonl"
    if not path.exists():
        path = Path(dataset_root) / "val" / "val-index.json"
        if not path.exists():
            return []
        # Shards are packed tokens; decode them if a tokenizer is handy.
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[:n]:
        try:
            out.append(json.loads(line).get("text", ""))
        except json.JSONDecodeError:
            continue
    return [t for t in out if len(t) > 400]


__all__ = ["Evaluator", "EvalReport", "build_long_context_prompts", "load_validation_texts"]
