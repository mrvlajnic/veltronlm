"""Evaluation CLI.

    python -m veltron.evaluate --checkpoint checkpoints/micro-pretrain \
        --suites support adversarial --perplexity --retrieval --long-context \
        --out reports/eval-micro.json

    python -m veltron.evaluate --configs micro --suite-all     # registry sweep, no weights

Every number printed is measured. Where a metric is not applicable it is reported as
``"not_evaluated"`` rather than omitted, so a reader can tell the difference between "bad"
and "not run".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .model.config import REGISTRY
from .utils.logging_utils import setup_logging


def _registry_matrix() -> dict[str, Any]:
    from .model.config import registry_table
    from .model.transformer import count_parameters_via_meta

    rows = []
    for r in registry_table():
        entry = REGISTRY[r["key"]]
        rows.append({
            **r,
            "analytic_matches_meta":
                r["params"] == count_parameters_via_meta(entry.config),
            "trainable_on_this_host": entry.trainable_on_this_host,
            "notes": entry.notes,
        })
    return {"rows": rows}


def _benchmark_matrix() -> dict[str, Any]:
    """The benchmark matrix, including rows that were not evaluated.

    A matrix with only the passing numbers is cherry-picking. Rows that could not be run on
    this host are present with an explicit reason.
    """
    return {
        "policy": (
            "Every capability is listed. Cells that were not measured read "
            "'not_evaluated' with a reason, never a guess."
        ),
        "rows": [
            {"benchmark": "Held-out perplexity", "veltronlm-micro": "measured",
             "veltronlm-4b": "not_evaluated: no weights",
             "qwen2.5-1.5b": "not_evaluated: not run on this host"},
            {"benchmark": "Support suite (16 items)", "veltronlm-micro": "measured",
             "veltronlm-4b": "not_evaluated: no weights",
             "qwen2.5-1.5b": "not_evaluated: not run on this host"},
            {"benchmark": "Adversarial suite (12 items)", "veltronlm-micro": "measured",
             "veltronlm-4b": "not_evaluated: no weights",
             "qwen2.5-1.5b": "not_evaluated: not run on this host"},
            {"benchmark": "Retrieval hit@1", "system": "measured",
             "veltronlm-4b": "not_applicable: retriever is model-independent",
             "qwen2.5-1.5b": "not_applicable"},
            {"benchmark": "Long-context needle recall", "veltronlm-micro": "measured",
             "veltronlm-4b": "not_evaluated: no weights",
             "qwen2.5-1.5b": "not_evaluated: not run on this host"},
            {"benchmark": "Standard LM benchmarks (MMLU/HellaSwag/etc.)",
             "veltronlm-micro": "not_evaluated: too small to be meaningful",
             "veltronlm-4b": "not_evaluated: no weights",
             "qwen2.5-1.5b": "not_evaluated: not run on this host"},
            {"benchmark": "Human evaluation", "all": "not_evaluated: no raters available"},
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m veltron.evaluate")
    ap.add_argument("--checkpoint", default="", help="checkpoint or run directory")
    ap.add_argument("--tokenizer", default="", help="override the tokenizer directory")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--suites", nargs="*", default=["support"],
                    help="support | adversarial | all")
    ap.add_argument("--perplexity", action="store_true")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--long-context", action="store_true")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--rag-index", default="models/rag_index")
    ap.add_argument("--dataset", default="datasets/dataset-v1")
    ap.add_argument("--out", default="reports/evaluation.json")
    ap.add_argument("--registry", action="store_true", help="print the model matrix and exit")
    ap.add_argument("--matrix", action="store_true", help="print the benchmark matrix and exit")
    args = ap.parse_args(argv)
    setup_logging(level="INFO")

    if args.registry:
        print(json.dumps(_registry_matrix(), indent=2))
        return 0
    if args.matrix:
        print(json.dumps(_benchmark_matrix(), indent=2))
        return 0

    payload: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "suites_requested": args.suites,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ---- model-independent measurements
    if args.retrieval:
        from .rag.pipeline import RAGConfig, RAGPipeline

        pipe = RAGPipeline(cfg=RAGConfig())
        if Path(args.rag_index, "chunks.json").exists():
            pipe.load_index(args.rag_index)
        else:
            pipe.build_index(args.rag_index)
        from .rag.ingest import knowledge_base_chunks

        chunks = knowledge_base_chunks()
        questions = [
            ("How long is the warranty?", {"policy-warranty"}),
            ("My hub shows as offline", {"trouble-hub-offline"}),
            ("veltron x1 status light blinking amber pairing mode",
             {"trouble-hub-offline", "product-veltron-hub"}),
            ("What is the turnaround time for an RMA?", {"policy-warranty"}),
            ("Koliko traje garancija?", {"policy-warranty"}),
            ("Can I get a refund after 45 days?", {"policy-refund"}),
            ("Where is my parcel? tracking says nothing", {"policy-shipping"}),
            ("I forgot my password", {"policy-security"}),
            ("What are the dimensions of the VeltronSense?", {"product-veltron-sense"}),
            ("How much is the Plus plan?", {"product-veltron-cloud"}),
            ("Someone logged into my account and I did not", {"policy-security"}),
            ("Can I close my account and delete my data?",
             {"policy-privacy-data", "faq-general", "faq-privacy-serbian"}),
            ("I want to send the device back for repair", {"policy-warranty", "policy-refund"}),
            ("Does the app need Bluetooth permission to pair?", {"trouble-app-pairing"}),
        ]
        from .evaluation.runner import Evaluator

        ev = Evaluator(engine=None, rag=pipe)
        payload["retrieval"] = ev.run_retrieval(pipe.retriever, questions)
        print(f"\nretrieval: hit@1={payload['retrieval']['hit@1']} "
              f"recall@5={payload['retrieval']['recall@5']}")

    # ---- model-dependent measurements
    if args.checkpoint:
        from .evaluation.runner import Evaluator, load_validation_texts
        from .inference.engine import load_engine
        from .rag.pipeline import RAGConfig, RAGPipeline

        engine = load_engine(checkpoint=args.checkpoint, tokenizer_dir=args.tokenizer or None,
                             device=args.device)
        payload["model"] = engine.info.as_dict()
        print(f"\nmodel: {engine.info.model_name} "
              f"({engine.info.parameters_billions:.4f}B) on {engine.info.device}")

        from .inference.generator import GenerationConfig

        gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature,
                                   greedy=args.temperature <= 0.0)

        rag = None
        if Path(args.rag_index, "chunks.json").exists():
            rag = RAGPipeline(generator=engine.generator, cfg=RAGConfig())
            rag.load_index(args.rag_index)
        else:
            rag = RAGPipeline(generator=engine.generator, cfg=RAGConfig())
            rag.build_index(args.rag_index)

        from .rag.ingest import knowledge_base_chunks

        chunks = knowledge_base_chunks()
        ev = Evaluator(engine=engine, rag=rag, gen_cfg=gen_cfg)

        suites = ["support", "adversarial"] if "all" in args.suites else args.suites
        if args.perplexity:
            texts = load_validation_texts(args.dataset, n=16)
            if texts:
                payload["perplexity"] = ev.perplexity(texts, max_tokens=120_000)
                print(f"perplexity: loss={payload['perplexity']['loss']:.4f} "
                      f"ppl={payload['perplexity']['perplexity']:.2f}")
            else:
                payload["perplexity"] = {"status": "not_evaluated",
                                         "reason": "no validation documents found"}

        for suite in suites:
            agg, records = ev.run_support_suite(suite, max_items=args.max_items)
            payload[suite] = agg
            print(f"\n{suite}: accuracy={agg['accuracy']} groundedness={agg['groundedness']} "
                  f"hallucination_rate={agg['hallucination_rate']} "
                  f"refusal_rate={agg['refusal_rate']}")
            payload.setdefault("_per_item", []).extend(records)

        if args.long_context:
            agg, records = ev.run_long_context(chunks)
            payload["long_context"] = agg
            print(f"\nlong_context: needle_recall={agg['needle_recall']} "
                  f"by_depth={agg['by_depth']}")
    elif any((args.perplexity, args.long_context)) or set(args.suites) - {"support"}:
        payload["model_dependent"] = "not_evaluated: --checkpoint was not provided"

    payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                              encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
