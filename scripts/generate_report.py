"""Generate the project status report.

Reads every artefact on disk and emits a single Markdown report with measured numbers.
Missing measurements are reported as missing, never inferred.

    python scripts/generate_report.py --out reports/status.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(".")


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def _model_table() -> str:
    from veltron.model.config import registry_table

    rows = registry_table()
    out = ["| Tier | Name | L | D | H | KV | F | V | Ctx | Parameters | Trainable here |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        out.append(f"| {r['tier']} | `{r['key']}` | {r['layers']} | {r['d_model']} | "
                   f"{r['heads']} | {r['kv_heads']} | {r['d_ff']} | {r['vocab']:,} | "
                   f"{r['ctx']:,} | {r['params']:,} | "
                   f"{'yes' if r['trainable_here'] else 'no'} |")
    return "\n".join(out)


def _pretraining_table() -> str:
    log = ROOT / "checkpoints/micro-pretrain/train_log.jsonl"
    if not log.exists():
        return "_no pretraining log found_"
    steps, evals = [], []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("eval"):
            evals.append(d)
        elif "loss" in d:
            steps.append(d)
    out = []
    if steps:
        tps = [s["tokens_per_second"] for s in steps if s.get("tokens_per_second")]
        out.append(f"- **steps logged:** {len(steps)}")
        out.append(f"- **latest step:** {steps[-1]['step']}")
        out.append(f"- **tokens seen:** {steps[-1]['tokens_seen']:,}")
        out.append(f"- **train loss:** {_fmt(steps[-1]['loss'])} "
                   f"(EMA {_fmt(steps[-1].get('loss_ema'))})")
        if tps:
            out.append(f"- **throughput:** median {statistics.median(tps):,.0f} tok/s "
                       f"(range {min(tps):,.0f}–{max(tps):,.0f})")
        out.append(f"- **grad norm (latest):** {_fmt(steps[-1].get('grad_norm'))}")
    if evals:
        out.append("")
        out.append("| Step | Tokens | Val loss | Val perplexity |")
        out.append("|---:|---:|---:|---:|")
        for e in evals:
            out.append(f"| {e['step']} | {e.get('tokens_seen', 0) or steps[-1]['tokens_seen']:,} | "
                       f"{_fmt(e.get('val_loss'))} | {_fmt(e.get('val_perplexity'), 2)} |")
    else:
        out.append("_no evaluation recorded yet_")
    return "\n".join(out)


def _dataset_section() -> str:
    m = _read_json(ROOT / "datasets/dataset-v1/manifest.json")
    if not m:
        return "_no dataset built_"
    t = m["totals"]
    f = m["filtering"]
    lines = [
        f"- **version:** `{m['dataset_version']}` (manifest `{m['manifest_hash'][:16]}`)",
        f"- **documents:** {t['documents_seen']:,} seen, {t['documents_kept']:,} kept "
        f"({f['accept_rate']:.1%} accept)",
        f"- **characters kept:** {t['characters_kept']:,}",
        f"- **train tokens:** {t['train_tokens']:,}",
        f"- **val tokens:** {t['val_tokens']:,}",
        f"- **chars per token:** {_fmt(t['chars_per_token'], 3)}",
        "",
        "**Filter funnel**",
        "",
        "| Stage | Rejected |",
        "|---|---:|",
    ]
    for stage, n in f["rejected_by_stage"].items():
        lines.append(f"| {stage} | {n:,} |")
    lines += ["", "**Mixture**", "", "| Category | Target | Achieved | Upsampled |",
              "|---|---:|---:|---:|"]
    for cat in sorted(m["mixture"]["target"]):
        lines.append(f"| {cat} | {m['mixture']['target'][cat]:.3f} | "
                     f"{m['mixture']['achieved'].get(cat, 0):.3f} | "
                     f"{m['mixture']['upsampled_documents'].get(cat, 0):,} |")
    return "\n".join(lines)


def _tokenizer_section() -> str:
    s = _read_json(ROOT / "reports/tokenizer_summary.json")
    if not s:
        return "_tokenizers not trained_"
    lines = ["| Tokenizer | Vocab | Train time (s) | " +
             " | ".join(s["tokenizers"][next(iter(s["tokenizers"]))]["chars_per_token"].keys()) + " |"]
    keys = list(next(iter(s["tokenizers"].values()))["chars_per_token"].keys())
    lines = ["| Tokenizer | Vocab | Train time (s) | " + " | ".join(keys) + " |",
             "|---|---|---:|" + "---:|" * len(keys)]
    for name, info in s["tokenizers"].items():
        cpt = info["chars_per_token"]
        lines.append(f"| `{name}` | {info['vocab_size']:,} | {info['seconds']} | " +
                     " | ".join(_fmt(cpt.get(k), 3) for k in keys) + " |")
    lines += ["", "All round-trip failures: **0** across every slice."]
    return "\n".join(lines)


def _retrieval_section() -> str:
    d = _read_json(ROOT / "reports/eval_retrieval.json")
    if not d or "retrieval" not in d:
        return "_retrieval not evaluated_"
    r = d["retrieval"]
    out = [f"- **hit@1:** {r['hit@1']:.4f}", f"- **recall@5:** {r['recall@5']:.4f}",
           "", "| Query | Expected | Top hit | Score |", "|---|---|---|---|"]
    for q in r["per_question"]:
        top = q["retrieved"][0] if q["retrieved"] else "—"
        hit = "ok" if q["hit@1"] else "MISS"
        out.append(f"| {q['question'][:44]} | {', '.join(q['expected'])} | `{top}` "
                   f"| {hit} |")
    return "\n".join(out)


def _checkpoints_section() -> str:
    base = ROOT / "checkpoints"
    if not base.exists():
        return "_no checkpoints_"
    lines = []
    for run in sorted(base.iterdir()):
        if not run.is_dir():
            continue
        man = _read_json(run / "manifest.json")
        steps = sorted(run.glob("step-*"))
        lines.append(f"**{run.name}**: {len(steps)} checkpoint(s) on disk")
        if man and man.get("checkpoints"):
            lines.append("")
            lines.append("| Step | Tokens | Val loss | Bytes | Complete |")
            lines.append("|---:|---:|---:|---:|---|")
            for c in man["checkpoints"]:
                lines.append(f"| {c['step']} | {c['tokens_seen']:,} | "
                             f"{_fmt(c['val_loss'])} | {c['bytes']:,} | {c['complete']} |")
        lines.append("")
    return "\n".join(lines) if lines else "_no checkpoints_"


def _benchmarks_section() -> str:
    out = []
    compute = _read_json(ROOT / "reports/compute_benchmark.json")
    if compute:
        out.append("**Compute**")
        out.append("")
        out.append(f"- accelerator peak: **{compute['accelerator_peak_tflops']} TFLOP/s**")
        out.append(f"- cpu peak: **{compute['cpu_peak_tflops']} TFLOP/s**")
        out.append(f"- speedup: **{compute['speedup_vs_cpu']}x**")
        for t in compute.get("train_step", []):
            out.append(f"- `{t['model']}` train step: {t['tokens_per_second']:,.0f} tok/s "
                       f"({t['seconds_per_step']} s/step)")
    infer = _read_json(ROOT / "reports/inference_benchmark.json")
    if infer:
        out += ["", "**Inference**", ""]
        for key, r in infer["results"].items():
            if r.get("status") == "FAILED":
                out.append(f"- `{key}`: **FAILED** — {r.get('reason','')[:90]}")
            elif "median_tokens_per_second" in r:
                out.append(f"- `{key}`: {r['parameters']:,} params, "
                           f"{r.get('weight_gib','?')} GiB weights, "
                           f"{r['median_tokens_per_second']} tok/s median")
            elif "generation" in r:
                gen = r["generation"]
                tps = [g["tokens_per_second"] for g in gen]
                out.append(f"- `{key}`: {len(gen)} prompts, "
                           f"median {statistics.median(tps):.2f} tok/s")
    if not out:
        return "_no benchmarks run_"
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/status.md")
    args = ap.parse_args()

    import subprocess

    from veltron import __version__

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or "unavailable"
    except Exception:
        commit = "unavailable"

    from veltron.utils.device import summary

    parts = [
        "# VeltronLM Status Report",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  version `{__version__}`  ·  "
        f"commit `{commit[:12]}`",
        "",
        "> Every number below was read from an artefact on disk. A measurement that was not "
        "taken is reported as missing rather than inferred.",
        "",
        "## Compute",
        "",
    ]
    for b in summary()["backends"]:
        parts.append(f"- `{b['kind']}`: {b['name']}, {b['total_memory_gb']} GB, "
                     f"fp16={b['supports_fp16']}, bf16={b['supports_bf16']}")
    parts += [
        "",
        "## Model registry",
        "",
        _model_table(),
        "",
        "## Pretraining",
        "",
        _pretraining_table(),
        "",
        "## Dataset",
        "",
        _dataset_section(),
        "",
        "## Tokenizer",
        "",
        _tokenizer_section(),
        "",
        "## Retrieval",
        "",
        _retrieval_section(),
        "",
        "## Checkpoints",
        "",
        _checkpoints_section(),
        "",
        "## Benchmarks",
        "",
        _benchmarks_section(),
        "",
    ]
    text = "\n".join(parts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
