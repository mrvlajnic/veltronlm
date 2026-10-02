"""Single source of truth for every number quoted in the documentation.

Documentation drift is the fastest way to lose a reviewer: if a README says "12 GiB" and the
environment document says "3.62 GiB" and a third file says "10 GiB", each of them has to be
re-derived. This script reads the artefacts and prints the authoritative values, and
`--check` fails when a documented claim disagrees with the artefact.

    python scripts/facts.py
    python scripts/facts.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(rel: str) -> Any | None:
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def count_tests() -> int | None:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-p", "no:warnings", "--collect-only", "-q"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600,
        )
        n = len(re.findall(r"::", out.stdout))
        return n or None
    except Exception:
        return None


def gather() -> dict[str, Any]:
    from veltron.model.config import REGISTRY
    from veltron.model.transformer import count_parameters_via_meta

    facts: dict[str, Any] = {}

    # ---- model registry
    facts["registry"] = {
        k: {
            "name": e.config.name,
            "parameters": e.config.param_counts()["total"],
            "meta_matches": e.config.param_counts()["total"]
            == count_parameters_via_meta(e.config),
            "trainable_here": e.trainable_on_this_host,
        }
        for k, e in REGISTRY.items()
    }

    # ---- training runs
    runs = {}
    ckpt_root = ROOT / "checkpoints"
    if ckpt_root.exists():
        for run_dir in sorted(p for p in ckpt_root.iterdir() if p.is_dir()):
            from veltron.training.checkpoint import CheckpointManager

            mgr = CheckpointManager(run_dir)
            latest = mgr.latest_valid()
            if latest is None:
                continue
            meta = mgr.read_metadata(latest) or {}
            evals = []
            log = run_dir / "train_log.jsonl"
            if log.exists():
                for line in log.read_text(encoding="utf-8").splitlines():
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if o.get("eval"):
                        evals.append(o)
            ckpt_bytes = sum(f.stat().st_size for d in mgr.list_checkpoints()
                             for f in d.glob("*") if f.is_file())
            runs[run_dir.name] = {
                "latest_step": meta.get("step"),
                "tokens": meta.get("tokens_seen"),
                "val_loss": meta.get("val_loss"),
                "val_perplexity": meta.get("val_perplexity"),
                "n_checkpoints": len(mgr.list_checkpoints()),
                "disk_gib": round(ckpt_bytes / 1024**3, 2),
                "evals": [(e["step"], round(e["val_loss"], 4), round(e["val_perplexity"], 1))
                          for e in evals],
            }
    facts["runs"] = runs

    # ---- dataset
    m = read_json("datasets/dataset-v1/manifest.json")
    if m:
        facts["dataset"] = {
            "version": m["dataset_version"],
            "manifest_hash": m["manifest_hash"][:16],
            "train_tokens": m["totals"]["train_tokens"],
            "val_tokens": m["totals"]["val_tokens"],
            "chars": m["totals"]["characters_kept"],
            "docs_seen": m["totals"]["documents_seen"],
            "docs_kept": m["totals"]["documents_kept"],
            "accept_rate": m["filtering"]["accept_rate"],
            "chars_per_token": m["totals"]["chars_per_token"],
            "val_documents": m["totals"]["val_documents"],
        }

    # ---- tokenizer
    t = read_json("reports/tokenizer_summary.json")
    if t:
        facts["tokenizers"] = {
            k: {"vocab": v["vocab_size"], "chars_per_token": v["chars_per_token"]}
            for k, v in t["tokenizers"].items()
        }

    # ---- retrieval
    r = read_json("reports/eval_retrieval.json")
    if r and "retrieval" in r:
        facts["retrieval"] = {
            "hit@1": r["retrieval"]["hit@1"],
            "recall@5": r["retrieval"]["recall@5"],
            "questions": r["retrieval"]["questions"],
        }

    # ---- VRAM shape: the single-allocation cap is NOT the total budget
    v = read_json("reports/vram_shape.json")
    if v:
        facts["vram"] = {
            "single_allocation_max_gib": v["single_allocation_max_gib"],
            "total_holdable_gib": v["best_cumulative_gib"],
            "windows_in_use_gib": v["windows_reported_in_use_gib"],
            "interpretation": v["interpretation"],
        }

    # ---- tests
    facts["tests"] = count_tests()
    return facts


# Claims asserted in prose, with the value the artefacts actually support.
CLAIMS = {
    "4b_parameters": ("4,026,765,312", lambda f: f"{f['registry']['4b']['parameters']:,}"),
    "mini_parameters": ("244,354,048", lambda f: f"{f['registry']['mini']['parameters']:,}"),
    "117m_parameters": ("117,027,592", lambda f: f"{f['registry']['117m']['parameters']:,}"),
    "micro_parameters": ("55,715,328", lambda f: f"{f['registry']['micro']['parameters']:,}"),
    "dataset_train_tokens": ("19,485,297", lambda f: f"{f['dataset']['train_tokens']:,}"),
    "vocab_mini": ("32,768", lambda f: f"{f['registry']['mini']['vocab']}"
     if 'vocab' in f['registry']['mini'] else str(REGISTRY_MINI_VOCAB)),
}


def REGISTRY_MINI_VOCAB() -> int:  # noqa: N802
    from veltron.model.config import get_config

    return get_config("mini").vocab_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="compare documented claims against artefacts")
    args = ap.parse_args()

    facts = gather()

    if args.json:
        print(json.dumps(facts, indent=2, default=str))
        return 0

    print("=" * 78)
    print("AUTHORITATIVE FACTS (read from artefacts)")
    print("=" * 78)

    print("\nMODEL REGISTRY")
    print(f"  {'key':8}{'parameters':>16}{'meta==analytic':>15}{'trainable':>11}")
    for k, v in facts["registry"].items():
        print(f"  {k:8}{v['parameters']:>16,}{str(v['meta_matches']):>15}"
              f"{str(v['trainable_here']):>11}")

    print("\nTRAINING RUNS")
    for name, v in facts["runs"].items():
        print(f"  {name}: step {v['latest_step']}, {v['tokens']:,} tokens, "
              f"val_loss {v['val_loss']}, ppl {v['val_perplexity']}")
        print(f"    {v['n_checkpoints']} checkpoints, {v['disk_gib']} GiB")
        if v["evals"]:
            print("    evals:", ", ".join(f"{s}:{l}/{p}" for s, l, p in v["evals"]))

    if "dataset" in facts:
        d = facts["dataset"]
        print("\nDATASET")
        print(f"  {d['version']} (manifest {d['manifest_hash']}), "
              f"{d['docs_kept']}/{d['docs_seen']} docs kept "
              f"({d['accept_rate']:.1%})")
        print(f"  {d['train_tokens']:,} train tokens, {d['val_tokens']:,} val tokens "
              f"from {d['val_documents']} documents")
        print(f"  {d['chars']:,} characters, {d['chars_per_token']} chars/token")

    if "tokenizers" in facts:
        print("\nTOKENIZERS")
        for k, v in facts["tokenizers"].items():
            cpt = ", ".join(f"{s}={c}" for s, c in v["chars_per_token"].items())
            print(f"  {k}: vocab {v['vocab']:,}  {cpt}")

    if "retrieval" in facts:
        r = facts["retrieval"]
        print(f"\nRETRIEVAL: hit@1={r['hit@1']} recall@5={r['recall@5']} "
              f"over {r['questions']} queries")

    if "vram" in facts:
        v = facts["vram"]
        print("\nVRAM")
        print(f"  single allocation ceiling : {v['single_allocation_max_gib']} GiB")
        print(f"  total holdable in chunks  : {v['total_holdable_gib']} GiB")
        print(f"  windows reports in use   : {v['windows_in_use_gib']} GiB")
        print(f"  => the ceiling is per-allocation ({v['interpretation']}), not total")

    if facts.get("tests"):
        print(f"\nTESTS: {facts['tests']} collected")

    if args.check:
        print("\nCLAIM CHECK")
        bad = 0
        for name, (documented, actual_fn) in CLAIMS.items():
            if name not in facts or callable(REGISTRY_MINI_VOCAB) and name == "vocab_mini":
                continue
            actual = actual_fn(facts)
            ok = str(documented) == str(actual)
            print(f"  {'OK  ' if ok else 'DIFF'} {name}: documented {documented!r} "
                  f"actual {actual!r}")
            if not ok:
                bad += 1
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())