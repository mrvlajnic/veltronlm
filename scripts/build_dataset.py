"""Build a versioned, tokenized, sharded dataset from the raw corpus.

Runs the full pipeline: filter -> PII handling -> dedup -> mixture -> document-level
split -> packing -> sharding -> manifest with provenance and content hash.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from veltron.data.pipeline import (
    DEFAULT_MIXTURE,
    MixtureComponent,
    build_dataset,
    normalize_mixture,
)
from veltron.tokenizer.trainer import VeltronTokenizer
from veltron.utils.logging_utils import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a VeltronLM dataset version")
    ap.add_argument("--version", default="dataset-v1")
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--tokenizer", default="models/tok-mini-32k")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-quality", type=float, default=0.0)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--re-acquire", action="store_true")
    ap.add_argument("--mixture", default="default",
                    help="default | no-sr | code-heavy | literature-heavy | <json path>")
    args = ap.parse_args()
    setup_logging(level="INFO")

    tok = VeltronTokenizer.load(args.tokenizer)
    print(f"tokenizer {tok.name} vocab={tok.get_vocab_size()} sha256={tok.sha256}")
    print(f"  eos_id={tok.eos_id} bos_id={tok.bos_id} pad_id={tok.pad_id}")
    if tok.get_vocab_size() > 32768:
        print("  NOTE: model configs for the 4B tier use vocab_size=65536")

    mixture = DEFAULT_MIXTURE
    if args.mixture == "no-sr":
        mixture = tuple(m for m in DEFAULT_MIXTURE if m.category != "sr")
    elif args.mixture == "code-heavy":
        mixture = tuple(
            MixtureComponent(m.category,
                            (0.55 if m.category == "code" else
                             0.10 if m.category in ("support", "general_public_domain") else
                             m.weight * 0.25),
                            m.label, m.rationale)
            for m in DEFAULT_MIXTURE
        )
    elif args.mixture == "literature-heavy":
        mixture = tuple(
            MixtureComponent(m.category,
                            (0.60 if m.category == "general_public_domain" else m.weight * 0.2),
                            m.label, m.rationale)
            for m in DEFAULT_MIXTURE
        )
    elif args.mixture != "default":
        raw = json.loads(Path(args.mixture).read_text(encoding="utf-8"))
        mixture = tuple(MixtureComponent(r["category"], float(r["weight"]), r.get("label", ""),
                                         r.get("rationale", "")) for r in raw)

    print("\nmixture:")
    for c in normalize_mixture(mixture):
        print(f"  {c.category:22} {c.weight:6.3f}  {c.label}")

    t0 = time.perf_counter()
    result = build_dataset(
        out_root=args.out_root,
        version=args.version,
        tokenizer=tok,
        eos_id=tok.eos_id,
        seq_len=args.seq_len,
        mixture=mixture,
        seed=args.seed,
        min_quality=args.min_quality,
        dedupe=not args.no_dedupe,
        re_acquire=args.re_acquire,
    )
    elapsed = time.perf_counter() - t0

    m = result.manifest
    print("\n" + "=" * 78)
    print(f"DATASET {args.version} BUILT in {elapsed:.1f}s")
    print("=" * 78)
    t = m["totals"]
    print(f"  documents seen        {t['documents_seen']:>12,}")
    print(f"  documents kept        {t['documents_kept']:>12,}")
    print(f"  characters kept       {t['characters_kept']:>12,}")
    print(f"  train tokens          {t['train_tokens']:>12,}")
    print(f"  val tokens            {t['val_tokens']:>12,}")
    print(f"  chars per token       {t['chars_per_token']:>12.3f}")
    print(f"  manifest hash         {m['manifest_hash'][:16]}")

    print("\n  filter funnel:")
    for stage, n in m["filtering"]["rejected_by_stage"].items():
        print(f"    {stage:34} {n:>10,}")

    print("\n  dedup:")
    for k, v in m["dedup"].items():
        print(f"    {k:34} {v}")

    print("\n  mixture achieved:")
    for cat, frac in sorted(m["mixture"]["achieved"].items(), key=lambda kv: -kv[1]):
        target = m["mixture"]["target"].get(cat, 0)
        print(f"    {cat:34} {frac:6.3f} (target {target:.3f})")

    ups = {k: v for k, v in m["mixture"]["upsampled_documents"].items() if v}
    if ups:
        print("\n  up-sampled (repeated) documents per category:")
        for k, v in ups.items():
            print(f"    {k:34} {v:>10,}")

    print("\n  per-source documents retained:")
    for src, info in sorted(m["per_source"].items(), key=lambda kv: -kv[1]["chars"]):
        print(f"    {src:34} {info['documents']:>6} docs {info['chars']/1e6:>8.2f} Mchars")

    print(f"\n  artifact: {result.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
