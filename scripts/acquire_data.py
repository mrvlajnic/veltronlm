"""Acquire the raw corpus from every declared source.

Writes datasets/raw/<source>.jsonl. Safe to re-run: existing files are reused unless
--force is passed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from veltron.data.acquire import (
    Document,
    DocumentWriter,
    fetch_all_wikipedia,
    fetch_gutenberg,
    fetch_permissive_code,
)
from veltron.data.sources import licence_report
from veltron.utils.logging_utils import setup_logging

OUT = Path("datasets/raw")


def support_docs() -> list[Document]:
    from veltron.support.knowledge_base import load_knowledge_base

    today = time.strftime("%Y-%m-%d")
    out = []
    for d in load_knowledge_base():
        out.append(Document(
            text=d["text"], source="veltron-synthetic-kb",
            license="CC0-1.0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            category="support", language=d.get("language", "en"),
            title=d.get("title", ""), url=f"veltron://kb/{d['doc_id']}",
            retrieval_date=today,
        ))
    return out


def run(name: str, fn, limit=None) -> dict:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    w = DocumentWriter(path)
    try:
        for d in fn():
            w.write(d)
    finally:
        info = w.close()
    info["seconds"] = round(time.perf_counter() - t0, 1)
    info["megabytes"] = round(info["chars"] / 1e6, 3)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="Acquire VeltronLM pretraining sources")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="source names to skip")
    ap.add_argument("--wikipedia-random", type=int, default=0,
                    help="random Wikipedia article sweep requests per language (0=curated only)")
    args = ap.parse_args()
    setup_logging()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("VELTRONLM SOURCE LICENCE REGISTER (checked before download)")
    print("=" * 78)
    for row in licence_report():
        print(f"  {row['key']:18} {row['license']:22} commercial={row['commercial_use']:10} "
              f"redistribution={row['redistribution']:12} attribution={row['attribution_required']}")
    print()

    results = {}
    jobs: list[tuple[str, object]] = [
        ("wikipedia", lambda: fetch_all_wikipedia(args.wikipedia_random)),
        ("code_technical", fetch_permissive_code),
        ("gutenberg", fetch_gutenberg),
        ("support_kb", lambda: support_docs()),
    ]
    total_chars = 0
    for name, fn in jobs:
        if name in args.skip:
            print(f"  skip {name}")
            continue
        target = out / f"{name}.jsonl"
        if target.exists() and not args.force:
            n = sum(1 for _ in target.open("r", encoding="utf-8"))
            chars = target.stat().st_size
            print(f"  reuse {name}: {n} docs ({chars/1e6:.2f} MB)")
            results[name] = {"documents": n, "reused": True}
            total_chars += chars
            continue
        info = run(name, fn)
        results[name] = info
        total_chars += info["chars"]
        print(f"  fetched {name}: {info['documents']} docs, {info['megabytes']:.2f} Mchars, "
              f"{info['seconds']}s")

    print()
    print(f"TOTAL: {total_chars/1e6:.2f} Mcharacters of raw text")
    print(f"Raw directory: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
