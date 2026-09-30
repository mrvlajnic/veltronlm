"""Background Wikipedia category crawl.

Runs detached from the main session so corpus growth and code development overlap.
Writes datasets/raw/wikipedia_categories.jsonl and reports progress to a log file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from veltron.data.acquire import DocumentWriter
from veltron.data.wikipedia_categories import fetch_all_categories
from veltron.utils.logging_utils import setup_logging

OUT = Path("datasets/raw")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles-per-category", type=int, default=400)
    ap.add_argument("--max-docs-per-lang", type=int, default=2000)
    ap.add_argument("--languages", nargs="*", default=["en", "sr", "sh"])
    ap.add_argument("--out", default=str(OUT / "wikipedia_categories.jsonl"))
    args = ap.parse_args()
    setup_logging(level="INFO")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = DocumentWriter(out)
    t0 = time.perf_counter()
    n = 0
    try:
        for doc in fetch_all_categories(args.titles_per_category,
                                        tuple(args.languages),
                                        args.max_docs_per_lang):
            w.write(doc)
            n += 1
            if n % 200 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {n} docs, "
                      f"{w.chars/1e6:.2f} Mchars, {time.perf_counter()-t0:.0f}s", flush=True)
    finally:
        info = w.close()
    info["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(info, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
