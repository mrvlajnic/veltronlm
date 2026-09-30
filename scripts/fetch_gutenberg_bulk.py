"""Bulk Project Gutenberg acquisition driven by the real catalogue CSV.

The hard-coded ID list in :mod:`veltron.data.acquire` only reaches ~40 books. The public
``pg_catalog.csv`` carries every catalogue entry, so this script derives a much larger,
reproducible set from it instead of hand-listing IDs.

Book selection is deterministic (sorted by ID, filtered by type/language) so two runs
fetch the same corpus and the dataset hash stays stable.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import urllib.request
from pathlib import Path

from veltron.data.acquire import USER_AGENT, Document, DocumentWriter, http_get, strip_gutenberg
from veltron.utils.logging_utils import setup_logging

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
OUT = Path("datasets/raw/gutenberg.jsonl")


def fetch_catalog() -> list[dict[str, str]]:
    req = urllib.request.Request(CATALOG_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    # The catalogue uses "Text#" as the ebook id, not "id". Titles containing embedded
    # newlines are quoted across lines, so csv.DictReader is required over a split.
    reader = csv.DictReader(io.StringIO(text))
    rows = [r for r in reader if r.get("Text#")]
    print(f"catalog rows with text: {len(rows)}", flush=True)
    return rows


#: Subjects that make a book useful as pretraining text. Gutenberg's catalogue is mostly
#: pamphlets and sermons; ranking on these keeps the corpus readable rather than a wall of
#: one-page documents.
GOOD_SUBJECTS = (
    "fiction", "novel", "poetry", "poems", "drama", "play", "essays", "history",
    "travel", "science", "philosophy", "psychology", "economics", "sociology",
    "education", "law", "medicine", "biology", "chemistry", "mathematics",
    "philosophy", "religion", "art", "music", "memoir", "autobiography",
    "children", "juvenile", "adventure", "romance", "detective",
)
BAD_SUBJECTS = ("bill", "resolution", "declaration", "constitution", "hearing",
                "speech", "circular", "bulletin", "report of", "testament")


def _is_useful(row: dict[str, str]) -> bool:
    subjects = (row.get("Subjects") or "").lower() + " " + (row.get("Bookshelves") or "").lower()
    title = (row.get("Title") or "").lower()
    if any(b in subjects or b in title for b in BAD_SUBJECTS):
        return False
    return any(g in subjects for g in GOOD_SUBJECTS)


def pick_ids(rows: list[dict[str, str]], limit: int) -> list[int]:
    """Deterministically select book IDs, preferring substantial, readable texts."""
    out: list[int] = []
    seen: set[int] = set()
    usable = [r for r in rows if _is_useful(r) and (r.get("Language") or "en").startswith("en")]
    ordered = sorted(usable, key=lambda r: (-len(r.get("Text#", "")), int(r["Text#"])))
    for r in ordered:
        try:
            bid = int(r["Text#"])
        except ValueError:
            continue
        if bid in seen:
            continue
        seen.add(bid)
        out.append(bid)
        if len(out) >= limit:
            break
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=900, help="number of books to fetch")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    setup_logging(level="WARNING")

    rows = fetch_catalog()
    ids = pick_ids(rows, args.limit)
    print(f"selected {len(ids)} book ids")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    today = time.strftime("%Y-%m-%d")

    w = DocumentWriter(out)
    t0 = time.perf_counter()
    n = 0
    chars = 0
    for i, gid in enumerate(ids):
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        status, body = http_get(url)
        if status != 200 or len(body) < 4000:
            continue
        try:
            text = strip_gutenberg(body.decode("utf-8", errors="replace"))
        except Exception:
            continue
        if len(text) < 2000:
            continue
        n += 1
        chars += len(text)
        w.write(Document(
            text=text,
            source="gutenberg",
            license="PublicDomain",
            license_url="https://www.gutenberg.org/policy/license.html",
            category="general",
            language="en",
            title=f"gutenberg-{gid}",
            url=f"https://www.gutenberg.org/ebooks/{gid}",
            retrieval_date=today,
        ))
        if n % 50 == 0:
            print(f"  {n} books, {chars/1e6:.1f} Mchars, {time.perf_counter()-t0:.0f}s")

    info = w.close()
    info["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"DONE {info['documents']} books, {info['chars']/1e6:.1f} Mchars in {info['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
