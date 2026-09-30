"""Train the VeltronLM byte-level BPE tokenizers.

Produces two vocabularies because the size is dictated by the model, not by taste:

* ``tok-mini-32k`` (32768) -- used by the models trainable on this host. A 64k embedding
  would dominate a 7M-parameter test model (16.8M embedding rows), so small models must
  not share the 4B vocabulary.
* ``tok-4b-64k`` (65536) -- the vocabulary the 4B architecture was sized for.

Both are trained on a *stratified sample* of the real corpus so that Serbian Cyrillic,
Serbian Latin, code, JSON and support prose all appear in the merge statistics. A
tokenizer trained on English alone shatters Cyrillic into single bytes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from veltron.tokenizer.trainer import TokenizerConfig, train_tokenizer
from veltron.utils.logging_utils import setup_logging

RAW = Path("datasets/raw")

#: Per-bucket character budget for the training sample. Weights are hypotheses to be
#: ablated by scripts/exp_tokenizer.py, not a settled optimum.
STRATA: list[tuple[str, int]] = [
    ("gutenberg_en", 6_000_000),
    ("wikipedia_en", 1_200_000),
    ("wikipedia_sr_cyrl", 800_000),
    ("wikipedia_sr_latin", 800_000),
    ("wikipedia_categories", 1_200_000),
    ("code", 2_500_000),
    ("support", 400_000),
]


def read_jsonl_texts(path: Path) -> list[str]:
    out: list[str] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text")
            if isinstance(t, str) and t:
                out.append(t)
    return out


def stratified_sample(seed: int = 1234) -> tuple[list[str], dict]:
    """Draw a representative sample, reporting what each stratum actually contributed."""
    rng = random.Random(seed)

    # Deterministic shuffle first, then take documents until the budget is spent. Working
    # in this order makes the cap trivially correct: the loop can add at most one more
    # document past the budget, and the selection is reproducible for a given seed.
    texts: list[str] = []
    report: dict = {"strata": {}, "total_chars": 0}

    buckets: dict[str, list[str]] = {
        "gutenberg_en": read_jsonl_texts(RAW / "gutenberg.jsonl"),
        "wikipedia_en": [],
        "wikipedia_sr_cyrl": [],
        "wikipedia_sr_latin": [],
        "wikipedia_categories": [],
        "code": read_jsonl_texts(RAW / "code_technical.jsonl"),
        "support": read_jsonl_texts(RAW / "support_kb.jsonl"),
    }

    for p in (RAW / "wikipedia.jsonl", RAW / "wikipedia_categories.jsonl"):
        is_cat = p.name.startswith("wikipedia_categories")
        for t in read_jsonl_texts(p):
            if is_cat:
                buckets["wikipedia_categories"].append(t)
                continue
            head = t[:600]
            is_cyrillic = any("Ѐ" <= ch <= "ӿ" for ch in head)
            # Serbian in Latin script uses diacritics that Cyrillic text never contains.
            is_latin_sr = sum(ch in "čćžšđČĆŽŠĐ" for ch in head) >= 2
            if is_cyrillic:
                buckets["wikipedia_sr_cyrl"].append(t)
            elif is_latin_sr:
                buckets["wikipedia_sr_latin"].append(t)
            else:
                buckets["wikipedia_en"].append(t)

    for name, budget in STRATA:
        pool = list(buckets.get(name, []))
        rng.shuffle(pool)
        picked: list[str] = []
        used = 0
        for t in pool:
            if used + len(t) > budget:
                # Skip documents that would overshoot; keep scanning, since many short
                # documents still fit into what is left of the budget.
                continue
            picked.append(t)
            used += len(t)
        if not picked and pool:
            # Every available document exceeds the budget: take the smallest one so the
            # stratum is represented at all rather than silently dropped.
            smallest = min(pool, key=len)
            picked = [smallest]
            used = len(smallest)
        texts.extend(picked)
        report["strata"][name] = {
            "available_documents": len(pool),
            "sampled_documents": len(picked),
            "sampled_chars": used,
            "budget": budget,
        }
        report["total_chars"] += used

    rng.shuffle(texts)
    return texts, report


def eval_samples(n: int = 120) -> dict[str, list[str]]:
    """Build labelled slices for compression measurement, drawn from the raw corpora.

    Slices are taken from the *raw* files rather than the training sample: the sample is
    deliberately budget-capped and skewed toward long documents, so measuring on it would
    understate coverage of the languages and formats that matter most (Serbian in both
    scripts, code, JSON, markdown).
    """
    slices: dict[str, list[str]] = {
        "en_prose": [], "sr_cyrillic": [], "sr_latin": [],
        "python": [], "rust_c": [], "technical_markdown": [], "json_structured": [],
    }

    def add(key: str, text: str) -> None:
        if len(slices[key]) < n and len(text) > 400:
            slices[key].append(text[:8000])

    # English prose from two independent sources.
    for t in read_jsonl_texts(RAW / "gutenberg.jsonl")[:400]:
        add("en_prose", t)
        if len(slices["en_prose"]) >= n:
            break

    for p in (RAW / "wikipedia.jsonl", RAW / "wikipedia_categories.jsonl"):
        for t in read_jsonl_texts(p):
            head = t[:600]
            if any("Ѐ" <= ch <= "ӿ" for ch in head):
                add("sr_cyrillic", t)
            elif sum(ch in "čćžšđČĆŽŠĐ" for ch in head) >= 2:
                add("sr_latin", t)

    # Support corpus includes a Serbian FAQ, guaranteeing a Latin-script Serbian slice
    # even when Wikipedia yields none.
    for t in read_jsonl_texts(RAW / "support_kb.jsonl"):
        if sum(ch in "čćžšđČĆŽŠĐ" for ch in t[:2000]) >= 2:
            add("sr_latin", t)
        if any("Ѐ" <= ch <= "ӿ" for ch in t[:2000]):
            add("sr_cyrillic", t)

    for t in read_jsonl_texts(RAW / "code_technical.jsonl"):
        if "python/cpython" in t[:400] or t.startswith(("def ", "class ", "import ", '"""')):
            add("python", t)
        elif t.startswith(("fn ", "pub ", "struct ", "impl ", "//", "use ", "#[")):
            add("rust_c", t)
        elif t.lstrip().startswith(("{", "[")) and t.count("\n") < 0.1 * len(t):
            add("json_structured", t)
        elif t.lstrip().startswith("#") or "\n## " in t:
            add("technical_markdown", t)
        if all(len(v) >= n for v in slices.values()):
            break

    return {k: v for k, v in slices.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="models")
    ap.add_argument("--sizes", nargs="*", type=int, default=[32768, 65536])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cache", default="datasets/tokenizer_train_sample.jsonl")
    args = ap.parse_args()
    setup_logging(level="INFO")

    # Materialise the stratified sample once so both vocabularies see identical text.
    cache = Path(args.cache)
    if cache.exists():
        texts = read_jsonl_texts(cache)
        report = json.loads(cache.with_suffix(".report.json").read_text(encoding="utf-8"))
        print(f"loaded cached sample: {len(texts)} texts, {report['total_chars']:,} chars")
    else:
        t0 = time.perf_counter()
        texts, report = stratified_sample(args.seed)
        print(f"stratified sample: {len(texts)} texts, {report['total_chars']:,} chars "
              f"in {time.perf_counter()-t0:.1f}s")
        for k, v in report["strata"].items():
            print(f"  {k:26} docs={v['available_documents']:>5} sampled={v['sampled_chars']:>10,}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w", encoding="utf-8") as fh:
            for t in texts:
                fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        cache.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not texts:
        print("ERROR: no training text available; run scripts/acquire_data.py first")
        return 1

    slices = eval_samples()
    summary: dict = {"sample": report, "tokenizers": {}}

    for vocab in args.sizes:
        name = "tok-mini-32k" if vocab <= 32768 else f"tok-4b-{vocab // 1024}k"
        out_dir = Path(args.out_root) / name
        print("\n" + "=" * 78)
        print(f"TRAINING {name}  vocab_size={vocab}")
        print("=" * 78)
        cfg = TokenizerConfig(vocab_size=vocab, min_frequency=2, name=name)
        t0 = time.perf_counter()
        tok, stats = train_tokenizer([cache], out_dir, cfg)
        elapsed = time.perf_counter() - t0

        eff = tok.efficiency(slices)
        print(f"\n{name}: vocab={stats['vocab_size']} trained in {elapsed:.1f}s")
        print(f"{'slice':12}{'docs':>6}{'chars':>10}{'tokens':>10}{'chars/tok':>11}{'tok/word':>10}{'fail':>6}")
        for label, rows in eff.items():
            r = rows
            print(f"{label:12}{r['samples']:>6}{r['chars']:>10,}{r['tokens']:>10,}"
                  f"{r['chars_per_token']:>11.3f}{r['tokens_per_word']:>10.3f}{r['roundtrip_failures']:>6}")

        report_path = Path("reports") / f"tokenizer-{name}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name,
            "vocab_size": stats["vocab_size"],
            "vocab_requested": vocab,
            "training_seconds": round(elapsed, 1),
            "training_stats": stats,
            "efficiency": eff,
            "sample_report": report,
            "special_token_ids": tok.special_ids(),
        }
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["tokenizers"][name] = {
            "vocab_size": stats["vocab_size"],
            "seconds": round(elapsed, 1),
            "chars_per_token": {k: v["chars_per_token"] for k, v in eff.items()},
        }

    print("\n" + "=" * 78)
    print(json.dumps(summary["tokenizers"], indent=2))
    Path("reports/tokenizer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
