"""Dataset build pipeline: acquire -> filter -> mix -> split -> tokenize -> shard.

Produces a versioned dataset directory that a checkpoint can reference:

    datasets/dataset-v1/
        manifest.json        dataset version, hashes, per-source counts, filter funnel
        train/               packed uint16/uint32 token shards + index
        val/
        provenance.jsonl     one record per source with licence and retrieval date

Data versioning is mandatory: a checkpoint records the dataset version and manifest hash,
so "which corpus produced these weights" is answerable without archaeology.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.hashing import sha256_file, sha256_json
from ..utils.logging_utils import get_logger
from .acquire import (
    Document,
    DocumentWriter,
    fetch_all_wikipedia,
    fetch_gutenberg,
    fetch_permissive_code,
)
from .filters import Deduper, FilterStats, process_document

log = get_logger(__name__)

DATASET_VERSION = "dataset-v1"
VAL_FRACTION = 0.005
TEST_FRACTION = 0.002


# ----------------------------------------------------------------------- mixture
@dataclass(frozen=True)
class MixtureComponent:
    """One weighted category in the pretraining mixture."""

    category: str
    weight: float
    label: str
    rationale: str


#: Initial hypothesis for the pretraining mixture. Every weight here is treated as an
#: experimental variable: scripts/exp_mixture.py re-runs the sweep and reports which
#: weighting actually validated best. The mixture is NOT claimed to be optimal.
DEFAULT_MIXTURE: tuple[MixtureComponent, ...] = (
    MixtureComponent("general", 0.34, "en.wikipedia",
                    "Broad knowledge and long-form structure."),
    MixtureComponent("code", 0.22, "permissive-repos (python/flask/rust)",
                    "Syntax, API surfaces, debugging patterns; code is a strong general reasoner."),
    MixtureComponent("technical", 0.12, "project READMEs, API docs, stdlib docstrings",
                    "Instruction-like documentation register needed for support answers."),
    MixtureComponent("general_public_domain", 0.10, "gutenberg literature",
                    "Long-range narrative coherence and stylometry."),
    MixtureComponent("sr", 0.12, "sr.wikipedia (Cyrillic + Latin)",
                    "Primary non-English target language; also teaches multilingual transfer."),
    MixtureComponent("support", 0.10, "veltron synthetic knowledge base + support transcripts",
                    "Customer-support domain register, policy language, escalation phrasing."),
)


def normalize_mixture(mix: Iterable[MixtureComponent]) -> list[MixtureComponent]:
    items = list(mix)
    total = sum(i.weight for i in items)
    if total <= 0:
        raise ValueError("mixture weights must sum to a positive number")
    return [
        MixtureComponent(i.category, i.weight / total, i.label, i.rationale) for i in items
    ]


# ------------------------------------------------------------------------- build
@dataclass
class BuildResult:
    dataset_version: str
    root: Path
    manifest: dict[str, Any]
    train_tokens: int
    val_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "root": str(self.root),
            "train_tokens": self.train_tokens,
            "val_tokens": self.val_tokens,
            "manifest_hash": self.manifest.get("manifest_hash"),
            "documents": self.manifest.get("totals", {}).get("documents_kept"),
        }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def collect_raw(
    out_dir: Path,
    with_wikipedia: bool = True,
    with_code: bool = True,
    with_gutenberg: bool = True,
    with_support: bool = True,
) -> dict[str, Any]:
    """Download every enabled source into per-source raw JSONL files."""
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    counts: dict[str, Any] = {}

    jobs: list[tuple[str, Callable[[], Iterable[Document]]]] = []
    if with_wikipedia:
        jobs.append(("wikipedia", fetch_all_wikipedia))
    if with_code:
        jobs.append(("code_technical", fetch_permissive_code))
    if with_gutenberg:
        jobs.append(("gutenberg", fetch_gutenberg))
    if with_support:
        jobs.append(("support_kb", _support_documents))

    for name, fn in jobs:
        t0 = time.perf_counter()
        path = raw / f"{name}.jsonl"
        writer = DocumentWriter(path)
        try:
            for doc in fn():
                writer.write(doc)
        finally:
            info = writer.close()
        info["seconds"] = round(time.perf_counter() - t0, 1)
        info["megabytes"] = round(info["chars"] / 1e6, 3)
        counts[name] = info
        log.info("acquired %s: %d docs, %.2f Mchars in %ss",
                 name, info["documents"], info["megabytes"], info["seconds"])
    return counts


def _support_documents() -> Iterator[Document]:
    """Load the synthetic product knowledge base as training documents.

    Marked synthetic on every record so a downstream audit can distinguish real product
    policy text from invented demo policy.
    """
    from ..support.knowledge_base import load_knowledge_base

    today = time.strftime("%Y-%m-%d")
    for doc in load_knowledge_base():
        yield Document(
            text=doc["text"],
            source="veltron-synthetic-kb",
            license="CC0-1.0 (project-authored synthetic data)",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            category="support",
            language=doc.get("language", "en"),
            title=doc.get("title", ""),
            url=f"veltron://knowledge_base/{doc.get('doc_id','')}",
            retrieval_date=today,
        )


# ------------------------------------------------------------------------ mixing
def _category_of(rec: dict[str, Any]) -> str:
    src = rec.get("source", "")
    cat = rec.get("category", "")
    lang = rec.get("language", "")
    if cat == "support":
        return "support"
    if cat == "code":
        return "code"
    if cat == "technical":
        return "technical"
    if src == "gutenberg":
        return "general_public_domain"
    if lang == "sr":
        return "sr"
    if src.startswith("wikipedia"):
        return "general"
    return cat or "general"


def mix_documents(
    docs: list[dict[str, Any]],
    mixture: Iterable[MixtureComponent] = DEFAULT_MIXTURE,
    seed: int = 1234,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resample documents to hit the target mixture weights.

    Categories with surplus are down-sampled and categories with deficit are up-sampled
    *with replacement*; the up-sampled count is reported so the report can state how much
    of the corpus is repeated.
    """
    mix = normalize_mixture(mixture)
    rng = random.Random(seed)

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for d in docs:
        by_cat.setdefault(_category_of(d), []).append(d)

    report: dict[str, Any] = {"target": {}, "achieved": {}, "available": {}, "upsampled": {}}
    for c in mix:
        pool = by_cat.get(c.category, [])
        report["available"][c.category] = len(pool)
        report["target"][c.category] = round(c.weight, 4)

    target_size = sum(len(v) for v in by_cat.values())
    out: list[dict[str, Any]] = []
    for c in mix:
        pool = by_cat.get(c.category, [])
        if not pool:
            log.warning("mixture category %r has no documents", c.category)
            continue
        want = int(round(target_size * c.weight))
        if want <= len(pool):
            chosen = rng.sample(pool, want)
            report["upsampled"][c.category] = 0
        else:
            chosen = list(pool) + [rng.choice(pool) for _ in range(want - len(pool))]
            report["upsampled"][c.category] = want - len(pool)
        out.extend(chosen)
        report["achieved"][c.category] = round(len(chosen) / max(1, len(out)), 4)

    rng.shuffle(out)
    total = max(1, len(out))
    report["achieved_final"] = {k: round(v / total, 4) for k, v in
                                ((c, sum(1 for d in out if _category_of(d) == c))
                                 for c in { _category_of(d) for d in out })}
    return out, report


# ------------------------------------------------------------------------- split
def split_documents(
    docs: list[dict[str, Any]],
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = 1234,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by *document*, not by line, so no document straddles two splits."""
    rng = random.Random(seed + 1)
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_fraction)
    n_test = int(len(idx) * test_fraction)
    val = [docs[i] for i in idx[:n_val]]
    test = [docs[i] for i in idx[n_val : n_val + n_test]]
    train = [docs[i] for i in idx[n_val + n_test :]]
    return train, val, test


# ---------------------------------------------------------------------- packing
DOC_SEP = "\n\n<|endoftext|>\n\n"


def pack_documents_list(docs: list[dict[str, Any]], tokenizer: Any, eos_id: int,
                        max_tokens: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    buf: list[int] = []
    for d in docs:
        ids = tokenizer.encode(d["text"], add_special_tokens=False)
        if not ids:
            continue
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= max_tokens:
            out.append(np.asarray(buf[:max_tokens], dtype=np.uint32))
            buf = buf[max_tokens:]
    if buf:
        out.append(np.asarray(buf, dtype=np.uint32))
    return out


def write_shards(shards: list[np.ndarray], out_dir: Path, prefix: str) -> dict[str, Any]:
    """Write token shards and an index recording the *actual* on-disk dtype.

    The dtype must be derived from the arrays, not assumed: a mismatch between the file
    and the index silently reinterprets every token id and produces out-of-bounds embedding
    lookups that look like a model bug.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    total = 0
    dtypes = {str(a.dtype) for a in shards if a.size}
    if len(dtypes) > 1:
        raise ValueError(f"shards for {prefix} have mixed dtypes: {dtypes}")
    dtype_name = dtypes.pop() if dtypes else "uint32"
    for i, arr in enumerate(shards):
        p = out_dir / f"{prefix}-{i:05d}.bin"
        arr.tofile(p)
        paths.append({"path": p.name, "tokens": int(arr.size), "sha256": sha256_file(p)[:16]})
        total += int(arr.size)
    (out_dir / f"{prefix}-index.json").write_text(
        json.dumps({"shards": paths, "total_tokens": total, "dtype": dtype_name}, indent=2),
        encoding="utf-8",
    )
    return {"shards": len(paths), "tokens": total, "dtype": dtype_name,
            "index": f"{prefix}-index.json"}


# --------------------------------------------------------------------- top-level
def build_dataset(
    out_root: str | Path = "datasets",
    version: str = DATASET_VERSION,
    tokenizer: Any = None,
    eos_id: int = 2,
    seq_len: int = 1024,
    mixture: Iterable[MixtureComponent] = DEFAULT_MIXTURE,
    seed: int = 1234,
    min_quality: float = 0.0,
    dedupe: bool = True,
    re_acquire: bool = False,
    with_wikipedia: bool = True,
    with_code: bool = True,
    with_gutenberg: bool = True,
    with_support: bool = True,
) -> BuildResult:
    """Run the full dataset build and write a versioned, hashed artifact."""
    if tokenizer is None:
        raise ValueError("build_dataset requires a trained tokenizer")

    out_root = Path(out_root)
    version_dir = out_root / version
    version_dir.mkdir(parents=True, exist_ok=True)
    raw = out_root / "raw"
    t_start = time.perf_counter()

    # ---- 1. acquire
    need_acquire = re_acquire or not raw.exists() or not any(raw.glob("*.jsonl"))
    acquire_stats = collect_raw(
        raw,
        with_wikipedia=with_wikipedia,
        with_code=with_code,
        with_gutenberg=with_gutenberg,
        with_support=with_support,
    ) if need_acquire else {"reused_existing_raw": True,
                            "files": [p.name for p in raw.glob("*.jsonl")]}

    # ---- 2. filter
    stats = FilterStats()
    deduper = Deduper()
    kept: list[dict[str, Any]] = []
    per_source: dict[str, dict[str, int]] = {}
    provenance: list[dict[str, Any]] = []
    seen_source_meta: dict[str, dict[str, Any]] = {}

    for path in sorted(raw.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_doc = json.loads(line)
                except json.JSONDecodeError:
                    stats.reject("bad_json_line")
                    continue
                cleaned = process_document(raw_doc, stats, deduper, min_quality=min_quality)
                if cleaned is None:
                    continue
                kept.append(cleaned)
                s = cleaned.get("source", "unknown")
                per_source.setdefault(s, {"documents": 0, "chars": 0})
                per_source[s]["documents"] += 1
                per_source[s]["chars"] += cleaned["chars"]
                if s not in seen_source_meta:
                    seen_source_meta[s] = {
                        "source": s,
                        "license": cleaned.get("license"),
                        "license_url": cleaned.get("license_url"),
                        "category": cleaned.get("category"),
                        "retrieval_date": cleaned.get("retrieval_date"),
                        "url_template_example": cleaned.get("url"),
                    }

    provenance = list(seen_source_meta.values())
    log.info("filter funnel: %s", stats.as_dict())

    if not kept:
        raise RuntimeError("no documents survived filtering; dataset build aborted")

    # ---- 3. mix
    mixed, mix_report = mix_documents(kept, mixture, seed=seed)
    log.info("mixture: %s", mix_report["achieved_final"])

    # ---- 4. split (by document)
    train_docs, val_docs, test_docs = split_documents(mixed, seed=seed)

    # ---- 5. pack + shard
    vocab_max = max(tokenizer.get_vocab_size() - 1, 65535)
    np_dtype = np.uint32 if vocab_max > 65535 else np.uint16

    def pack(docs: list[dict[str, Any]]) -> list[np.ndarray]:
        arrs = pack_documents_list(docs, tokenizer, eos_id, seq_len)
        return [a.astype(np_dtype) for a in arrs]

    train_shards = pack(train_docs)
    val_shards = pack(val_docs)
    test_shards = pack(test_docs)

    train_info = write_shards(train_shards, version_dir / "train", "train")
    val_info = write_shards(val_shards, version_dir / "val", "val")
    if test_shards:
        write_shards(test_shards, version_dir / "test", "test")

    _write_jsonl(version_dir / "documents-train.jsonl", train_docs[:2000])
    (version_dir / "provenance.json").write_text(
        json.dumps({"sources": provenance, "per_source_counts": per_source}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ---- 6. manifest
    manifest: dict[str, Any] = {
        "dataset_version": version,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "sequence_length": seq_len,
        "tokenizer": {
            "vocab_size": tokenizer.get_vocab_size(),
            "name": getattr(tokenizer, "name", "unknown"),
            "sha256": getattr(tokenizer, "sha256", "unknown"),
        },
        "totals": {
            "documents_seen": stats.seen,
            "documents_kept": stats.accepted,
            "characters_kept": sum(len(d["text"]) for d in kept),
            "train_documents": len(train_docs),
            "val_documents": len(val_docs),
            "test_documents": len(test_docs),
            "train_tokens": train_info["tokens"],
            "val_tokens": val_info["tokens"],
            "total_tokens": train_info["tokens"] + val_info["tokens"],
            "chars_per_token": round(
                sum(len(d["text"]) for d in mixed) / max(1, train_info["tokens"] + val_info["tokens"]), 3
            ),
        },
        "filtering": stats.as_dict(),
        "dedup": deduper.stats(),
        "mixture": {
            "target": mix_report["target"],
            "achieved": mix_report["achieved_final"],
            "upsampled_documents": mix_report["upsampled"],
            "components": [
                {"category": c.category, "weight": c.weight, "source": c.label, "rationale": c.rationale}
                for c in normalize_mixture(mixture)
            ],
        },
        "per_source": per_source,
        "provenance": provenance,
        "acquisition": acquire_stats,
        "shards": {"train": train_info, "val": val_info},
        "build_seconds": round(time.perf_counter() - t_start, 1),
        "splits": {"val_fraction": VAL_FRACTION, "test_fraction": TEST_FRACTION},
    }
    manifest["manifest_hash"] = sha256_json({k: v for k, v in manifest.items()
                                             if k not in ("built_at", "build_seconds")})
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(
        "dataset %s built: %d docs, %s train tokens, %s val tokens in %ss",
        version, stats.accepted, f"{train_info['tokens']:,}", f"{val_info['tokens']:,}",
        manifest["build_seconds"],
    )
    return BuildResult(version, version_dir, manifest, train_info["tokens"], val_info["tokens"])


__all__ = [
    "DATASET_VERSION",
    "MixtureComponent",
    "DEFAULT_MIXTURE",
    "normalize_mixture",
    "build_dataset",
    "collect_raw",
    "mix_documents",
    "split_documents",
    "write_shards",
    "pack_documents_list",
    "BuildResult",
]
