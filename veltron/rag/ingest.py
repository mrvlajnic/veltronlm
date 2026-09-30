"""Document ingestion and chunking.

Chunks are structure-aware: headings, list items and code blocks become their own units
with a breadcrumb of the section path. That matters for support answers, where the useful
span is usually a numbered procedure under "Turnaround times" rather than a paragraph that
happens to contain the right numbers.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..utils.logging_utils import get_logger

log = get_logger(__name__)

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_BULLET = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")


@dataclass
class Chunk:
    """One retrievable unit."""

    chunk_id: str
    doc_id: str
    text: str
    title: str
    section_path: str
    category: str
    language: str = "en"
    tags: tuple[str, ...] = ()
    ordinal: int = 0
    char_start: int = 0
    char_end: int = 0
    synthetic: bool = True
    disclaimer: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def chunk_document(
    doc: dict[str, Any],
    max_chars: int = 1100,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Split one document into heading-aware chunks.

    Heading structure is preserved as a ``section_path`` breadcrumb so a retrieved chunk
    can cite "Warranty and RMA > Not covered" instead of an opaque chunk id.
    """
    doc_id = doc.get("doc_id") or _stable_id(doc.get("title", ""), doc.get("text", "")[:256])
    title = doc.get("title", "") or doc_id
    body = doc.get("body") or doc.get("text", "") or ""
    tags = tuple(doc.get("tags", ()) or ())
    category = doc.get("category", "unknown")
    language = doc.get("language", "en")

    sections: list[tuple[str, str]] = []       # (section_path, text)
    path: list[str] = []
    buf: list[str] = []
    cur_path = "(preamble)"
    in_code = False
    offset = 0

    def flush() -> None:
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            sections.append((cur_path, text))
        buf = []

    for line in body.split("\n"):
        offset += len(line) + 1
        fence = line.strip().startswith("```")
        if fence:
            in_code = not in_code
            buf.append(line)
            continue
        m = _HEADING.match(line)
        if m and not in_code:
            flush()
            level = len(m.group(1))
            text = m.group(2).strip()
            path = path[: level - 1] + [text]
            cur_path = " > ".join(path)
            continue
        buf.append(line)
    flush()

    # Fall back to fixed windows when a document has no heading structure at all.
    if len(sections) <= 1 and sections and len(sections[0][1]) > max_chars:
        flat = sections[0][1]
        sections = [("(whole document)", flat[i : i + max_chars])
                    for i in range(0, len(flat), max(max_chars - overlap_chars, 1))]

    chunks: list[Chunk] = []
    ordinal = 0
    cursor = 0
    for sec_path, text in sections:
        start = 0
        while start < len(text):
            piece = text[start : start + max_chars]
            if not piece.strip():
                break
            cid = _stable_id(doc_id, sec_path, str(start), piece[:64])
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    text=piece.strip(),
                    title=title,
                    section_path=sec_path,
                    category=category,
                    language=language,
                    tags=tags,
                    ordinal=ordinal,
                    char_start=cursor + start,
                    char_end=cursor + start + len(piece),
                    synthetic=bool(doc.get("synthetic", True)),
                    disclaimer=doc.get("disclaimer", ""),
                )
            )
            ordinal += 1
            if start + max_chars >= len(text):
                break
            # Overlap keeps procedures that straddle a boundary retrievable from either side.
            step = max(1, max_chars - overlap_chars)
            nxt = start + step
            if nxt <= start:
                break
            start = nxt
        cursor += len(text) + 2
    return chunks


# ------------------------------------------------------------------- file loaders
def load_md(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    title = p.stem.replace("-", " ").replace("_", " ").title()
    for line in text.split("\n"):
        m = _HEADING.match(line)
        if m:
            title = m.group(2).strip()
            break
    return {
        "doc_id": _stable_id(str(p)),
        "title": title,
        "text": text,
        "category": p.parent.name or "docs",
        "language": _guess_language(text),
        "tags": tuple(_extract_tags(text)),
        "source_path": str(p),
    }


def _extract_tags(text: str) -> list[str]:
    return re.findall(r"(?m)^tags?:\s*(.+)$", text)[:1] + \
        [w.lower() for w in re.findall(r"`([^`]{3,32})`", text)][:12]


def _guess_language(text: str) -> str:
    from ..data.filters import detect_language

    lang, conf = detect_language(text[:4000])
    return lang if lang in ("en", "sr") else "en"


def load_txt(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return {
        "doc_id": _stable_id(str(p)),
        "title": p.stem.replace("_", " ").replace("-", " ").title(),
        "text": p.read_text(encoding="utf-8", errors="replace"),
        "category": p.parent.name or "docs",
        "language": _guess_language(p.read_text(encoding="utf-8", errors="replace")[:4000]),
        "tags": (),
        "source_path": str(p),
    }


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("documents", [])
    out = []
    for item in items:
        item = dict(item)
        item.setdefault("doc_id", _stable_id(item.get("title", ""), str(item.get("body", ""))[:128]))
        item.setdefault("title", item["doc_id"])
        item.setdefault("text", item.get("body", ""))
        out.append(item)
    return out if isinstance(payload, list) else {"documents": out}


def load_pdf(path: str | Path) -> list[dict[str, Any]]:
    """Extract text from a PDF, one document per page-group.

    Falls back to a recorded error rather than raising when ``pypdf`` is absent, so a
    mixed knowledge-base directory still ingests its readable formats.
    """
    p = Path(path)
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except Exception as exc:
        log.error("cannot read PDF %s: pypdf unavailable (%s)", p, exc)
        return []
    try:
        reader = PdfReader(str(p))
    except Exception as exc:
        log.error("cannot parse PDF %s: %s", p, exc)
        return []
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if len(text) < 50:
            continue
        pages.append({
            "doc_id": _stable_id(str(p), str(i)),
            "title": f"{p.stem} page {i + 1}",
            "text": text,
            "category": p.parent.name or "docs",
            "language": "en",
            "tags": (),
            "source_path": f"{p}#page={i + 1}",
        })
    return pages


LOADERS = {".md": load_md, ".markdown": load_md, ".txt": load_txt,
           ".json": load_json, ".pdf": load_pdf, ".html": load_md}


def ingest_directory(root: str | Path, max_chars: int = 1100,
                     overlap_chars: int = 150) -> list[Chunk]:
    """Ingest every supported file under ``root`` and return its chunks."""
    root = Path(root)
    if not root.exists():
        log.warning("knowledge root does not exist: %s", root)
        return []

    chunks: list[Chunk] = []
    files_seen: list[str] = []
    files_failed: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            continue
        files_seen.append(str(path))
        try:
            loaded = loader(path)
        except Exception as exc:
            log.error("failed to ingest %s: %s", path, exc)
            files_failed.append(f"{path}: {exc}")
            continue
        docs = loaded if isinstance(loaded, list) else (
            loaded.get("documents", []) if isinstance(loaded, dict) and "documents" in loaded
            else [loaded]
        )
        for doc in docs:
            chunks.extend(chunk_document(doc, max_chars=max_chars, overlap_chars=overlap_chars))

    log.info("ingested %d files -> %d chunks (%d failures)",
             len(files_seen), len(chunks), len(files_failed))
    if files_failed:
        for f in files_failed[:5]:
            log.error("  %s", f)
    return chunks


def knowledge_base_chunks(max_chars: int = 1100, overlap_chars: int = 150) -> list[Chunk]:
    """Chunks from the bundled synthetic knowledge base plus any files on disk."""
    from ..support.knowledge_base import load_knowledge_base

    chunks: list[Chunk] = []
    for doc in load_knowledge_base():
        chunks.extend(chunk_document(doc, max_chars=max_chars, overlap_chars=overlap_chars))
    disk = ingest_directory(Path(__file__).resolve().parents[2] / "knowledge_base",
                            max_chars=max_chars, overlap_chars=overlap_chars)
    seen = {c.chunk_id for c in chunks}
    chunks.extend(c for c in disk if c.chunk_id not in seen)
    return chunks


__all__ = ["Chunk", "chunk_document", "ingest_directory", "knowledge_base_chunks",
           "load_md", "load_txt", "load_json", "load_pdf", "LOADERS"]
