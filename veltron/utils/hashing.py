"""Content hashing for dataset/tokenizer/config provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    """Hash a JSON-serialisable object with sorted keys so it is order-independent."""
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))


def short(digest: str, n: int = 12) -> str:
    return digest[:n]


def hash_directory(path: str | Path, pattern: str = "*") -> str:
    """Order-independent hash over a directory's file contents."""
    h = hashlib.sha256()
    root = Path(path)
    for p in sorted(root.rglob(pattern)):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode("utf-8"))
            h.update(sha256_file(p).encode("ascii"))
    return h.hexdigest()


__all__ = ["sha256_bytes", "sha256_file", "sha256_text", "sha256_json", "short", "hash_directory"]
