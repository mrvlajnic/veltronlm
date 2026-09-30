"""Document acquisition.

Downloads from the declared sources in :mod:`veltron.data.sources` with a real user agent,
honouring HTTP semantics, rate limiting and the Wikimedia API's continuation protocol.
Every fetched document is emitted as JSONL with a full provenance record so a dataset can
be reconstructed, audited and re-cited later.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..utils.hashing import sha256_text
from ..utils.logging_utils import get_logger
from .sources import (
    PERMISSIVE_REPOS,
    SOURCES,
    WIKIPEDIA_SR_LATIN_TITLES,
    WIKIPEDIA_TITLES,
)

log = get_logger(__name__)

USER_AGENT = "VeltronLM/0.1 (open research pretraining corpus; contact: repository maintainer)"
RATE_LIMIT_S = 0.35
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0


@dataclass
class Document:
    """One training document plus provenance."""

    text: str
    source: str
    license: str
    license_url: str
    category: str
    language: str
    title: str = ""
    doc_id: str = ""
    url: str = ""
    retrieval_date: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = sha256_text(f"{self.source}|{self.url}|{self.text[:512]}")[:24]

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["chars"] = len(self.text)
        d["approx_tokens"] = max(1, len(self.text) // 4)
        d["sha256_12"] = self.doc_id
        return d


class RateLimiter:
    """Simple monotonic-interval limiter shared across all fetchers."""

    def __init__(self, min_interval: float = RATE_LIMIT_S) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


def http_get(url: str, timeout: int = 30, retries: int = MAX_RETRIES) -> tuple[int, bytes]:
    """GET with exponential backoff. Returns ``(status, body)``; body empty on failure."""
    limiter = getattr(http_get, "_limiter", None)
    if limiter is None:
        limiter = RateLimiter()
        http_get._limiter = limiter  # type: ignore[attr-defined]
    last_err: Exception | None = None
    for attempt in range(retries):
        limiter.wait()
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            # 4xx other than 429 will not improve on retry.
            if 400 <= e.code < 500 and e.code != 429:
                return e.code, b""
            last_err = e
        except Exception as e:  # network flake
            last_err = e
        sleep_for = RETRY_BACKOFF_S * (2**attempt)
        log.debug("retry %d/%d for %s in %.1fs (%s)", attempt + 1, retries, url[:80], sleep_for, last_err)
        time.sleep(sleep_for)
    log.warning("giving up on %s: %s", url[:100], last_err)
    return -1, b""


# ----------------------------------------------------------------------- wikipedia
_WIKITEXT_CLEAN = re.compile(r"\n{3,}")


def fetch_wikipedia(lang: str, titles: Iterable[str], variant: str = "standard") -> Iterator[Document]:
    """Fetch plain-text extracts from a Wikimedia API in batches of 20 titles."""
    src_key = "wikipedia-sr" if lang != "en" else "wikipedia-en"
    src = SOURCES[src_key]
    titles = list(titles)
    today = time.strftime("%Y-%m-%d")
    base = src.url_template.replace("{lang}", lang).split("&titles=")[0]

    for i in range(0, len(titles), 20):
        chunk = titles[i : i + 20]
        joined = "|".join(chunk)
        url = f"{base}&titles={urllib.parse.quote(joined, safe='')}"
        status, body = http_get(url)
        if status != 200 or not body:
            log.warning("wikipedia %s batch %d failed: status=%s", lang, i // 20, status)
            continue
        try:
            payload = json.loads(body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            log.warning("wikipedia %s batch %d bad json: %s", lang, i // 20, exc)
            continue

        for page in payload.get("query", {}).get("pages", []) or []:
            if page.get("missing"):
                continue
            text = (page.get("extract") or "").strip()
            if len(text) < 400:
                continue
            title = page.get("title", "")
            yield Document(
                text=_WIKITEXT_CLEAN.sub("\n\n", text),
                source=src_key,
                license=src.license,
                license_url=src.license_url,
                category=src.category,
                language=lang,
                title=title,
                url=f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                retrieval_date=today,
            )
        log.info("wikipedia[%s] batch %d/%d ok", lang, i // 20 + 1, (len(titles) + 19) // 20)


def fetch_wikipedia_random(
    lang: str = "en",
    requests_count: int = 40,
    per_request: int = 50,
    variant: str = "random",
) -> Iterator[Document]:
    """Sweep random articles from a Wikimedia project.

    The MediaWiki ``generator=random`` endpoint returns up to ``per_request`` pages per
    call, which is an order of magnitude more efficient than the title-batched ``extracts``
    endpoint (whose ``exlimit`` caps at 20). The sweep offset is recorded by the caller in
    the dataset manifest so the same article set can be re-fetched.
    """
    src = SOURCES["wikipedia-random"]
    today = time.strftime("%Y-%m-%d")
    base = (
        f"https://{lang}.wikipedia.org/w/api.php?action=query&format=json"
        f"&generator=random&grnnamespace=0&grnlimit={per_request}&prop=extracts"
        f"&explaintext=1&exlimit=20&formatversion=2"
    )
    n = 0
    for i in range(requests_count):
        status, body = http_get(base)
        if status != 200 or not body:
            log.warning("wikipedia random %s request %d failed: %s", lang, i, status)
            continue
        try:
            payload = json.loads(body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            log.warning("wikipedia random %s bad json: %s", lang, exc)
            continue
        pages = payload.get("query", {}).get("pages", []) or []
        for page in pages:
            if page.get("missing"):
                continue
            text = (page.get("extract") or "").strip()
            if len(text) < 400:
                continue
            title = page.get("title", "")
            n += 1
            yield Document(
                text=_WIKITEXT_CLEAN.sub("\n\n", text),
                source=f"wikipedia-{variant}" if lang == "en" else f"wikipedia-sr-{variant}",
                license=src.license,
                license_url=src.license_url,
                category="multilingual" if lang != "en" else "general",
                language=lang,
                title=title,
                url=f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                retrieval_date=today,
            )
        log.info("wikipedia random[%s] %d/%d: %d cumulative docs",
                 lang, i + 1, requests_count, n)
    log.info("wikipedia random[%s] total %d documents", lang, n)


def fetch_all_wikipedia(random_requests: int = 0) -> Iterator[Document]:
    """Curated titles first, then optionally a random-article sweep for volume."""
    yield from fetch_wikipedia("en", WIKIPEDIA_TITLES["en"])
    yield from fetch_wikipedia("sr", WIKIPEDIA_TITLES["sr"], variant="cyrl")
    yield from fetch_wikipedia("sr", WIKIPEDIA_SR_LATIN_TITLES, variant="latin")
    if random_requests > 0:
        yield from fetch_wikipedia_random("en", random_requests, 50, "random-en")
        yield from fetch_wikipedia_random("sr", random_requests, 50, "random-sr")


# ------------------------------------------------------------------------------ code
def _iter_permissive_repo_files() -> Iterator[tuple[RepoFileAny, str]]:
    for rf in PERMISSIVE_REPOS:
        yield rf, rf.path


RepoFileAny = Any


def fetch_permissive_code(limit: int | None = None) -> Iterator[Document]:
    """Fetch source files from the curated MIT/Apache/BSD/PSF repositories."""
    today = time.strftime("%Y-%m-%d")
    n = 0
    for rf, path in _iter_permissive_repo_files():
        if limit is not None and n >= limit:
            break
        if rf.spdx == "GPL-2.0-only":
            # GPL text is not mixed into the corpus: it would impose copyleft on the
            # model weights as a derivative work in most jurisdictions.
            log.info("skipping GPL-2.0-only file %s/%s", rf.repo, path)
            continue
        url = f"https://raw.githubusercontent.com/{rf.repo}/{rf.ref}/{path}"
        status, body = http_get(url)
        if status != 200 or not body:
            continue
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(text) < 200:
            continue
        ext = path.rsplit(".", 1)[-1].lower()
        n += 1
        yield Document(
            text=text,
            source=f"github:{rf.repo}",
            license=rf.spdx,
            license_url=f"https://spdx.org/licenses/{rf.spdx.replace(' OR ', '-or-')}.html",
            category="technical" if rf.role == "technical" else "code",
            language=_code_language(ext),
            title=f"{rf.repo}/{path}",
            url=url,
            retrieval_date=today,
        )
    log.info("fetched %d permissive code/technical files", n)


def _code_language(ext: str) -> str:
    return {
        "py": "python", "rs": "rust", "js": "javascript", "ts": "typescript",
        "go": "go", "java": "java", "c": "c", "h": "c", "cpp": "cpp",
        "md": "markdown", "rst": "rst", "txt": "text", "json": "json",
        "sh": "shell", "yaml": "yaml", "yml": "yaml", "toml": "toml", "html": "html",
        "css": "css", "rb": "ruby", "php": "php", "sql": "sql", "xml": "xml",
    }.get(ext, "text")


# ------------------------------------------------------------------------ literature
GUTENBERG_IDS: tuple[int, ...] = (
    11, 1342, 84, 1661, 2701, 74, 76, 35, 174, 98, 1400, 345, 1232, 4300,
    100, 2600, 5200, 33, 2554, 2148, 219, 2814, 203, 42, 1497, 5827, 768, 1184,
    120, 8800, 1727, 996, 1399, 2600, 2009, 205, 1934, 1250, 1023, 730, 3600,
)


def fetch_gutenberg(limit: int | None = None) -> Iterator[Document]:
    """Fetch public-domain plain texts. The Gutenberg header/footer is stripped."""
    today = time.strftime("%Y-%m-%d")
    ids = GUTENBERG_IDS[:limit] if limit else GUTENBERG_IDS
    src = SOURCES["gutenberg"]
    n = 0
    for gid in ids:
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        status, body = http_get(url)
        if status != 200 or not body:
            url = f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt"
            status, body = http_get(url)
        if status != 200 or not body:
            continue
        text = strip_gutenberg(body.decode("utf-8", errors="replace"))
        if len(text) < 2000:
            continue
        n += 1
        yield Document(
            text=text,
            source="gutenberg",
            license=src.license,
            license_url=src.license_url,
            category="general",
            language="en",
            title=f"gutenberg-{gid}",
            url=f"https://www.gutenberg.org/ebooks/{gid}",
            retrieval_date=today,
        )
    log.info("fetched %d gutenberg texts", n)


_START_MARK = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", re.I)
_END_MARK = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", re.I)


def strip_gutenberg(text: str) -> str:
    m_start = _START_MARK.search(text)
    m_end = _END_MARK.search(text)
    if m_start:
        text = text[m_start.end() :]
    if m_end:
        text = text[: m_end.start()]
    return text.strip()


# -------------------------------------------------------------------------- writer
class DocumentWriter:
    """Append documents to a JSONL shard with run-level provenance metadata."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("w", encoding="utf-8")
        self.n = 0
        self.chars = 0

    def write(self, doc: Document) -> None:
        self.fh.write(json.dumps(doc.to_json(), ensure_ascii=False) + "\n")
        self.n += 1
        self.chars += len(doc.text)

    def close(self) -> dict[str, Any]:
        self.fh.close()
        return {"path": str(self.path), "documents": self.n, "chars": self.chars}

    def __enter__(self) -> DocumentWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["Document", "http_get", "RateLimiter", "fetch_wikipedia", "fetch_wikipedia_random",
           "fetch_all_wikipedia", "fetch_permissive_code", "fetch_gutenberg",
           "strip_gutenberg", "DocumentWriter"]
