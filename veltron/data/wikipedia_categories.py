"""Bulk Wikipedia acquisition by category traversal.

Replaces random-article sweeps, which waste 30 of every 50 pages because MediaWiki caps
``exlimit`` at 20. Walking categories and then batching titles 20-at-a-time into
``prop=extracts`` reaches the API's actual throughput ceiling.

Also the only approach that yields *balanced* coverage: a curated category list keeps the
corpus from collapsing onto whatever a random sweep happens to return.

Politeness: one request at a time with a floor on the interval, exponential backoff on
429, and a hard cap on requests per language.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from collections.abc import Iterator

from ..utils.logging_utils import get_logger
from .acquire import Document, http_get
from .sources import SOURCES

log = get_logger(__name__)

#: (category, depth) per language. Serbian uses its own category namespace names.
CATEGORY_PLAN: dict[str, tuple[tuple[str, int], ...]] = {
    "en": (
        ("Machine learning", 1), ("Computer science", 1), ("Mathematics", 1),
        ("Physics", 1), ("Chemistry", 1), ("Biology", 1), ("History", 1),
        ("Geography", 1), ("Economics", 1), ("Software engineering", 1),
        ("Programming languages", 1), ("Operating systems", 1), ("Databases", 1),
        ("Networks", 1), ("Artificial intelligence", 1), ("Electronics", 1),
        ("Statistics", 1), ("Linguistics", 1), ("Medicine", 1), ("Law", 1),
        ("Economics of manufacturing", 2), ("Internet", 2), ("Telecommunications", 2),
        ("Climate change", 1), ("Astronomy", 1), ("Serbia", 2), ("Balkans", 2),
    ),
    "sr": (
        ("Машинско учење", 1), ("Информатика", 1), ("Математика", 1),
        ("Физика", 1), ("Хемија", 1), ("Биологија", 1), ("Историја", 1),
        ("Географија", 1), ("Економија", 1), ("Програмирање", 1),
        ("Програмски језици", 1), ("Оперативни систем", 1), ("Базе података", 1),
        ("Рачунарске мреже", 1), ("Вештачка интелигенција", 1),
        ("Електроника", 1), ("Статистика", 1), ("Лингвистика", 1),
        ("Медицина", 1), ("Право", 1), ("Интернет", 1), ("Астрономија", 1),
        ("Србија", 2), ("Југoslavија", 2), ("Климатологија", 1),
    ),
    "sh": (
        ("Računarstvo", 1), ("Matematika", 1), ("Fizika", 1), ("Historija", 1),
        ("Biologija", 1), ("Hemija", 1), ("Geografija", 1), ("Ekonomija", 1),
        ("Programiranje", 1), ("Umjetna inteligencija", 1), ("Srbija", 2),
    ),
}

API = "https://{lang}.wikipedia.org/w/api.php"


def list_category(lang: str, category: str, depth: int, max_titles: int = 400) -> list[str]:
    """Collect article titles from a category, optionally descending into subcategories."""
    titles: list[str] = []
    seen: set[str] = set()
    frontier = [category]
    level = 0

    while frontier and level <= depth and len(titles) < max_titles:
        next_frontier: list[str] = []
        for cat in frontier:
            cont: dict[str, str] = {}
            while len(titles) < max_titles:
                params = {
                    "action": "query", "format": "json", "formatversion": "2",
                    "list": "categorymembers", "cmtitle": f"Category:{cat}",
                    "cmnamespace": "0", "cmlimit": "100", "cmtype": "page",
                }
                if cont:
                    params.update(cont)
                qs = urllib.parse.urlencode(params)
                status, body = http_get(f"{API.format(lang=lang)}?{qs}")
                if status != 200 or not body:
                    break
                try:
                    payload = json.loads(body.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    break
                members = payload.get("query", {}).get("categorymembers", []) or []
                for m in members:
                    t = m.get("title")
                    if t and t not in seen:
                        seen.add(t)
                        titles.append(t)
                if "continue" not in payload or not members:
                    break
                cont = {k: str(v) for k, v in payload["continue"].items()}
            # Subcategory descent is a separate query pass.
            sub_params = {
                "action": "query", "format": "json", "formatversion": "2",
                "list": "categorymembers", "cmtitle": f"Category:{cat}",
                "cmnamespace": "14", "cmlimit": "50", "cmtype": "subcat",
            }
            status, body = http_get(f"{API.format(lang=lang)}?{urllib.parse.urlencode(sub_params)}")
            if status == 200 and body:
                try:
                    payload = json.loads(body.decode("utf-8", errors="replace"))
                    for m in payload.get("query", {}).get("categorymembers", []) or []:
                        if level < depth:
                            next_frontier.append(m["title"].replace("Category:", ""))
                except json.JSONDecodeError:
                    pass
        frontier = next_frontier
        level += 1

    log.info("category %r [%s]: %d titles", category, lang, len(titles))
    return titles[:max_titles]


def fetch_extracts(lang: str, titles: list[str], batch: int = 20) -> Iterator[tuple[str, str]]:
    """Fetch plain-text extracts, ``batch`` titles per request (``exlimit`` max is 20)."""
    today = time.strftime("%Y-%m-%d")
    del today
    for i in range(0, len(titles), batch):
        chunk = titles[i : i + batch]
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "prop": "extracts", "explaintext": "1", "exlimit": "20",
            "redirects": "1", "titles": "|".join(chunk),
        }
        status, body = http_get(f"{API.format(lang=lang)}?{urllib.parse.urlencode(params)}")
        if status != 200 or not body:
            log.warning("extracts batch %d/%s failed: %s", i // batch, len(titles) // batch, status)
            continue
        try:
            payload = json.loads(body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        for page in payload.get("query", {}).get("pages", []) or []:
            if page.get("missing"):
                continue
            text = (page.get("extract") or "").strip()
            if len(text) >= 400:
                yield page.get("title", ""), text
        if (i // batch) % 10 == 0:
            log.info("extracts %s %d/%d batches", lang, i // batch + 1, (len(titles) + batch - 1) // batch)


def fetch_category_corpus(
    lang: str,
    max_titles_per_category: int = 400,
    max_docs_per_lang: int | None = None,
) -> Iterator[Document]:
    """Walk every planned category for ``lang`` and yield full-text documents."""
    src = SOURCES["wikipedia-random"]
    today = time.strftime("%Y-%m-%d")
    plan = CATEGORY_PLAN.get(lang)
    if not plan:
        log.warning("no category plan for language %r", lang)
        return

    seen: set[str] = set()
    produced = 0
    for category, depth in plan:
        titles = list_category(lang, category, depth, max_titles_per_category)
        fresh = [t for t in titles if t not in seen]
        seen.update(fresh)
        if not fresh:
            continue
        for title, text in fetch_extracts(lang, fresh):
            import re

            text = re.sub(r"\n{3,}", "\n\n", text)
            produced += 1
            yield Document(
                text=text,
                source=f"wikipedia-cat-{lang}",
                license=src.license,
                license_url=src.license_url,
                category="multilingual" if lang != "en" else "general",
                language=lang,
                title=title,
                url=f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                retrieval_date=today,
            )
        if max_docs_per_lang and produced >= max_docs_per_lang:
            log.info("stopping %s at %d documents", lang, produced)
            break
    log.info("wikipedia-cat[%s]: %d documents total", lang, produced)


def fetch_all_categories(
    max_titles_per_category: int = 400,
    languages: tuple[str, ...] = ("en", "sr", "sh"),
    max_docs_per_lang: int | None = 2000,
) -> Iterator[Document]:
    for lang in languages:
        yield from fetch_category_corpus(lang, max_titles_per_category, max_docs_per_lang)


__all__ = ["CATEGORY_PLAN", "list_category", "fetch_extracts", "fetch_category_corpus",
           "fetch_all_categories"]
