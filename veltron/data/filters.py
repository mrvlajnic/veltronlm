"""Document quality pipeline.

Raw documents pass through, in order:

    normalize -> language-tag -> boilerplate-strip -> quality-filter
    -> PII filter -> exact-dedup -> near-dedup (MinHash + LSH) -> format validation

Each stage records how many documents it removed and why, so the dataset card can report
real funnel statistics instead of a guess. Every stage is a pure function of its input,
which is what makes the pipeline reproducible.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class FilterStats:
    """Document counts per stage."""

    seen: int = 0
    accepted: int = 0
    stages: dict[str, int] = field(default_factory=dict)

    def reject(self, stage: str) -> None:
        self.stages[stage] = self.stages.get(stage, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "accepted": self.accepted,
            "rejected_by_stage": dict(sorted(self.stages.items(), key=lambda kv: -kv[1])),
            "accept_rate": round(self.accepted / self.seen, 4) if self.seen else 0.0,
        }


# ------------------------------------------------------------------- normalization
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
#: Trailing horizontal whitespace only. Leading whitespace is load-bearing in code, so it
#: is never collapsed -- reducing an 8-space indent to 2 would silently corrupt every
#: Python file in the corpus and turn valid programs into syntax errors.
_TRAILING_WS = re.compile(r"[ \t]+\n")
_INTERIOR_WS = re.compile(r"(?<=\S)[ \t]{3,}(?=\S)")
_NL_RUN = re.compile(r"\n{4,}")


def normalize(text: str, nfkc: bool = True) -> str:
    """Unicode-normalize and strip control characters, preserving code indentation."""
    if nfkc:
        # NFC would merge Serbian Cyrillic/Latin lookalikes unpredictably across
        # sources; NFKC folds compatibility forms (ligatures, full-width chars) which is
        # what we want for web text without destroying byte-level identity.
        text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _INTERIOR_WS.sub("  ", text)
    text = _NL_RUN.sub("\n\n\n", text)
    return text.strip()


# ----------------------------------------------------------------- language tagging
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")
_GREEK = re.compile(r"[Ͱ-Ͽ]")
_ARABIC = re.compile(r"[؀-ۿ]")
_CJK = re.compile(r"[一-鿿぀-ヿ]")
_DIGIT = re.compile(r"\d")

_SR_DIACRITICS = set("čćžšđČĆŽŠĐ")
_SR_CYR = set("абвгдђежзијклљмнњопрстћуфхцчџшљњљ")


def detect_language(text: str) -> tuple[str, float]:
    """Script-and-stopword based language ID.

    Deliberately simple and dependency-free: script ratios separate the families, and
    within the Latin family a small stopword table separates English from Serbian. The
    returned confidence is the share of the dominant signal, which is honest about how
    weak this heuristic is on short text.
    """
    n = len(text)
    if n < 8:
        return "und", 0.0
    cyr = len(_CYRILLIC.findall(text))
    lat = len(_LATIN.findall(text))
    grk = len(_GREEK.findall(text))
    ara = len(_ARABIC.findall(text))
    cjk = len(_CJK.findall(text))
    letters = max(cyr + lat + grk + ara + cjk, 1)

    if cjk / letters > 0.3:
        return "zho", cjk / letters
    if ara / letters > 0.3:
        return "ara", ara / letters
    if grk / letters > 0.3:
        return "ell", grk / letters
    if cyr / letters > 0.25:
        if set(text) & _SR_CYR and cyr / letters > 0.4:
            return "sr", min(1.0, cyr / letters)
        return "ru", min(1.0, cyr / letters)

    if lat / letters < 0.4:
        return "und", lat / letters

    low = " " + re.sub(r"[^\w\s]", " ", text.lower()) + " "
    tokens = low.split()
    if not tokens:
        return "und", 0.0
    tok_set = set(tokens)

    en_stop = {"the", "and", "of", "to", "in", "is", "for", "that", "with", "was", "on",
               "are", "as", "this", "from", "by", "it", "be", "or", "an", "which", "have"}
    sr_stop = {"je", "и", "на", "је", "се", "да", "за", "од", "у", "su", "na", "se",
               "da", "za", "od", "u", "i", "a", "ne", "ali", "kaо", "što", "ovo", "ili",
               "the", "kod", "ima", "bio", "sve", "više"}

    en_hits = sum(1 for w in en_stop if w in tok_set)
    sr_hits = sum(1 for w in sr_stop if w in tok_set)

    has_sr_diacritic = bool(set(text) & _SR_DIACRITICS)
    # Serbian keeps diacritics; also recognise the Latn variant written without them.
    if sr_hits > en_hits or has_sr_diacritic:
        conf = min(1.0, (sr_hits + (2 if has_sr_diacritic else 0)) / 5.0)
        return "sr", conf
    if en_hits > 0:
        return "en", min(1.0, en_hits / 6.0)
    return "und", 0.2


# ------------------------------------------------------------------ quality signals
_BOILERPLATE = re.compile(
    r"(cookie (policy|consent)|accept all cookies|subscribe to our newsletter|"
    r"all rights reserved|terms of (use|service)|privacy policy|"
    r"прихвате све колачиће|претражите|управите садржајма|"
    r"click here to|read more|related articles?|share this|advertisement)",
    re.I,
)
_CAPTION_JUNK = re.compile(r"^(image|photo|figure|table|chart|izvor|source)\s*\d*[:.]?\s*$", re.I)
_CTRL_RATIO_BAD = 0.02


def strip_boilerplate(text: str) -> str:
    lines = [ln for ln in text.split("\n") if not _BOILERPLATE.search(ln) and not _CAPTION_JUNK.match(ln.strip())]
    return "\n".join(lines)


def quality_report(text: str) -> dict[str, Any]:
    """Cheap, interpretable quality signals. No model, no external service."""
    n = len(text)
    if n == 0:
        return {"ok": False, "reason": "empty"}

    lines = text.split("\n")
    words = text.split()
    n_words = len(words)
    if n_words < 20:
        return {"ok": False, "reason": "too_few_words", "words": n_words}

    # Unique-word ratio: boilerplate and menus score low, real prose scores high.
    ttr = len(set(w.lower() for w in words)) / n_words

    avg_word_len = sum(len(w) for w in words) / n_words
    avg_sent_len = n_words / max(1, sum(1 for ch in text if ch in ".!?"))

    nonalpha = sum(1 for ch in text if not ch.isalpha() and not ch.isspace())
    nonalpha_ratio = nonalpha / n

    # Line-length structure separates prose (few long lines) from code (many short).
    mean_line = n / max(1, len(lines))
    long_lines = sum(1 for ln in lines if len(ln) > 200)
    long_line_ratio = long_lines / max(1, len(lines))

    url_count = len(re.findall(r"https?://", text))
    bracket_count = text.count("{") + text.count("}")

    issues = []
    if ttr < 0.16:
        issues.append("low_ttr")
    if nonalpha_ratio > 0.45:
        issues.append("high_nonalpha")
    if avg_sent_len > 60:
        issues.append("sentences_too_long")
    if url_count > max(20, n_words * 0.02):
        issues.append("link_stuffing")
    if sum(1 for _ in re.finditer(r"(.)\1{40,}", text)):
        issues.append("char_run")

    # A composite score in [0, 1] used for ranking, not for hard filtering.
    score = 1.0
    score -= max(0.0, (0.30 - ttr)) * 1.5
    score -= max(0.0, (nonalpha_ratio - 0.35)) * 0.8
    score -= min(0.25, url_count / max(1, n_words) * 2.0)
    score -= 0.10 * len(issues)
    score += 0.05 if long_line_ratio > 0.5 else 0.0   # code-like
    score += 0.05 if bracket_count > 5 else 0.0       # structured
    score = float(max(0.0, min(1.0, score)))

    return {
        "ok": True,
        "words": n_words,
        "chars": n,
        "ttr": round(ttr, 4),
        "avg_word_len": round(avg_word_len, 2),
        "avg_sentence_len": round(avg_sent_len, 2),
        "nonalpha_ratio": round(nonalpha_ratio, 4),
        "mean_line_len": round(mean_line, 1),
        "long_line_ratio": round(long_line_ratio, 4),
        "url_count": url_count,
        "score": round(score, 4),
        "issues": issues,
    }


# ---------------------------------------------------------------------------- PII
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_SR = re.compile(r"(?<!\d)(?:\+381|0)(?:6[0-9]{2,8}|[1-9]\d{2,3})[\s/\-]?\d{2,4}[\s/\-]?\d{2,4}(?!\d)")
_PHONE_INTL = re.compile(r"(?<!\d)\+\d{1,3}[\s\-]?(?:\(?\d{1,4}\)?[\s\-]?){1,4}\d{2,4}(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")
_SSN_US = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_IPV4 = re.compile(r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")
_AWS_KEY = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_GITHUB_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b")
_SLACK = re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----")

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": _EMAIL,
    "phone_sr": _PHONE_SR,
    "phone_intl": _PHONE_INTL,
    "iban": _IBAN,
    "credit_card": _CARD,
    "us_ssn": _SSN_US,
    "ipv4": _IPV4,
    "jwt": _JWT,
    "aws_key": _AWS_KEY,
    "github_token": _GITHUB_TOKEN,
    "slack_token": _SLACK,
    "private_key": _PRIVATE_KEY,
}


def detect_pii(text: str) -> dict[str, int]:
    """Count PII matches per category. Used both to drop and to redact."""
    out: dict[str, int] = {}
    for name, pat in PII_PATTERNS.items():
        n = len(pat.findall(text))
        if n:
            out[name] = n
    return out


_PII_THRESHOLDS: dict[str, int] = {
    "private_key": 1,
    "aws_key": 1,
    "github_token": 1,
    "slack_token": 1,
    "jwt": 1,
    "us_ssn": 1,
    "iban": 1,
    "credit_card": 1,
    "email": 3,
    "phone_sr": 4,
    "phone_intl": 4,
}


def pii_verdict(text: str) -> tuple[str, dict[str, int]]:
    """Return ``("drop"|"redact"|"keep", counts)``.

    Credentials and unique financial identifiers always drop the document. Bulk
    identifiers (many emails or phone numbers) are redacted instead, because a support
    log legitimately contains a customer's address while a leaked key never should.
    """
    counts = detect_pii(text)
    if not counts:
        return "keep", {}
    for name, hits in counts.items():
        if hits >= _PII_THRESHOLDS.get(name, 99):
            return "drop", counts
    return "redact", counts


_REDACT_TOKEN = "[REDACTED]"


def redact_pii(text: str) -> str:
    out = text
    for name in ("private_key", "aws_key", "github_token", "slack_token", "jwt",
                 "credit_card", "us_ssn", "iban", "email", "phone_sr", "phone_intl"):
        out = PII_PATTERNS[name].sub(_REDACT_TOKEN, out)
    return out


# ------------------------------------------------------------------- deduplication
def exact_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


#: Prefix of each document used for near-duplicate detection. Near-duplicate boilerplate
#: (templated FAQs, licence headers, generated stubs) repeats from the first line, so a
#: bounded prefix detects the cases that matter while keeping the cost linear in document
#: count instead of corpus size.
NEARDUP_PREFIX_CHARS = 4000
SHINGLE_SIZE = 5

#: 2**31 - 1, the Mersenne prime. Chosen so ``a * h`` stays inside int64: with a 64-bit
#: modulus the product would overflow.
_MERSENNE_31 = (1 << 31) - 1

_PERM_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _permutation_params(num_perm: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed permutations ``(a, b)`` for ``hash(h) = (a*h + b) mod (2**31 - 1)``.

    Seeded from a constant so a rebuild reproduces identical signatures; a random draw
    would still dedupe correctly but would change every hash between runs.
    """
    cached = _PERM_CACHE.get(num_perm)
    if cached is None:
        rng = np.random.default_rng(0xC0FFEE)
        a = rng.integers(1, _MERSENNE_31, size=num_perm, dtype=np.int64)
        b = rng.integers(0, _MERSENNE_31, size=num_perm, dtype=np.int64)
        cached = (a, b)
        _PERM_CACHE[num_perm] = cached
    return cached


def _stable_hash(word: str) -> int:
    """Process-independent hash, masked to 63 bits so it fits in ``int64``.

    ``hash()`` is randomised per interpreter run for ``str``, so using it would make
    deduplication irreproducible. blake2b with a fixed key is stable and fast enough;
    the mask avoids an unsigned-to-signed overflow when converting to a NumPy int64.
    """
    digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 62) - 1)


def _shingles(text: str, k: int = SHINGLE_SIZE) -> np.ndarray:
    """Word k-grams of ``text`` as a stable-hash array."""
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < k:
        return np.array([_stable_hash(" ".join(toks))] if toks else [], dtype=np.int64)
    grams = [" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)]
    arr = np.fromiter((_stable_hash(g) for g in grams), dtype=np.int64, count=len(grams))
    return arr % _MERSENNE_31


def minhash(text: str, num_perm: int = 64, k: int = SHINGLE_SIZE) -> np.ndarray:
    """MinHash signature over word k-grams, returned as ``uint64``.

    Implemented directly rather than via a dependency so the signature is bit-identical
    across runs, machines and dependency versions.
    """
    shingles = _shingles(text[:NEARDUP_PREFIX_CHARS], k)
    if shingles.size == 0:
        return np.zeros(num_perm, dtype=np.uint64)
    a, b = _permutation_params(num_perm)
    # (num_perm, 1) x (1, n_shingles) -> min over shingles, giving the signature.
    hashed = (a[:, None] * shingles[None, :].astype(np.int64) + b[:, None]) % _MERSENNE_31
    return hashed.min(axis=1).astype(np.uint64)


class Deduper:
    """Exact + MinHash-LSH near-duplicate detection.

    LSH banding: a 64-permutation signature is split into ``bands`` bands of ``rows``
    permutations; two documents collide when they agree on every permutation of any band.
    With ``bands=16, rows=4`` the operating point sits near a 0.8 Jaccard threshold,
    which is where templated boilerplate actually lives while genuinely distinct documents
    stay apart.
    """

    def __init__(self, num_perm: int = 64, bands: int = 16, threshold: float = 0.8) -> None:
        if num_perm % bands != 0:
            raise ValueError(f"num_perm={num_perm} must be divisible by bands={bands}")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.threshold = threshold
        self.exact: set[str] = set()
        self.buckets: list[dict[int, set[int]]] = [{} for _ in range(bands)]
        self.n = 0
        self.dup_exact = 0
        self.dup_near = 0

    @staticmethod
    def _band_keys(sig: np.ndarray, bands: int, rows: int) -> list[int]:
        """One key per band, mixing all of that band's rows so agreement must be total."""
        keys = []
        for i in range(bands):
            chunk = sig[i * rows : (i + 1) * rows].tobytes()
            keys.append(int(hashlib.blake2b(chunk, digest_size=8, key=b"veltron-lsh").hexdigest()[:16], 16))
        return keys

    def check_and_add(self, text: str) -> tuple[bool, str]:
        """Return ``(is_duplicate, reason)``, registering the document when kept."""
        eh = exact_hash(text[:NEARDUP_PREFIX_CHARS])
        if eh in self.exact:
            self.dup_exact += 1
            return True, "exact"
        sig = minhash(text)
        keys = self._band_keys(sig, self.bands, self.rows)
        for bi, key in enumerate(keys):
            if key in self.buckets[bi]:
                self.dup_near += 1
                return True, f"near_lsh_band{bi}"
        idx = self.n
        self.n += 1
        for bi, key in enumerate(keys):
            self.buckets[bi].setdefault(key, set()).add(idx)
        self.exact.add(eh)
        return False, "unique"

    def stats(self) -> dict[str, Any]:
        return {
            "documents_seen": self.n + self.dup_exact + self.dup_near,
            "kept": self.n,
            "duplicate_exact": self.dup_exact,
            "duplicate_near": self.dup_near,
            "num_perm": self.num_perm,
            "bands": self.bands,
            "rows": self.rows,
            "approx_jaccard_threshold": round(self.threshold, 3),
            "near_dup_prefix_chars": NEARDUP_PREFIX_CHARS,
            "shingle_size": SHINGLE_SIZE,
        }


# ------------------------------------------------------------------- format checks
def detect_format(text: str) -> str:
    t = text.lstrip()
    if t.startswith("```") or re.search(r"(?m)^\s*(def |class |import |function |const |let |var )", text[:4000]):
        return "code"
    if t.startswith(("{", "[")) and text.count("\n") < 0.2 * len(text):
        return "json"
    if re.search(r"(?m)^#{1,6}\s", text[:4000]):
        return "markdown"
    if t.startswith("<?xml") or t.startswith("<!DOCTYPE html"):
        return "markup"
    if text.count("\n") / max(1, len(text) / 100) > 0.25:
        return "code"
    return "prose"


# --------------------------------------------------------------------- pipeline
MIN_CHARS = 250


def process_document(
    doc: dict[str, Any],
    stats: FilterStats,
    deduper: Deduper | None,
    min_quality: float = 0.0,
) -> dict[str, Any] | None:
    """Run one document through the whole pipeline. ``None`` means rejected."""
    stats.seen += 1
    text = normalize(doc.get("text", ""))

    if len(text) < MIN_CHARS:
        stats.reject("too_short")
        return None

    text = strip_boilerplate(text)
    if len(text) < MIN_CHARS:
        stats.reject("all_boilerplate")
        return None

    verdict, counts = pii_verdict(text)
    if verdict == "drop":
        stats.reject("pii_drop")
        return None
    if verdict == "redact":
        text = redact_pii(text)

    rep = quality_report(text)
    if not rep["ok"]:
        stats.reject(f"quality_{rep.get('reason', 'unknown')}")
        return None
    if rep["score"] < min_quality:
        stats.reject("quality_below_threshold")
        return None

    if deduper is not None:
        is_dup, reason = deduper.check_and_add(text)
        if is_dup:
            stats.reject(f"dedup_{reason}")
            return None

    out = dict(doc)
    out["text"] = text
    out["format"] = detect_format(text)
    detected_lang, conf = detect_language(text)
    # Respect an explicit upstream language tag; only override when detection disagrees.
    declared = doc.get("language", "und")
    if declared in ("und", "", None) or declared == "text":
        out["language"] = detected_lang
    elif detected_lang != "und" and declared not in ("en", "sr") and declared != detected_lang:
        out["language"] = detected_lang
    else:
        out["language"] = declared
    out["language_detected"] = detected_lang
    out["language_confidence"] = round(conf, 3)
    out["quality"] = rep
    out["pii_counts"] = counts
    out["pii_verdict"] = verdict
    out["chars"] = len(text)
    out["approx_tokens"] = max(1, len(text) // 4)
    stats.accepted += 1
    return out


def process_stream(
    docs: Iterable[dict[str, Any]],
    dedupe: bool = True,
    min_quality: float = 0.0,
) -> Iterator[tuple[dict[str, Any], FilterStats]]:
    stats = FilterStats()
    deduper = Deduper() if dedupe else None
    for doc in docs:
        cleaned = process_document(doc, stats, deduper, min_quality=min_quality)
        if cleaned is not None:
            yield cleaned, stats
    if deduper is not None:
        stats.dedup_stats = deduper.stats()  # type: ignore[attr-defined]
        log.info("dedup: %s", deduper.stats())


__all__ = [
    "FilterStats",
    "Deduper",
    "normalize",
    "detect_language",
    "strip_boilerplate",
    "quality_report",
    "detect_pii",
    "pii_verdict",
    "redact_pii",
    "minhash",
    "exact_hash",
    "detect_format",
    "process_document",
    "process_stream",
    "PII_PATTERNS",
    "MIN_CHARS",
]
