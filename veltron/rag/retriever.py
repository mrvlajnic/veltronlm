"""Retrieval: BM25 lexical search fused with a hashed-embedding vector search.

Why hybrid: a support corpus is full of exact identifiers -- serial prefixes, policy
numbers, product names, error strings -- that embedding models blur together, and also of
paraphrases ("it shows up as offline" vs "the hub is not reachable") that BM25 misses.
Reciprocal Rank Fusion combines the two without needing a learned reranker or a score
calibration between them.

No external vector database is required: the index is a ``sqlite`` blob plus numpy arrays,
so the whole retrieval layer is inspectable and dependency-light.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.hashing import sha256_text
from ..utils.logging_utils import get_logger
from .ingest import Chunk

log = get_logger(__name__)

#: Word pattern covering Latin, Latin-Extended (for Serbian čćžšđ) and Cyrillic.
#:
#: An ASCII-only ``[a-z]`` class silently drops every Serbian token: ``re.IGNORECASE``
#: does not widen an ASCII range to other scripts, so ``"Гаранција"`` tokenised to
#: nothing at all and Serbian queries matched only by accident. That is why the
#: character classes are spelled out per script.
_WORD = re.compile(
    r"(?:[a-zA-Z0-9čćžšđČĆŽŠĐ]|[Ѐ-ӿ])"      # first character: Latin or Cyrillic
    r"(?:[a-zA-Z0-9čćžšđČĆŽŠĐ_'-]|[Ѐ-ӿ])*",  # continuation, may include an apostrophe
    re.UNICODE,
)

#: Deliberately small stopword list. Removing too much hurts support search, where words
#: like "not", "no" and "can" carry the meaning of a policy question.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of", "to", "in",
    "on", "at", "for", "with", "and", "or", "as", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "my", "your", "me", "do", "does",
    "did", "have", "has", "had", "can", "could", "would", "should", "will", "not", "but",
    "from", "by", "so", "if", "then", "than", "when", "what", "which", "who", "how",
    "je", "и", "на", "се", "да", "за", "од", "у", "су", "је", "ни", "али", "или", "као",
    "što", "ovo", "ona", "tako", "već", "samo", "kod", "ima", "bio", "sve", "više",
}


def tokenize(text: str) -> list[str]:
    return [w for w in (m.group(0).lower() for m in _WORD.finditer(text)) if w not in STOPWORDS]


#: Serbian -> English domain vocabulary.
#:
#: The knowledge base is mostly English while Serbian customers write in Serbian, so a
#: literal query like "koliko traje garancija" shares *zero* tokens with the English
#: warranty document. Expanding the Serbian terms is what makes cross-lingual retrieval
#: work without training a multilingual encoder. The list is deliberately narrow and
#: domain-specific: a general bilingual dictionary would add noise faster than recall.
BILINGUAL_TERMS: dict[str, tuple[str, ...]] = {
    "garancija": ("warranty", "guarantee", "rma"),
    "garanciju": ("warranty", "rma"),
    "reklamacija": ("warranty", "claim", "rma", "defect"),
    "povrat": ("refund", "return"),
    "povratna": ("refund", "return"),
    "vratiti": ("refund", "return"),
    "racun": ("invoice", "billing"),
    "račun": ("invoice", "billing"),
    "plaćanje": ("payment", "billing"),
    "placanje": ("payment", "billing"),
    "kartica": ("card", "payment"),
    "dostava": ("shipping", "delivery"),
    "isporuka": ("shipping", "delivery"),
    "pošiljka": ("parcel", "shipping", "tracking"),
    "posiljka": ("parcel", "shipping", "tracking"),
    "praćenje": ("tracking", "shipping"),
    "pracenje": ("tracking", "shipping"),
    "lozinka": ("password", "login", "access"),
    "prijava": ("login", "access", "signin"),
    "ulogovati": ("login", "signin", "account"),
    "nalog": ("account", "login"),
    "uređaj": ("device", "hub", "product"),
    "uredjaj": ("device", "hub", "product"),
    "ne radi": ("not working", "offline", "troubleshooting"),
    "povezivanje": ("pairing", "connection", "network"),
    "povezivanja": ("pairing", "connection"),
    "greška": ("error", "fault"),
    "greska": ("error", "fault"),
    "problem": ("issue", "problem", "fault"),
    "otkazati": ("cancelled", "cancel"),
    "otkazivanje": ("cancellation", "cancel"),
    "pretplata": ("subscription", "plan"),
    "cena": ("price", "cost", "plan"),
    "cenovnik": ("price", "plan", "pricing"),
    "specifikacija": ("specification", "specs"),
    "dimenzije": ("dimensions", "size"),
    "podrška": ("support", "help"),
    "podrska": ("support", "help"),
    "servis": ("service", "support"),
    "kvar": ("fault", "defect", "broken"),
    "servisiranje": ("repair", "service"),
    "popravka": ("repair", "fix"),
    "bezbjedno": ("secure", "security", "password"),
    "bezbedno": ("secure", "security", "password"),
    "sifra": ("password",),
    "šifra": ("password",),
    "kredit": ("card", "payment", "billing"),
    "zrakoplov": ("aircraft",),
    "avion": ("aircraft",),
    "subota": ("saturday",),
    "nedelja": ("sunday",),
}


def expand_query(query: str) -> str:
    """Append English equivalents of any Serbian domain terms found in the query."""
    low = query.lower()
    extra: list[str] = []
    for sr, en in BILINGUAL_TERMS.items():
        if sr in low:
            extra.extend(en)
    return f"{query} {' '.join(extra)}" if extra else query


# --------------------------------------------------------------------------- BM25
@dataclass
class BM25Index:
    """Okapi BM25 over the chunk collection."""

    k1: float = 1.5
    b: float = 0.75

    doc_tokens: list[list[str]] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    idf: dict[str, float] = field(default_factory=dict)
    doc_len: list[int] = field(default_factory=list)
    avg_len: float = 0.0
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def build(cls, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> BM25Index:
        idx = cls(k1=k1, b=b)
        for c in chunks:
            # Title, section path and tags are indexed alongside the body so that a query
            # naming a product or section ranks its chunks even when the words appear only
            # in the heading.
            text = f"{c.title} {c.section_path} {' '.join(c.tags)} {c.text}"
            toks = tokenize(text)
            idx.doc_tokens.append(toks)
            idx.doc_len.append(len(toks))
            counts = Counter(toks)
            for term, tf in counts.items():
                idx.df[term] += 1
                idx.postings.setdefault(term, []).append((len(idx.doc_tokens) - 1, tf))
        n = max(1, len(chunks))
        idx.avg_len = sum(idx.doc_len) / n
        # Lucene-style IDF with the +1 guard, which stays positive for terms in every doc.
        idx.idf = {
            t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in idx.df.items()
        }
        return idx

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        scores: dict[int, float] = {}
        for term in toks:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_i, tf in self.postings[term]:
                dl = self.doc_len[doc_i] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_len, 1e-9))
                scores[doc_i] = scores.get(doc_i, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return ranked

    def to_dict(self) -> dict[str, Any]:
        return {
            "k1": self.k1, "b": self.b,
            "n_docs": len(self.doc_tokens),
            "vocab": len(self.df),
            "avg_len": self.avg_len,
        }

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({
                "k1": self.k1, "b": self.b,
                "doc_tokens": self.doc_tokens, "df": dict(self.df),
                "idf": self.idf, "doc_len": self.doc_len, "avg_len": self.avg_len,
                "postings": {t: v for t, v in self.postings.items()},
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        idx = cls(k1=blob["k1"], b=blob["b"])
        idx.doc_tokens = blob["doc_tokens"]
        idx.df = Counter(blob["df"])
        idx.idf = blob["idf"]
        idx.doc_len = blob["doc_len"]
        idx.avg_len = blob["avg_len"]
        idx.postings = blob["postings"]
        return idx


# --------------------------------------------------------------------- vector side
class HashedVectorIndex:
    """Hashed n-gram embeddings: a linear random projection of a bag-of-ngrams.

    No neural encoder is used on purpose. A support knowledge base is small (hundreds of
    chunks), and what "dense" retrieval must supply here is lexical-overlap sensitivity --
    near-duplicate phrasings of the same policy -- which hashed character and word n-grams
    capture well while remaining fully deterministic and dependency-free. It also means the
    retriever cannot hallucinate semantics from an unrelated embedding model.
    """

    def __init__(self, dim: int = 512, ngrams: tuple[int, ...] = (2, 3)) -> None:
        self.dim = dim
        self.ngrams = ngrams
        self.matrix: np.ndarray | None = None
        self.chunk_ids: list[str] = []

    @staticmethod
    def _features(text: str) -> list[str]:
        toks = tokenize(text)
        feats: list[str] = []
        for n in (1,) + tuple(self_ngrams):
            for i in range(len(toks) - n + 1):
                feats.append("w" + "".join(toks[i : i + n]))
        low = re.sub(r"\s+", " ", text.lower())
        for n in (3, 4):
            for i in range(len(low) - n + 1):
                feats.append("c" + low[i : i + n])
        return feats

    def _hash(self, feat: str) -> int:
        return int.from_bytes(sha256_text(feat)[:8].encode("ascii"), "little") % self.dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        feats = self._features(text)
        if not feats:
            return v
        for f in feats:
            # Sign trick: hashing collisions average out instead of always adding.
            h = self._hash(f)
            v[h] += 1.0 if (h & 1) == 0 else -1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    def build(self, chunks: Sequence[Chunk]) -> np.ndarray:
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.matrix = np.stack([self.embed(c.text) for c in chunks]) if chunks else np.zeros((0, self.dim), np.float32)
        return self.matrix

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        if self.matrix is None or len(self.matrix) == 0:
            return []
        q = self.embed(query)
        sims = self.matrix @ q
        idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idx]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(str(path), matrix=self.matrix if self.matrix is not None else np.zeros((0, self.dim), np.float32),
                            chunk_ids=np.array(self.chunk_ids, dtype=object), allow_pickle=True)

    @classmethod
    def load(cls, path: str | Path, dim: int = 512) -> HashedVectorIndex:
        z = np.load(str(path), allow_pickle=True)
        idx = cls(dim=dim)
        idx.matrix = z["matrix"]
        idx.chunk_ids = list(z["chunk_ids"])
        return idx


#: Global so ``_features`` can reference ``self_ngrams`` inside a staticmethod-style body.
self_ngrams = (2, 3)


# ------------------------------------------------------------------- fusion + rerank
@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    title: str
    section_path: str
    category: str
    language: str
    tags: list[str]
    score: float
    bm25_rank: int | None
    vector_rank: int | None
    bm25_score: float
    vector_score: float
    rerank_score: float = 0.0
    synthetic: bool = True
    disclaimer: str = ""
    #: Absolute retrieval confidence in [0, 1]. Unlike :attr:`score`, this is *not*
    #: renormalised against the best hit, so it stays meaningful when the whole query
    #: matches nothing -- which is exactly the case the escalation threshold must catch.
    confidence: float = 0.0

    def citation(self) -> str:
        return f"{self.title} > {self.section_path}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]] | None = None,
    k: int = 60,
) -> dict[int, float]:
    """RRF: score(d) = sum over lists of 1/(k + rank(d)).

    Rank-based fusion is used instead of summing normalised scores because BM25 scores are
    unbounded and cosine similarity is in [-1, 1]; their scales are not comparable, and
    calibrating them on a handful of documents would be more fragile than ignoring them.
    """
    fused: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, doc in enumerate(lst, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    """Scale a score map to [0, 1] with the best hit at 1.0.

    Only for *ranking and display*. Thresholds must use absolute confidence, because a
    renormalised map is 1.0 at the top for any query at all -- including one that matches
    nothing.
    """
    if not scores:
        return {}
    top = max(scores.values())
    if top <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / top for k, v in scores.items()}


def absolute_confidence(bm_score: float, bm_max: float, vec_z: float) -> float:
    """Absolute retrieval confidence in [0, 1].

    Two independent, individually interpretable signals are combined:

    * ``bm_score / bm_max`` -- how strongly the query's rare vocabulary matches, relative
      to the best lexical hit in this query. Stays near zero when nothing matches.
    * ``vec_z`` -- the embedding similarity expressed as a z-score against the mean
      similarity of this query's candidate set, squashed to [0, 1]. A raw cosine is *not*
      usable here: on a small corpus every chunk shares common n-grams, so raw cosines
      cluster around 0.8 regardless of relevance and carry almost no signal. Centering is
      what turns the number into a measure of "more similar than typical for this query".

    Lexical evidence carries more weight (0.70/0.30) because a support corpus is dominated
    by exact identifiers -- serial prefixes, error strings, policy numbers -- that an
    n-gram embedding blurs together. The embedding's job is paraphrase recall, not
    precision. The weighting was set by measurement on scripts/verify_rag.py: at 0.50/0.50
    a 2% BM25 shortfall on "how much is the Plus plan" was overturned by the embedding and
    the pricing document fell to rank 2.
    """
    lex = 0.0 if bm_max <= 0 else max(0.0, min(1.0, bm_score / bm_max))
    # ``sigmoid(0) == 0.5``, so a raw logistic would hand every candidate half a point of
    # vector credit even when it is merely average for this query -- and a nonsense query
    # produces nothing but average similarities, which would look like weak evidence.
    # Subtracting the midpoint makes "no better than typical" score exactly zero, so only a
    # genuinely above-average match contributes.
    vec = max(0.0, (1.0 / (1.0 + float(np.exp(-vec_z))) - 0.5) * 2.0)
    return round(0.70 * lex + 0.30 * vec, 6)


def _zscores(values: dict[int, float]) -> dict[int, float]:
    """Standardise a score map across the candidate set."""
    if len(values) < 2:
        return {k: 0.0 for k in values}
    arr = np.array(list(values.values()), dtype=np.float64)
    mean, std = float(arr.mean()), float(arr.std())
    if std < 1e-9:
        return {k: 0.0 for k in values}
    return {k: (v - mean) / std for k, v in values.items()}


def diversity_rerank(
    hits: list[RetrievedChunk],
    per_doc_cap: int = 2,
    lambda_mmr: float = 0.25,
) -> list[RetrievedChunk]:
    """Cap chunks per document and damp redundant near-duplicates.

    Without the cap, a three-chunk FAQ section can occupy every slot in the context window
    and crowd out the one policy the question actually needs. Ordering is by absolute
    confidence, so the cap removes whole redundant documents without letting the
    rank-normalised fusion score take over.
    """
    by_doc: Counter = Counter()
    kept: list[RetrievedChunk] = []
    for h in sorted(hits, key=lambda x: -x.confidence):
        if by_doc[h.doc_id] >= per_doc_cap:
            continue
        by_doc[h.doc_id] += 1
        h.rerank_score = h.confidence * (1.0 - lambda_mmr * min(1.0, h.score))
        kept.append(h)
    return kept


class Retriever:
    """Hybrid lexical + vector retrieval over a chunk collection."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        bm25: BM25Index | None = None,
        vectors: HashedVectorIndex | None = None,
        dim: int = 512,
    ) -> None:
        self.chunks = list(chunks)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = bm25 if bm25 is not None else BM25Index.build(self.chunks)
        self._title_tokens: list[set[str]] | None = None
        self.vectors = vectors if vectors is not None else HashedVectorIndex(dim=dim)
        if vectors is None:
            self.vectors.build(self.chunks)

    # ------------------------------------------------------------------ search
    #: Multiplier applied to the IDF mass of a query term found in a chunk's title,
    #: section path or tags. Tuned on scripts/verify_rag.py.
    TITLE_BOOST = 2.5

    def _title_token_sets(self) -> None:
        if self._title_tokens is not None:
            return
        self._title_tokens = [
            set(tokenize(f"{c.title} {c.section_path} {' '.join(c.tags)}"))
            for c in self.chunks
        ]

    def _title_match_scores(self, query: str) -> dict[int, float]:
        self._title_token_sets()
        q_terms = [t for t in tokenize(query) if t in self.bm25.idf]
        out: dict[int, float] = {}
        if not q_terms:
            return out
        for i, title_toks in enumerate(self._title_tokens):  # type: ignore[arg-type]
            bonus = sum(self.bm25.idf.get(t, 0.0) for t in q_terms if t in title_toks)
            if bonus:
                out[i] = self.TITLE_BOOST * bonus
        return out

    def _max_title_bonus(self, bm_max: float) -> float:
        """Upper bound on the title bonus, used only to keep the lexical denominator sane.

        Capped at a fraction of ``bm_max`` so that adding a heading match cannot make the
        confidence of every candidate collapse toward zero.
        """
        return min(bm_max, self.TITLE_BOOST * 3.0) if bm_max > 0 else 0.0

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        top_k: int = 8,
        candidates: int = 40,
        per_doc_cap: int = 2,
        rerank: bool = True,
        confidence_query: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve candidates and score their absolute confidence.

        Args:
            query: the (possibly augmented) query used to select candidates.
            confidence_query: an optional *unaugmented* query used to compute each hit's
                confidence. Vocabulary bridging is a recall mechanism: it helps find the
                right document for "it doesn't work" -> "offline error troubleshooting".
                Feeding its expanded form into the confidence calculation would manufacture
                evidence, because an off-topic question inherits every term the taxonomy
                added and scores as well as an on-topic one. Confidence must therefore
                reflect how well the *customer's own words* match.
        """
        bm = self.bm25.search(expand_query(query), top_k=candidates)
        vc = self.vectors.search(expand_query(query), top_k=candidates)
        bm_scores = dict(bm)
        vc_scores = dict(vc)
        fused = reciprocal_rank_fusion(
            [[i for i, _ in bm], [i for i, _ in vc]],
            scores=[bm_scores, vc_scores],
        )
        bm_rank = {i: r + 1 for r, (i, _) in enumerate(bm)}
        vc_rank = {i: r + 1 for r, (i, _) in enumerate(vc)}

        # Rank-normalised score drives ordering and display.
        norm_fused = _normalize_scores(fused)
        # Absolute signals drive the "do we actually know the answer?" decision.
        if confidence_query is not None and confidence_query != query:
            conf_bm = dict(self.bm25.search(expand_query(confidence_query), top_k=candidates))
            conf_vc = dict(self.vectors.search(expand_query(confidence_query), top_k=candidates))
        else:
            conf_bm, conf_vc = bm_scores, vc_scores
        bm_max = max(conf_bm.values()) if conf_bm else 0.0
        vc_z = _zscores(conf_vc)
        title_bonus = self._title_match_scores(expand_query(confidence_query or query))

        hits: list[RetrievedChunk] = []
        for i, f in fused.items():
            c = self.chunks[i]
            cos = conf_vc.get(i, 0.0)
            # Title and heading matches are added to the lexical score. A customer asking
            # "how much is the Plus plan" shares almost as many body tokens with a
            # paragraph about subscription retention as with the pricing table itself; the
            # section heading "Veltron Cloud plans" is what actually settles it.
            lex_raw = conf_bm.get(i, 0.0) + title_bonus.get(i, 0.0)
            hits.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text,
                    title=c.title, section_path=c.section_path, category=c.category,
                    language=c.language, tags=list(c.tags),
                    score=round(norm_fused.get(i, 0.0), 6),
                    bm25_rank=bm_rank.get(i), vector_rank=vc_rank.get(i),
                    bm25_score=round(conf_bm.get(i, 0.0), 6),
                    vector_score=round(cos, 6),
                    confidence=absolute_confidence(lex_raw, bm_max + self._max_title_bonus(bm_max),
                                                  vc_z.get(i, 0.0)),
                    synthetic=c.synthetic, disclaimer=c.disclaimer,
                )
            )
        # Order by absolute confidence so a strong lexical hit is not displaced by a
        # merely high rank-fusion score.
        hits.sort(key=lambda h: -h.confidence)
        hits = hits[:top_k]
        return diversity_rerank(hits, per_doc_cap=per_doc_cap) if rerank else hits

    # -------------------------------------------------------------------- I/O
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "chunks.json").open("w", encoding="utf-8") as fh:
            json.dump([c.as_dict() for c in self.chunks], fh, ensure_ascii=False, indent=1)
        self.bm25.save(d / "bm25.pkl")
        self.vectors.save(d / "vectors.npz")
        (d / "index_meta.json").write_text(
            json.dumps({"n_chunks": len(self.chunks), "bm25": self.bm25.to_dict(),
                        "vector_dim": self.vectors.dim}, indent=2),
            encoding="utf-8",
        )
        log.info("index saved: %d chunks -> %s", len(self.chunks), d)
        return d

    @classmethod
    def load(cls, directory: str | Path, dim: int = 512) -> Retriever:
        d = Path(directory)
        raw = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**{k: (tuple(v) if k == "tags" else v) for k, v in c.items()}) for c in raw]
        bm = BM25Index.load(d / "bm25.pkl")
        vec = HashedVectorIndex.load(d / "vectors.npz", dim=dim)
        return cls(chunks, bm25=bm, vectors=vec, dim=dim)

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": len(self.chunks),
            "documents": len({c.doc_id for c in self.chunks}),
            "languages": dict(Counter(c.language for c in self.chunks)),
            "categories": dict(Counter(c.category for c in self.chunks)),
            "bm25_vocab": len(self.bm25.df),
            "avg_chunk_chars": round(
                sum(len(c.text) for c in self.chunks) / max(1, len(self.chunks)), 1
            ),
        }


__all__ = [
    "BM25Index",
    "HashedVectorIndex",
    "Retriever",
    "RetrievedChunk",
    "reciprocal_rank_fusion",
    "absolute_confidence",
    "diversity_rerank",
    "expand_query",
    "tokenize",
    "STOPWORDS",
    "BILINGUAL_TERMS",
]
