"""End-to-end retrieval-augmented generation.

    question -> (intent/rewrite) -> retrieve -> rerank -> build context
             -> VeltronLM -> grounded answer -> citations

The design constraint that shapes everything here: **the knowledge base may not contain the
answer**, and in that case the correct output is a refusal, not a fluent invention. Every
stage therefore reports a numeric signal that the escalation policy can threshold on,
rather than relying on the model to self-report uncertainty.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..support.triage import (
    Classification,
    classify_ticket,
    decide_escalation,
    detect_safety_signals,
    escalation_message,
    extract_entities,
)
from ..utils.logging_utils import get_logger
from .ingest import Chunk, knowledge_base_chunks
from .retriever import RetrievedChunk, Retriever

log = get_logger(__name__)

DEFAULT_INDEX_DIR = "models/rag_index"

SYSTEM_PROMPT_EN = """You are Veltron Support, a customer support assistant for Veltron Industries.

Rules you must follow:
1. Answer ONLY from the CONTEXT section. The context is the entire authority you have.
2. If the context does not contain the answer, say exactly that you cannot find it and
   offer to escalate to a human agent. Never guess a policy, price, date or specification.
3. Never state a policy, price, warranty period, deadline or technical specification that
   is not present in the context.
4. Never ask for or repeat a password, 2FA code, full card number, or another customer's data.
5. Keep answers short and actionable. Use numbered steps when the context gives a procedure.
6. If the context says something applies only to a specific region, plan or condition, state
   that qualification in your answer.
7. End with a `Sources:` line listing the citation numbers you actually used, like
   `Sources: [1]`. Omit the line only if you refused to answer.

CONTEXT:
{context}"""

SYSTEM_PROMPT_SR = """Ti si Veltron Support, asistent za korisničku podršku kompanije Veltron Industries.

Pravila koja moraš poštovati:
1. Odgovaraj ISKLJUČIVO iz dela CONTEXT. To je jedini izvor na koji imaš pravo da se osloniš.
2. Ako odgovor nije u kontekstu, reci da ga ne možeš naći i ponudi eskalaciju ljudskom agentu.
   Nikada ne izmišljaj politiku, cenu, rok ili tehničku specifikaciju.
3. Ne traži i ne ponavljaj lozinku, 2FA kod, ceo broj kartice ili podatke drugih kupaca.
4. Odgovori kratko i sa konkretnim koracima. Ako kontekst daje proceduru, koristi numerisane korake.
5. Završi linijom `Izvori: [1]` sa citatima koje si stvarno koristio.

CONTEXT:
{context}"""

REFUSAL_EN = (
    "I can't find that in the Veltron support documentation available to me, so I won't "
    "invent an answer. I'd rather escalate this to a human agent who can check the account "
    "and confirm the details."
)
REFUSAL_SR = (
    "Ne mogu da nađem taj odgovor u dostupnoj dokumentaciji Veltron podrške, pa neću da "
    "izmišljam. Predlažem da slučaj preuzme ljudski agent."
)

_CITATION_LINE = re.compile(r"^\s*(?:Sources?|Izvori?)\s*:?\s*\[.*\]\s*$", re.I | re.M)


@dataclass
class RAGConfig:
    top_k: int = 6
    candidates: int = 40
    per_doc_cap: int = 2
    max_context_chars: int = 4200
    chunk_chars: int = 1100
    overlap_chars: int = 150
    min_retrieval_score: float = 0.18
    max_answer_tokens: int = 320
    temperature: float = 0.3
    top_p: float = 0.9
    rerank: bool = True
    rewrite_query: bool = True
    citation_format: str = "numbered"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Citation:
    index: int
    doc_id: str
    chunk_id: str
    title: str
    section_path: str
    score: float
    quoted: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RAGAnswer:
    """Everything the API and the UI need for one answered question."""

    question: str
    answer: str
    refused: bool
    escalated: bool
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    classification: dict[str, Any] = field(default_factory=dict)
    entities: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    escalation: dict[str, Any] = field(default_factory=dict)
    rewritten_query: str = ""
    latency_seconds: float = 0.0
    generation: dict[str, Any] = field(default_factory=dict)
    synthetic_data_notice: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)


# ------------------------------------------------------------------- query rewrite
_LEAD_FILLER = re.compile(
    r"^(?:hi|hey|hello|good\s+(?:morning|afternoon|evening)|please|pls|kindly|"
    r"dear\s+(?:support|team|customer)|zdravo|dobar\s+dan)\b[,\s!]*", re.I)
_TRAILING_FILLER = re.compile(
    r"[,\s]*(?:please|pls|thanks|thank you|thx|can you help|any\w*)\s*[?.!]*$", re.I)


def rewrite_query(question: str, language: str = "en") -> str:
    """Normalise a support question into a retrieval-friendly query.

    Only *filler* is removed. An earlier version also stripped the customer's framing
    ("my VeltronHub X1 is not working"), which deleted the product name and the symptom --
    the two most informative terms in the sentence. Framing carries signal in support
    queries, so the product and the symptom are deliberately kept; the vocabulary bridging
    that does the real work happens in ``Retriever._augment_query`` and
    :func:`veltron.rag.retriever.expand_query`.
    """
    q = question.strip()
    q = _LEAD_FILLER.sub("", q)
    q = _TRAILING_FILLER.sub("", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q or question.strip()


# --------------------------------------------------------------------- context build
def build_context(hits: Sequence[RetrievedChunk], max_chars: int) -> tuple[str, list[Citation]]:
    """Assemble numbered context blocks and the matching citation list."""
    blocks: list[str] = []
    cites: list[Citation] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        header = f"[{i}] {h.title} > {h.section_path}  (category: {h.category}, language: {h.language})"
        block = f"{header}\n{h.text}"
        if used + len(block) > max_chars:
            # Try a shorter excerpt of the same chunk before dropping it entirely.
            room = max_chars - used - len(header) - 8
            if room < 220:
                break
            excerpt = h.text[:room].rsplit(" ", 1)[0] + " ..."
            block = f"{header}\n{excerpt}"
        blocks.append(block)
        cites.append(
            Citation(index=i, doc_id=h.doc_id, chunk_id=h.chunk_id,
                     title=h.title, section_path=h.section_path,
                     score=h.score, quoted=h.text[:240])
        )
        used += len(block) + 2
    return "\n\n".join(blocks), cites


def verify_citations(answer: str, citations: Sequence[Citation]) -> dict[str, Any]:
    """Check that every citation the answer claims is real and was retrieved.

    A model asked for a `Sources:` line will sometimes invent ``[4]`` when only three
    chunks were retrieved, so the line is validated rather than trusted.
    """
    match = _CITATION_LINE.search(answer or "")
    claimed: list[int] = []
    if match:
        claimed = [int(n) for n in re.findall(r"\[(\d+)\]", match.group(0))]
    valid = {c.index for c in citations}
    fabricated = [n for n in claimed if n not in valid]
    grounded = [n for n in claimed if n in valid]
    return {
        "claimed": claimed,
        "grounded": grounded,
        "fabricated": fabricated,
        "has_source_line": bool(match),
        "all_valid": not fabricated,
        "coverage": (len(grounded) / max(1, len(citations))) if citations else 0.0,
    }


def strip_citation_line(answer: str) -> str:
    return _CITATION_LINE.sub("", answer or "").strip()


# ------------------------------------------------------------------------ pipeline
class RAGPipeline:
    """Retrieval + grounded generation + escalation."""

    def __init__(
        self,
        generator: Any = None,
        retriever: Retriever | None = None,
        cfg: RAGConfig | None = None,
    ) -> None:
        self.cfg = cfg or RAGConfig()
        self.generator = generator
        self.retriever = retriever
        self.model_name = getattr(getattr(generator, "model", None).cfg, "name", "unavailable") \
            if generator is not None else "unavailable"

    # -------------------------------------------------------------- index I/O
    def build_index(self, out_dir: str | Path = DEFAULT_INDEX_DIR,
                    chunks: Sequence[Chunk] | None = None) -> Retriever:
        chunks = list(chunks) if chunks is not None else knowledge_base_chunks(
            max_chars=self.cfg.chunk_chars, overlap_chars=self.cfg.overlap_chars
        )
        self.retriever = Retriever(chunks)
        self.retriever.save(out_dir)
        log.info("RAG index: %s", json.dumps(self.retriever.stats()))
        return self.retriever

    def load_index(self, directory: str | Path = DEFAULT_INDEX_DIR) -> Retriever:
        self.retriever = Retriever.load(directory)
        return self.retriever

    # ------------------------------------------------------------------ query
    def answer(self, question: str, language: str | None = None,
               include_debug: bool = False) -> RAGAnswer:
        t0 = time.perf_counter()
        if self.retriever is None:
            raise RuntimeError("RAGPipeline has no retriever; call build_index() or load_index()")

        classification = classify_ticket(question)
        entities = extract_entities(question)
        signals = detect_safety_signals(question)
        lang = language or entities.language
        rewritten = rewrite_query(question, lang) if self.cfg.rewrite_query else question

        retrieval_query = self._augment_query(rewritten, classification)
        hits = self.retriever.search(
            retrieval_query,
            top_k=self.cfg.top_k,
            candidates=self.cfg.candidates,
            per_doc_cap=self.cfg.per_doc_cap,
            rerank=self.cfg.rerank,
            # Confidence is judged on the *rewritten* customer query, never on the
            # taxonomy-augmented one. Augmentation exists to widen the candidate pool; if
            # it also fed the confidence score, an off-topic question would inherit every
            # injected term and look as well-supported as a real one.
            confidence_query=rewritten,
        )
        # Absolute confidence, not the rank-normalised score: the latter is 1.0 for the top
        # hit of every query, including ones that match nothing at all.
        top_score = max((h.confidence for h in hits), default=0.0)

        context, citations = build_context(hits, self.cfg.max_context_chars)

        escalation = decide_escalation(
            classification, signals,
            retrieval_score=top_score, retrieved=len(hits),
        )

        if escalation.escalate:
            text = escalation_message(lang, escalation.template_key)
            return RAGAnswer(
                question=question, answer=text, refused=True, escalated=True,
                citations=[c.as_dict() for c in citations],
                retrieved=[h.as_dict() for h in hits] if include_debug else [],
                classification=classification.as_dict(),
                entities=entities.as_dict(),
                safety=signals.as_dict(),
                escalation=escalation.as_dict(),
                rewritten_query=rewritten,
                latency_seconds=round(time.perf_counter() - t0, 4),
            )

        if self.generator is None:
            # Retrieval-only mode: no model available, so return the grounded evidence and
            # say so rather than pretending an answer was produced.
            return RAGAnswer(
                question=question,
                answer=(f"No generator is loaded, so no answer was synthesised. "
                        f"{len(citations)} supporting passage(s) were retrieved."),
                refused=True, escalated=False,
                citations=[c.as_dict() for c in citations],
                retrieved=[h.as_dict() for h in hits],
                classification=classification.as_dict(),
                entities=entities.as_dict(),
                safety=signals.as_dict(),
                escalation=escalation.as_dict(),
                rewritten_query=rewritten,
                latency_seconds=round(time.perf_counter() - t0, 4),
            )

        prompt = (SYSTEM_PROMPT_SR if lang == "sr" else SYSTEM_PROMPT_EN).format(context=context)
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": question}]

        gen_cfg = _generation_config(self.cfg)
        result = self.generator.generate_chat(messages, gen_cfg)

        cite_check = verify_citations(result.text, citations)
        answer_text = strip_citation_line(result.text)
        # A retrieval-only refusal or an empty completion is treated as "not answered"
        # rather than shown to the customer as if it were support.
        if not answer_text.strip():
            answer_text = REFUSAL_SR if lang == "sr" else REFUSAL_EN

        final_escalation = decide_escalation(
            classification, signals,
            retrieval_score=top_score, retrieved=len(hits),
            answer_confidence=self._answer_confidence(result, citations, answer_text),
        )
        if final_escalation.escalate and not cite_check["grounded"]:
            answer_text = (escalation_message(lang, final_escalation.template_key) + "\n\n"
                           + answer_text).strip()

        return RAGAnswer(
            question=question,
            answer=answer_text,
            refused=bool(final_escalation.escalate),
            escalated=bool(final_escalation.escalate),
            citations=[c.as_dict() for c in citations],
            retrieved=[h.as_dict() for h in hits] if include_debug else [],
            classification=classification.as_dict(),
            entities=entities.as_dict(),
            safety=signals.as_dict(),
            escalation=final_escalation.as_dict(),
            rewritten_query=rewritten,
            latency_seconds=round(time.perf_counter() - t0, 4),
            generation={
                "completion_tokens": result.completion_tokens,
                "prompt_tokens": result.prompt_tokens,
                "tokens_per_second": round(result.tokens_per_second, 3),
                "time_to_first_token": round(result.time_to_first_token, 4),
                "finish_reason": result.finish_reason,
                "model": self.model_name,
                "citation_check": cite_check,
            },
            synthetic_data_notice=any(h.synthetic for h in hits),
        )

    def _augment_query(self, query: str, classification: Classification) -> str:
        """Add taxonomy vocabulary to the query to bridge customer wording and docs."""
        vocab = {
            "technical_issue": "offline error fault not working reset troubleshooting",
            "account_access": "login password pairing account locked access",
            "billing": "charge invoice refund VAT subscription renewal payment",
            "shipping": "shipping delivery tracking dispatch carrier parcel lost",
            "refund_request": "refund return RMA money back policy",
            "product_info": "specification specifications dimensions compatible",
            "bug_report": "bug defect expected actual behaviour",
            "how_to": "configure setup install enable how to",
        }.get(classification.category, "")
        return f"{query} {vocab}".strip() if vocab else query

    @staticmethod
    def _answer_confidence(result: Any, citations: Sequence[Citation], answer: str) -> float:
        """Blend the generator's own sequence confidence with retrieval strength.

        ``logprob`` is absent from some generators, in which case confidence falls back to
        retrieval strength alone rather than being invented.
        """
        retrieval = max((c.score for c in citations), default=0.0)  # already absolute confidence
        tokens = max(1, getattr(result, "completion_tokens", 1))
        avg_lp = getattr(result, "mean_logprob", None)
        if avg_lp is None:
            return round(retrieval, 4)
        length_conf = float(min(1.0, tokens / 40.0))
        return round(max(0.0, min(1.0, 0.5 * retrieval + 0.3 * length_conf + 0.2 * float(avg_lp) + 0.2)), 4)

    # ------------------------------------------------------------ convenience
    def search_only(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if self.retriever is None:
            raise RuntimeError("no retriever loaded")
        q = rewrite_query(question)
        hits = self.retriever.search(q, top_k=top_k or self.cfg.top_k,
                                     per_doc_cap=self.cfg.per_doc_cap,
                                     rerank=self.cfg.rerank)
        return [h.as_dict() for h in hits]


def _generation_config(cfg: RAGConfig) -> Any:
    from ..inference.generator import GenerationConfig

    return GenerationConfig(
        max_new_tokens=cfg.max_answer_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=0,
        repetition_penalty=1.05,
        greedy=cfg.temperature <= 0.0,
    )


__all__ = [
    "RAGPipeline",
    "RAGConfig",
    "RAGAnswer",
    "Citation",
    "rewrite_query",
    "build_context",
    "verify_citations",
    "strip_citation_line",
    "SYSTEM_PROMPT_EN",
    "SYSTEM_PROMPT_SR",
    "REFUSAL_EN",
    "REFUSAL_SR",
    "DEFAULT_INDEX_DIR",
]
