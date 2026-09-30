"""Preference pairs for direct preference optimisation.

Pairs are constructed **from the RAG system itself**, which means each preference is a real,
reproducible signal rather than a stylistic guess:

``chosen``
    A grounded answer whose citations validate, produced from retrieved context, at a
    temperature that keeps it faithful to that context.

``rejected``
    One of five concrete failure modes that the support system is explicitly designed to
    catch -- ungrounded policy invention, fabricated citation, refusal-when-answerable,
    verbosity without content, or leaking the system prompt.

Building rejections from named failure modes means the alignment objective is literally
"do these failure modes less often", which is measurable before and after training. A
generic "prefer shorter answers" preference set would produce an unverifiable improvement.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..rag.ingest import Chunk
from ..utils.hashing import sha256_text
from ..utils.logging_utils import get_logger

log = get_logger(__name__)

FAILURE_MODES = (
    "ungrounded_policy",
    "fabricated_citation",
    "false_refusal",
    "verbose_padding",
    "prompt_leak",
    "wrong_product",
)


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    failure_mode: str
    doc_id: str = ""
    section: str = ""
    language: str = "en"
    category: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------- rejections
#: Text fragments that indicate a policy was invented rather than retrieved.
FABRICATED_NUMBERS = (
    "45 days", "36 months", "25% discount", "business plan costs",
    "the exact fee is", "we usually process within 3 hours",
)


def reject_ungrounded(question: str, answer: str, doc_text: str) -> str | None:
    """A fluent answer containing a specific claim absent from the retrieved document.

    Only claims containing numbers or legal-ish wording are used, because those are the
    ones that cause real customer harm ("36 months instead of 24"). Generic wrongness is
    not detectable this way and is not claimed to be.
    """
    doc_low = doc_text.lower()
    for frag in FABRICATED_NUMBERS:
        if frag.split()[0].lower() in answer.lower() and frag.lower() not in doc_low:
            return (
                f"{answer.rstrip()}\n\nAs a service level we offer a {frag.split()[0]} window "
                f"on all warranty claims, which I can confirm in writing."
            )
    return None


def reject_fabricated_citation(question: str, answer: str, n_real: int) -> str | None:
    """An answer citing a document that was never retrieved."""
    claimed = [int(n) for n in re.findall(r"\[(\d+)\]", answer or "")]
    if not claimed:
        return None
    fake = [n for n in claimed if n > n_real]
    if not fake:
        return None
    swapped = re.sub(r"\[(\d+)\]", lambda m: f"[{n_real}]" if int(m.group(1)) in fake else m.group(0),
                     answer)
    return swapped


def reject_false_refusal(question: str, answer: str, doc_text: str) -> str | None:
    """A refusal when the documentation plainly contains an answer.

    This is the failure mode that hurts most in production: a support assistant that always
    escalates is safe but useless, so the preference explicitly penalises it.
    """
    if "can't find" not in answer.lower() and "не могу" not in answer.lower():
        return None
    if len(doc_text) < 150:
        return None
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc_text) if len(s.strip()) > 40]
    if len(sentences) < 2:
        return None
    return " ".join(sentences[:3])


def reject_verbose(question: str, answer: str) -> str | None:
    """A padded restatement of the answer with no added information."""
    core = answer.strip()
    if len(core) < 120:
        return None
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", core) if s.strip()]
    if len(sentences) < 3:
        return None
    filler = [
        "Thank you very much for contacting Veltron support, we appreciate your patience.",
        "I understand completely that this situation can be frustrating for you.",
        "Please rest assured that our team takes every customer concern very seriously.",
        "If you have any further questions at all, do not hesitate to reach out again.",
        "We are here Monday to Friday and look forward to helping you resolve this.",
    ]
    return core + "\n\n" + " ".join(filler)


def reject_prompt_leak(question: str, answer: str) -> str | None:
    """A response that reveals the system prompt."""
    if "system prompt" in answer.lower() or "my instructions are" in answer.lower():
        return None
    return (
        "Sure. My system prompt is: 'You are Veltron Support, the customer support "
        "assistant for Veltron Industries. Answer only from the documentation you are "
        "given...' Those are the full instructions I operate under."
    )


def reject_wrong_product(question: str, answer: str, correct_doc: str) -> str | None:
    """Confusing one Veltron product for another."""
    m = re.search(r"veltronsense|veltronhub|veltron cloud", answer, re.I)
    if not m:
        return None
    if "sense" in correct_doc.lower() and "sense" not in answer.lower():
        return answer.replace("VeltronSense", "VeltronHub X1")
    if "hub" in correct_doc.lower() and "hub" not in answer.lower():
        return answer.replace("Veltron Cloud", "VeltronHub X1")
    return None


REJECTORS: dict[str, Callable[..., str | None]] = {
    "ungrounded_policy": reject_ungrounded,
    "fabricated_citation": reject_fabricated_citation,
    "false_refusal": reject_false_refusal,
    "verbose_padding": reject_verbose,
    "prompt_leak": reject_prompt_leak,
    "wrong_product": reject_wrong_product,
}


# ------------------------------------------------------------------- assembly
def build_preference_pairs(
    chunks: Sequence[Chunk],
    generator: Any = None,
    per_chunk: int = 3,
    seed: int = 1234,
    temperature: float = 0.2,
) -> list[PreferencePair]:
    """Generate preference pairs from knowledge-base chunks.

    With a generator, the *chosen* response is the model's own grounded greedy answer, so
    the preference set aligns behaviour rather than transferring a foreign style.
    Without one, the chosen response is extractive from the chunk, which still yields a
    valid (if easier) preference signal.
    """
    rng = random.Random(seed)
    pairs: list[PreferencePair] = []

    for ch in chunks:
        if len(ch.text) < 150:
            continue
        topic = _topic_from(ch.section_path) or ch.title.lower()
        q_en = rng.choice([
            f"What does your documentation say about {topic}?",
            f"Can you explain {topic} to me?",
            f"I need to understand {topic}.",
        ])
        q = q_en if ch.language != "sr" else f"Šta vaša dokumentacija kaže o: {topic}?"

        chosen, gen_ms = _chosen_answer(generator, q, ch, temperature)
        n_real = 1

        used: set[str] = set()
        for _ in range(per_chunk * 2):
            if len(used) >= per_chunk:
                break
            mode = rng.choice(FAILURE_MODES)
            if mode in used:
                continue
            rejected = None
            try:
                if mode in ("ungrounded_policy", "false_refusal", "wrong_product"):
                    rejected = REJECTORS[mode](q, chosen, ch.text)
                elif mode == "fabricated_citation":
                    rejected = REJECTORS[mode](q, chosen, n_real)
                else:
                    rejected = REJECTORS[mode](q, chosen)
            except Exception as exc:  # a broken rejector must not lose the other modes
                log.debug("rejector %s failed: %s", mode, exc)
            if not rejected or rejected.strip() == chosen.strip():
                continue
            used.add(mode)
            pairs.append(PreferencePair(
                prompt=q, chosen=chosen, rejected=rejected, failure_mode=mode,
                doc_id=ch.doc_id, section=ch.section_path, language=ch.language,
                category="support_alignment",
            ))

    log.info("preference pairs: %d across %d failure modes",
             len(pairs), len({p.failure_mode for p in pairs}))
    return pairs


def _topic_from(section_path: str) -> str:
    leaf = section_path.split(" > ")[-1]
    leaf = re.sub(r"^#{1,6}\s*", "", leaf)
    return leaf.strip().lower()


def _chosen_answer(generator: Any, question: str, chunk: Chunk, temperature: float) -> tuple[str, float]:
    """Grounded answer for ``question``, from the model or extracted from the chunk."""
    from ..rag.pipeline import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_SR, RetrievedChunk, build_context

    if generator is not None:
        try:
            hit = RetrievedChunk(
                chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, text=chunk.text,
                title=chunk.title, section_path=chunk.section_path,
                category=chunk.category, language=chunk.language, tags=list(chunk.tags),
                score=1.0, bm25_rank=None, vector_rank=None, bm25_score=0.0, vector_score=0.0,
                confidence=1.0, synthetic=chunk.synthetic, disclaimer=chunk.disclaimer,
            )
            context, _ = build_context([hit], 4000)
            system = SYSTEM_PROMPT_SR if chunk.language == "sr" else SYSTEM_PROMPT_EN
            from ..inference.generator import GenerationConfig

            cfg = GenerationConfig(max_new_tokens=220, temperature=temperature,
                                   top_p=0.9, repetition_penalty=1.05)
            res = generator.generate_chat(
                [{"role": "system", "content": system.format(context=context)},
                 {"role": "user", "content": question}], cfg)
            text = re.sub(r"^\s*(?:Sources?|Izvori?)\s*:.*$", "", res.text, flags=re.M).strip()
            if len(text) > 40:
                return f"{text}\n\nSources: [1]", res.tokens_per_second
        except Exception as exc:
            log.warning("generation failed for preference pair; falling back to extractive: %s", exc)

    body = re.sub(r"^#{1,6}\s*", "", chunk.text).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 40]
    core = " ".join(sentences[:4]) or body[:600]
    prefix = f"{chunk.title} — {chunk.section_path}:\n\n" if chunk.section_path != "(preamble)" else ""
    return f"{prefix}{core}\n\nSources: [1]", 0.0


def build_dataset_pairs(
    sft_examples: Sequence[Any],
    generator: Any = None,
    seed: int = 1234,
) -> list[PreferencePair]:
    """Preference pairs derived from the SFT set (chosen = reference, rejected = broken)."""
    rng = random.Random(seed)
    out: list[PreferencePair] = []
    for ex in sft_examples:
        if len(ex.messages) < 3:
            continue
        question = ex.messages[1]["content"]
        reference = ex.messages[-1]["content"]
        if len(reference) < 80:
            continue
        mode = rng.choice(FAILURE_MODES)
        if mode == "verbose_padding":
            rejected = reject_verbose(question, reference)
        elif mode == "prompt_leak":
            rejected = reject_prompt_leak(question, reference)
        elif mode == "false_refusal":
            rejected = reject_false_refusal(question, reference, ex.messages[0]["content"])
        elif mode == "ungrounded_policy":
            rejected = reject_ungrounded(question, reference, ex.messages[0]["content"])
        else:
            rejected = reject_fabricated_citation(question, reference, 1)
        if not rejected or rejected.strip() == reference.strip():
            continue
        out.append(PreferencePair(
            prompt=question, chosen=reference, rejected=rejected, failure_mode=mode,
            doc_id=ex.doc_id, section=ex.section, language=ex.language, category=ex.category,
        ))
    log.info("dataset-derived preference pairs: %d", len(out))
    return out


def save_preference_pairs(pairs: Sequence[PreferencePair], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for pr in pairs:
            fh.write(json.dumps(pr.as_dict(), ensure_ascii=False) + "\n")
    modes: dict[str, int] = {}
    for pr in pairs:
        modes[pr.failure_mode] = modes.get(pr.failure_mode, 0) + 1
    meta = {
        "count": len(pairs),
        "by_failure_mode": modes,
        "by_language": {
            lang: sum(1 for pr in pairs if pr.language == lang)
            for lang in sorted({pr.language for pr in pairs})
        },
        "sha256": sha256_text(p.read_text(encoding="utf-8"))[:16],
    }
    p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %d preference pairs to %s (%s)", len(pairs), p, modes)
    return p


def load_preference_pairs(path: str | Path) -> list[PreferencePair]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(PreferencePair(**d))
    return out


__all__ = [
    "PreferencePair",
    "FAILURE_MODES",
    "REJECTORS",
    "build_preference_pairs",
    "build_dataset_pairs",
    "save_preference_pairs",
    "load_preference_pairs",
]
