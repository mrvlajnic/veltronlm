"""Ticket classification, entity extraction and escalation policy.

These are deterministic, auditable components rather than model calls. In customer support
the classification step must be inspectable and stable: an agent needs to know *why* a
ticket was routed where it went, and a wrong escalation is more expensive than a wrong
category.

The model is used only for phrasing the reply. Anything that decides a customer's money,
policy or account state is a rule, so it can be tested and explained.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..utils.logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- taxonomy
@dataclass(frozen=True)
class Category:
    key: str
    label: str
    description: str
    priority: int          # 1 = highest
    requires_human: bool = False
    tags: tuple[str, ...] = ()


CATEGORIES: tuple[Category, ...] = (
    Category("billing", "Billing and payments", "Charges, invoices, VAT, payment methods, renewals", 3,
             tags=("invoice", "charge", "charged", "payment", "refund", "vat", "tax", "bill",
                   "billing", "price", "cost", "subscription", "renew", "card", "receipt")),
    Category("technical_issue", "Technical fault", "Device or service not working as specified", 2,
             tags=("offline", "not working", "broken", "crash", "error", "fails", "failed",
                   "won't", "wont", "cannot", "can't", "problem", "issue", "fault", "bug",
                   "stuck", "unresponsive", "reset", "restart")),
    Category("account_access", "Account and access", "Login, password, pairing, lockout, permissions", 2,
             tags=("login", "log in", "sign in", "password", "locked", "lockout", "2fa",
                   "two-factor", "cannot access", "can't access", "forgot", "reset password",
                   "account", "authenticate", "pairing", "pair")),
    Category("bug_report", "Bug report", "Reproducible defect with expected vs actual behaviour", 2,
             tags=("bug", "defect", "reproduce", "expected", "actual", "steps to reproduce",
                   "wrong output", "incorrect", "always happens")),
    Category("refund_request", "Refund and return", "Return merchandise, refund, money back", 2,
             tags=("refund", "money back", "return", "returned", "send it back", "rma",
                   "reimburse", "cancel and refund")),
    Category("feature_request", "Feature request", "Request for behaviour the product lacks", 4,
             tags=("feature request", "would be great", "please add", "suggestion", "support for",
                   "can you add", "it would be nice", "wish it", "suggest",
                   "would love", "i propose", "add support for", "add an option")),
    Category("shipping", "Shipping and delivery", "Dispatch, tracking, delay, loss, customs", 2,
             tags=("shipping", "delivery", "delivered", "dispatch", "tracking", "courier",
                   "carrier", "post", "dhl", "package", "parcel", "lost", "delay", "late",
                   "customs", "not arrived")),
    Category("product_info", "Product information", "Specifications, compatibility, availability", 4,
             tags=("spec", "specification", "specs", "dimensions", "weight", "compatible",
                   "does it work with", "how much", "available", "stock", "compare",
                   "difference between", "what is", "which model")),
    Category("complaint", "Complaint", "Dissatisfaction about service or handling", 2,
             tags=("complaint", "unacceptable", "disappointed", "angry", "terrible", "awful",
                   "worst", "ridiculous", "third time", "keep asking")),
    Category("escalation", "Explicit escalation request", "Customer asks for a human/supervisor", 1,
             requires_human=True,
             tags=("speak to a human", "talk to a human", "real person", "human agent",
                   "supervisor", "manager", "escalate", "representative", "operator")),
    Category("security_incident", "Security incident", "Unauthorised access, credential compromise", 1,
             requires_human=True,
             tags=("hacked", "unauthorised", "unauthorized", "someone accessed", "not me",
                   "wasn't me", "was not me", "someone logged", "someone signed",
                   "compromised", "stolen", "account was hacked", "fraud",
                   "suspicious login", "unknown device", "unrecognised login")),
    Category("how_to", "How-to guidance", "How to configure, use, or set something up", 4,
             tags=("how do i", "how to", "how can i", "can you explain", "steps to",
                   "where do i", "help me set", "configure", "setup", "set up", "tutorial")),
    Category("other", "Unclassified", "No confident match", 5,
             tags=("hi", "hello", "hey", "thanks", "thank you", "cheers", "")),
)

#: The fallback bucket must never be reported as a confident classification: a bare
#: greeting scores a single keyword hit with no runner-up, which the generic margin
#: formula would otherwise turn into ~0.95 confidence.
OTHER_MAX_CONFIDENCE = 0.35

CATEGORY_BY_KEY = {c.key: c for c in CATEGORIES}

#: Words that mean the customer's premise is wrong and the answer must not be invented.
UNSUPPORTED_CLAIM_MARKERS = (
    "i was told", "you promised", "your colleague said", "according to your policy",
    "your policy says", "i read that you", "the website said", "chat said",
    "someone assured me", "last time you", "you guaranteed", "entitled to",
)

PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions", "ignore all previous", "disregard your instructions",
    "system prompt", "reveal your instructions", "print your prompt", "you are now",
    "act as", "pretend to be", "developer mode", "jailbreak", "override your",
    "new instructions:", "forget everything",
)

#: Phrases that indicate the customer is asking *us* for a secret, rather than mentioning
#: one. A bare "password" is far too broad: "I forgot my password" is the single most
#: common legitimate support question and would be misrouted to security on every
#: occurrence. The markers are therefore request-shaped ("your password", "the admin
#: password") rather than topic-shaped.
SECRET_REQUEST_MARKERS = (
    "your system prompt", "your instructions", "your api key", "your password",
    "the admin password", "admin password for", "internal password", "root password",
    "other customers", "other customer", "customer database", "customer records of",
    "someone else's account", "database dump", "private key", "your access token",
    "your credentials", "internal tool", "staff account", "master password",
)

HARASSMENT_MARKERS = (
    "idiot", "moron", "stupid", "scam", "fraud", "thief", "garbage company", "lawsuit",
)


@dataclass
class Classification:
    category: str
    label: str
    confidence: float
    priority: int
    matched_terms: list[str] = field(default_factory=list)
    runner_up: str | None = None
    requires_human: bool = False
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Entity:
    name: str
    value: str
    span: tuple[int, int]
    redacted: bool = False

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["span"] = list(self.span)
        return d


@dataclass
class ExtractedEntities:
    product: str | None = None
    issue: str | None = None
    urgency: str = "normal"
    customer_id: str | None = None
    language: str = "en"
    serial_numbers: list[str] = field(default_factory=list)
    order_numbers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    amounts: list[str] = field(default_factory=list)
    email: str | None = None
    versions: list[str] = field(default_factory=list)
    raw: list[Entity] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SafetySignals:
    prompt_injection: bool = False
    asks_for_secrets: bool = False
    unsupported_claim: bool = False
    harassment: bool = False
    personal_data_requested: bool = False
    matched: dict[str, list[str]] = field(default_factory=dict)

    @property
    def any_trigger(self) -> bool:
        return (self.prompt_injection or self.asks_for_secrets or self.unsupported_claim
                or self.harassment or self.personal_data_requested)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------- classification
_URGENCY_MARKERS = {
    "critical": ("urgent", "emergency", "asap", "immediately", "critical", "outage",
                 "data loss", "smoke", "fire", "safety"),
    "high": ("soon", "quickly", "waiting", "frustrated", "third time", "still not",
             "no response", "days ago", "blocked", "cannot work"),
}


def detect_safety_signals(text: str) -> SafetySignals:
    low = " " + text.lower() + " "
    sig = SafetySignals(matched={})
    for marker in PROMPT_INJECTION_MARKERS:
        if marker in low:
            sig.prompt_injection = True
            sig.matched.setdefault("prompt_injection", []).append(marker)
    for marker in SECRET_REQUEST_MARKERS:
        if marker in low:
            sig.asks_for_secrets = True
            sig.matched.setdefault("asks_for_secrets", []).append(marker)
    for marker in UNSUPPORTED_CLAIM_MARKERS:
        if marker in low:
            sig.unsupported_claim = True
            sig.matched.setdefault("unsupported_claim", []).append(marker)
    for marker in HARASSMENT_MARKERS:
        if marker in low:
            sig.harassment = True
            sig.matched.setdefault("harassment", []).append(marker)
    if "other customer" in low or "someone else" in low or "someone's account" in low:
        sig.personal_data_requested = True
        sig.matched.setdefault("personal_data_requested", []).append("other customer data")
    return sig


def classify_ticket(text: str, use_safety: bool = True) -> Classification:
    """Score every category by weighted keyword hits and return the best.

    Weighting a match by the specificity of its phrase ("cannot access" is far more
    diagnostic than "account") keeps generic words from dominating.
    """
    low = " " + text.lower().strip() + " "
    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}

    for cat in CATEGORIES:
        score = 0.0
        hits: list[str] = []
        for tag in cat.tags:
            if not tag:
                continue
            occurrences = low.count(tag)
            if occurrences:
                # Longer phrases are more specific and therefore more informative.
                weight = 1.0 + 0.55 * (len(tag.split()) - 1)
                score += occurrences * weight
                hits.append(tag)
        if score > 0:
            scores[cat.key] = score
            matched[cat.key] = hits

    signals = detect_safety_signals(text) if use_safety else SafetySignals()

    # Safety escalations outrank everything: a security incident misrouted to tier-1
    # billing wastes customer time and can hide a compromise.
    if signals.personal_data_requested or signals.asks_for_secrets:
        cat = CATEGORY_BY_KEY["security_incident"]
        return Classification(cat.key, cat.label, 0.95, cat.priority,
                              matched.get(cat.key, []), "escalation", True,
                              "Request touches credentials or another person's data.")
    if signals.prompt_injection:
        cat = CATEGORY_BY_KEY["escalation"]
        return Classification(cat.key, cat.label, 0.9, cat.priority,
                              list(signals.matched.get("prompt_injection", [])), None, True,
                              "Prompt-injection attempt detected in the message body.")
    if "escalation" in scores and scores["escalation"] >= 1.0:
        cat = CATEGORY_BY_KEY["escalation"]
        return Classification(cat.key, cat.label, 0.9, cat.priority,
                              matched["escalation"], "other", True,
                              "Customer explicitly asked for a human agent.")

    if not scores:
        cat = CATEGORY_BY_KEY["other"]
        return Classification(cat.key, cat.label, 0.2, cat.priority, [], None, cat.requires_human,
                              "No category keywords matched.")

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_key, best_score = ranked[0]
    cat = CATEGORY_BY_KEY[best_key]
    runner_up = ranked[1][0] if len(ranked) > 1 else None

    total = sum(scores.values())
    margin = (best_score - (ranked[1][1] if len(ranked) > 1 else 0.0)) / max(total, 1e-9)
    # Confidence combines share-of-evidence with separation from the runner-up, so a
    # single weak keyword match cannot be reported as high confidence.
    confidence = max(0.0, min(0.95, 0.35 * (best_score / total) + 0.65 * min(1.0, margin * 2)))
    if best_key == "other":
        # A greeting scores one keyword with no competition, which the generic formula
        # would report as near-certain. "Unclassified" is precisely the absence of
        # evidence, so its confidence is capped well below any real category.
        confidence = min(confidence, OTHER_MAX_CONFIDENCE)

    return Classification(
        category=cat.key,
        label=cat.label,
        confidence=round(confidence, 4),
        priority=cat.priority,
        matched_terms=matched.get(best_key, [])[:8],
        runner_up=runner_up,
        requires_human=cat.requires_human,
        rationale=f"matched {matched.get(best_key, [])[:5]} (score {best_score:.2f} of {total:.2f})",
    )


# ------------------------------------------------------------------- entity extraction
_SERIAL = re.compile(r"\b(?:S/N|SN|VH|VS|VR)[- ]?([A-Z0-9]{4,12})\b", re.I)
_ORDER = re.compile(r"\b(?:order|invoice|rma|ticket)(?:\s*(?:no\.?|number|id|#))?\s*[:#]?\s*([A-Z0-9][A-Z0-9\-]{3,15})\b", re.I)
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b", re.I)
_AMOUNT = re.compile(r"(\d+[.,]\d{2})\s*(?:EUR|€|USD|\$|RSD|din)?", re.I)
_VERSION = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?)\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_CUSTOMER_ID = re.compile(r"\b(?:account|customer|klijent|korisnik)\s*(?:id|no\.?|number)?\s*[:#]?\s*([A-Za-z0-9\-]{4,20})\b", re.I)
_PRODUCT_PATTERNS = (
    r"veltronhub\s*x1", r"veltronhub", r"veltronsense", r"veltron\s*cloud",
    r"veltron\s*app", r"vr-1", r"veltron",
)

_PRODUCT_RE = re.compile("|".join(_PRODUCT_PATTERNS), re.I)
_ISSUE_RE = re.compile(
    r"(?P<issue>(?:is|are|keeps|won't|will not|does not|doesn't|cannot|can't)\s+"
    r"(?:not\s+)?[a-z]+(?:ing)?(?:\s+[a-z]+){0,4})", re.I)


def extract_entities(text: str) -> ExtractedEntities:
    """Extract structured fields from a ticket.

    Personal identifiers are extracted for *routing* only; the support agent prompt is
    told never to echo a full card number or an email back to the customer.
    """
    from ..data.filters import detect_language

    out = ExtractedEntities()

    m = _PRODUCT_RE.search(text)
    if m:
        out.product = m.group(0).strip()
    else:
        for kw, canon in (("hub x1", "VeltronHub X1"), ("veltronsense", "VeltronSense"),
                          ("veltron cloud", "Veltron Cloud"), ("veltron app", "Veltron App"),
                          ("veltron", "Veltron product")):
            if kw in text.lower():
                out.product = canon
                break

    m = _ISSUE_RE.search(text)
    out.issue = " ".join(m.group("issue").split())[:120] if m else None

    low = text.lower()
    for level, markers in _URGENCY_MARKERS.items():
        if any(w in low for w in markers):
            out.urgency = level
            break
    if any(w in low for w in ("emergency", "safety", "fire", "smoke", "data loss")):
        out.urgency = "critical"

    m = _CUSTOMER_ID.search(text)
    out.customer_id = m.group(1) if m else None

    out.serial_numbers = [f"{m.group(0)}" for m in _SERIAL.finditer(text)][:5]
    out.order_numbers = [m.group(0) for m in _ORDER.finditer(text)][:5]
    out.dates = [m.group(1) for m in _DATE.finditer(text)][:5]
    out.amounts = [m.group(0).strip() for m in _AMOUNT.finditer(text)][:8]
    out.versions = [m.group(0) for m in _VERSION.finditer(text)][:5]
    m = _EMAIL.search(text)
    if m:
        out.email = "".join(ch if random_mask(i) else "*" for i, ch in enumerate(m.group(0)))

    lang, conf = detect_language(text)
    out.language = lang if lang in ("en", "sr") else "en"

    out.raw = [
        Entity("serial_number", v, (0, 0)) for v in out.serial_numbers
    ] + [
        Entity("order_number", v, (0, 0)) for v in out.order_numbers
    ]
    return out


def random_mask(i: int) -> bool:
    """Keep the first two characters of a string (the TLD is lost, the local part masked)."""
    return i < 2 or i > 6


# ----------------------------------------------------------------------- escalation
ESCALATION_TEMPLATE_EN = (
    "I don't have enough verified information to answer this safely, so I won't guess. "
    "I'm escalating this to a human support agent who can review it directly."
)
ESCALATION_TEMPLATE_SR = (
    "Nemam dovoljno proverenih informacija da bih bezbedno odgovorio, pa neću nagađati. "
    "Prosim da ovaj slučaj preuzme ljudski operater."
)
UNKNOWN_TEMPLATE_EN = (
    "That isn't covered by the documentation I have access to, so I can't give you a "
    "reliable answer. I can point you to a human agent who can check directly."
)

ESCALATION_REASONS = (
    "security_incident_category",
    "explicit_human_request",
    "prompt_injection_detected",
    "requests_credentials_or_third_party_data",
    "no_retrieval_above_threshold",
    "retrieved_answers_do_not_address_question",
    "contradictory_documentation",
    "out_of_warranty_quote_required",
    "low_classification_confidence",
    "model_low_answer_confidence",
)


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str | None
    confidence: float
    template_key: str = "unknown"
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Retrieval score below which the system refuses rather than guesses.
MIN_RETRIEVAL_SCORE = 0.18
#: Classification confidence below which a human should look at the ticket.
MIN_CLASSIFICATION_CONFIDENCE = 0.18


def decide_escalation(
    classification: Classification,
    signals: SafetySignals,
    retrieval_score: float | None = None,
    retrieved: int = 0,
    answer_confidence: float | None = None,
) -> EscalationDecision:
    """Single place where the system decides to hand off to a human.

    Every branch names a reason from :data:`ESCALATION_REASONS` so escalations can be
    audited and counted rather than being implicit in the prompt.
    """
    if signals.prompt_injection:
        return EscalationDecision(True, "prompt_injection_detected", 0.9, "escalate",
                                  {"matched": signals.matched.get("prompt_injection", [])})
    if signals.asks_for_secrets or signals.personal_data_requested:
        return EscalationDecision(True, "requests_credentials_or_third_party_data", 0.95,
                                  "escalate", {"matched": signals.matched.get("asks_for_secrets", [])})
    if classification.category in ("security_incident", "escalation"):
        return EscalationDecision(True, f"{classification.category}_category",
                                  classification.confidence, "escalate",
                                  {"category": classification.category})

    if retrieval_score is not None:
        if retrieved == 0 or retrieval_score < MIN_RETRIEVAL_SCORE:
            return EscalationDecision(True, "no_retrieval_above_threshold",
                                      classification.confidence, "unknown",
                                      {"retrieval_score": retrieval_score, "retrieved": retrieved})

    if answer_confidence is not None and answer_confidence < 0.15:
        return EscalationDecision(True, "model_low_answer_confidence",
                                  answer_confidence, "unknown", {})

    if classification.confidence < MIN_CLASSIFICATION_CONFIDENCE:
        return EscalationDecision(True, "low_classification_confidence",
                                  classification.confidence, "unknown", {})

    return EscalationDecision(False, None, classification.confidence, "none", {})


def escalation_message(language: str = "en", template_key: str = "unknown") -> str:
    if template_key == "escalate":
        return ESCALATION_TEMPLATE_SR if language == "sr" else ESCALATION_TEMPLATE_EN
    return UNKNOWN_TEMPLATE_EN


def describe_taxonomy() -> list[dict[str, Any]]:
    return [
        {
            "key": c.key,
            "label": c.label,
            "description": c.description,
            "priority": c.priority,
            "requires_human": c.requires_human,
            "keyword_count": len(c.tags),
        }
        for c in CATEGORIES
    ]


__all__ = [
    "CATEGORIES",
    "CATEGORY_BY_KEY",
    "Category",
    "Classification",
    "ExtractedEntities",
    "Entity",
    "SafetySignals",
    "EscalationDecision",
    "classify_ticket",
    "extract_entities",
    "detect_safety_signals",
    "decide_escalation",
    "escalation_message",
    "describe_taxonomy",
    "ESCALATION_REASONS",
    "ESCALATION_TEMPLATE_EN",
    "ESCALATION_TEMPLATE_SR",
    "MIN_RETRIEVAL_SCORE",
    "MIN_CLASSIFICATION_CONFIDENCE",
    "OTHER_MAX_CONFIDENCE",
]
