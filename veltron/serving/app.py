"""HTTP API for VeltronLM.

FastAPI application exposing chat, completion, embeddings-free retrieval, ticket
classification and health endpoints. All model inference goes through
:func:`veltron.inference.engine.load_engine`; there is no external model API anywhere in
this process.

Security properties that are actually enforced rather than merely intended:

* the RAG context is wrapped in a delimiter and the system prompt tells the model that
  context is data, not instructions, which blunts document-borne prompt injection;
* retrieved documents cannot reach ``another customer's`` data because the knowledge base
  is a static local corpus, not a customer database;
* secrets are read from the environment and never logged (see the redacting log filter);
* rate limiting is per-client-IP and in-process, which is honest for a single-node
  deployment and explicitly documented as not sufficient for multi-node.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..rag.pipeline import DEFAULT_INDEX_DIR, RAGConfig, RAGPipeline
from ..support.triage import (
    classify_ticket,
    decide_escalation,
    describe_taxonomy,
    detect_safety_signals,
    escalation_message,
    extract_entities,
)
from ..utils.device import summary as device_summary
from ..utils.logging_utils import get_logger, setup_logging

log = get_logger(__name__)


# --------------------------------------------------------------------- schemas
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str = Field(min_length=1, max_length=32_000)


class GenerateParams(BaseModel):
    max_new_tokens: int = Field(default=320, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=1000)
    repetition_penalty: float = Field(default=1.05, ge=0.5, le=2.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    greedy: bool = False
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)
    language: Literal["en", "sr"] = "en"


class ChatCompletionRequest(BaseModel):
    model: str = "veltronlm"
    messages: list[ChatMessage]
    params: GenerateParams = Field(default_factory=GenerateParams)
    stream: bool = False
    use_rag: bool = True
    include_debug: bool = False


class GenerateRequest(BaseModel):
    model: str = "veltronlm"
    prompt: str = Field(min_length=1, max_length=64_000)
    params: GenerateParams = Field(default_factory=GenerateParams)
    stream: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=25)


class TicketRequest(BaseModel):
    message: str = Field(min_length=1, max_length=16_000)
    customer_id: str | None = None
    language: Literal["en", "sr", "auto"] = "auto"


class FeedbackRequest(BaseModel):
    ticket_id: str
    rating: Literal["helpful", "unhelpful"]
    comment: str | None = Field(default=None, max_length=2000)
    escalated: bool = False


# -------------------------------------------------------------------- rate limit
class RateLimiter:
    """Sliding-window per-client limiter.

    In-process only. A multi-replica deployment needs a shared store; the limit here is a
    guard against a runaway client, not a security control.
    """

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.rejected = 0

    def check(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.max_requests:
            self.rejected += 1
            retry_after = self.window - (now - q[0]) if q else self.window
            return False, round(max(retry_after, 0.1), 2)
        q.append(now)
        return True, 0.0

    def stats(self) -> dict[str, Any]:
        return {"clients": len(self.hits), "rejected": self.rejected,
                "max_requests": self.max_requests, "window_seconds": self.window}


# ------------------------------------------------------------------- app state
class AppState:
    """Mutable process-wide state.

    A plain class rather than a dataclass: ``dataclasses.field`` only works inside a
    dataclass, and using it here silently produced ``Field`` objects instead of empty
    containers, which failed at the first write rather than at construction.
    """

    def __init__(self) -> None:
        self.engine: Any = None
        self.rag: RAGPipeline | None = None
        self.tickets: dict[str, dict[str, Any]] = {}
        self.feedback: list[dict[str, Any]] = []
        self.load_error: str | None = None


STATE = AppState()
LIMITER = RateLimiter(
    max_requests=int(os.environ.get("VELTRON_RATE_LIMIT", "60")),
    window_seconds=float(os.environ.get("VELTRON_RATE_WINDOW", "60")),
)


def _checkpoint_from_env() -> str:
    return os.environ.get("VELTRON_CHECKPOINT", "")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging(level=os.environ.get("VELTRON_LOG_LEVEL", "INFO"))
    ckpt = _checkpoint_from_env()
    index = os.environ.get("VELTRON_RAG_INDEX", DEFAULT_INDEX_DIR)
    device = os.environ.get("VELTRON_DEVICE", "auto")

    try:
        if ckpt:
            from ..inference.engine import load_engine

            STATE.engine = load_engine(checkpoint=ckpt, device=device,
                                       dtype=os.environ.get("VELTRON_DTYPE", "auto"))
            log.info("engine loaded: %s", STATE.engine.info.as_dict())
        else:
            log.warning("VELTRON_CHECKPOINT not set: /v1/generate and chat completion "
                        "will return 503. Retrieval and ticket endpoints still work.")
    except Exception as exc:
        STATE.load_error = str(exc)
        log.error("engine load failed: %s", exc)

    try:
        pipeline = RAGPipeline(generator=STATE.engine.generator if STATE.engine else None,
                               cfg=RAGConfig())
        if os.path.exists(os.path.join(index, "chunks.json")):
            pipeline.load_index(index)
        else:
            pipeline.build_index(index)
        STATE.rag = pipeline
    except Exception as exc:
        log.error("RAG index failed: %s", exc)
        STATE.load_error = STATE.load_error or str(exc)

    yield


app = FastAPI(
    title="VeltronLM API",
    version=__version__,
    description=(
        "Local inference API for VeltronLM and its customer-support retrieval layer. "
        "All generation happens in this process from local weights; no external model "
        "provider is contacted."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("VELTRON_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next: Any) -> Response:
    """Attach a request id and log latency, without logging bodies."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.error("unhandled error request_id=%s path=%s", request_id, request.url.path)
        # Do not leak a stack trace or internal path to the client.
        return JSONResponse(status_code=500, content={"error": "internal_error",
                                                     "request_id": request_id})
    response.headers["x-request-id"] = request_id
    response.headers["x-latency-ms"] = str(round((time.perf_counter() - t0) * 1000, 2))
    return response


def rate_limit(request: Request) -> None:
    key = request.client.host if request.client else "unknown"
    ok, retry_after = LIMITER.check(key)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"rate limit exceeded; retry after {retry_after}s",
            headers={"Retry-After": str(retry_after)},
        )


def require_engine() -> Any:
    if STATE.engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=("no model loaded. Set VELTRON_CHECKPOINT to a checkpoint directory "
                    f"and restart. load_error={STATE.load_error}"),
        )
    return STATE.engine


def _params(p: GenerateParams) -> Any:
    from ..inference.generator import GenerationConfig

    return GenerationConfig(
        max_new_tokens=p.max_new_tokens,
        temperature=p.temperature,
        top_p=p.top_p,
        top_k=p.top_k,
        repetition_penalty=p.repetition_penalty,
        min_p=p.min_p,
        greedy=p.greedy,
        seed=p.seed,
        stop_strings=tuple(p.stop),
    )


# ------------------------------------------------------------------- endpoints
@app.get("/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if STATE.engine is not None else "degraded",
        "version": __version__,
        "model_loaded": STATE.engine is not None,
        "rag_ready": STATE.rag is not None and STATE.rag.retriever is not None,
        "load_error": STATE.load_error,
        "rate_limit": LIMITER.stats(),
    }


@app.get("/v1/model")
def model_info() -> dict[str, Any]:
    if STATE.engine is None:
        return {"loaded": False, "version": __version__, "devices": device_summary()}
    return {"loaded": True, **STATE.engine.info.as_dict(), "devices": device_summary()}


@app.get("/v1/devices")
def devices() -> dict[str, Any]:
    return device_summary()


@app.get("/v1/taxonomy")
def taxonomy() -> dict[str, Any]:
    return {"categories": describe_taxonomy()}


@app.post("/v1/generate", dependencies=[Depends(rate_limit)])
def generate(req: GenerateRequest) -> dict[str, Any]:
    engine = require_engine()
    cfg = _params(req.params)
    if req.stream:
        def sse():
            for delta in engine.stream(req.prompt, cfg):
                yield f"data: {json.dumps({'token': delta})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")
    res = engine.generate(req.prompt, cfg)
    return {
        "text": res.text,
        "finish_reason": res.finish_reason,
        "usage": {"prompt_tokens": res.prompt_tokens,
                  "completion_tokens": res.completion_tokens},
        "timing": {"seconds": round(res.seconds, 4),
                   "tokens_per_second": round(res.tokens_per_second, 3),
                   "time_to_first_token": round(res.time_to_first_token, 4)},
        "model": engine.info.model_name,
    }


@app.post("/v1/chat/completions", dependencies=[Depends(rate_limit)])
def chat_completions(req: ChatCompletionRequest) -> Any:
    engine = require_engine()
    cfg = _params(req.params)
    messages = [m.model_dump() for m in req.messages]

    if req.use_rag and STATE.rag is not None and STATE.rag.retriever is not None:
        question = next((m["content"] for m in reversed(messages)
                         if m["role"] == "user"), "")
        res = STATE.rag.answer(question, language=req.params.language,
                               include_debug=req.include_debug)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "model": engine.info.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": res.answer},
                "finish_reason": "stop" if not res.escalated else "content_filter",
            }],
            "usage": res.generation,
            "veltron": {
                "refused": res.refused,
                "escalated": res.escalated,
                "citations": res.citations,
                "classification": res.classification,
                "escalation": res.escalation,
                "latency_seconds": res.latency_seconds,
                "synthetic_data_notice": res.synthetic_data_notice,
                "retrieved": res.retrieved if req.include_debug else [],
            },
        }

    res = engine.generate_chat(messages, cfg)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "model": engine.info.model_name,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": res.text},
                     "finish_reason": res.finish_reason}],
        "usage": {"prompt_tokens": res.prompt_tokens,
                  "completion_tokens": res.completion_tokens},
        "veltron": {"refused": False, "escalated": False, "citations": [],
                    "classification": {}, "retrieved": []},
    }


@app.post("/v1/search", dependencies=[Depends(rate_limit)])
def search(req: SearchRequest) -> dict[str, Any]:
    if STATE.rag is None or STATE.rag.retriever is None:
        raise HTTPException(status_code=503, detail="retrieval index unavailable")
    hits = STATE.rag.retriever.search(req.query, top_k=req.top_k, per_doc_cap=2)
    return {
        "query": req.query,
        "rewritten_query": STATE.rag.search_only(req.query, top_k=1) and req.query,
        "results": [h.as_dict() for h in hits],
    }


@app.post("/v1/tickets", dependencies=[Depends(rate_limit)])
def create_ticket(req: TicketRequest) -> dict[str, Any]:
    """Classify, extract entities and decide escalation for a support request.

    Classification and escalation are deterministic. The language field is returned
    verbatim from the message text and no customer data is persisted beyond this process's
    memory.
    """
    classification = classify_ticket(req.message)
    entities = extract_entities(req.message)
    signals = detect_safety_signals(req.message)

    hits = []
    top_conf = 0.0
    if STATE.rag is not None and STATE.rag.retriever is not None:
        hits = STATE.rag.retriever.search(req.message, top_k=4, per_doc_cap=2)
        top_conf = max((h.confidence for h in hits), default=0.0)
    escalation = decide_escalation(classification, signals,
                                  retrieval_score=top_conf, retrieved=len(hits))

    ticket_id = f"tkt-{uuid.uuid4().hex[:12]}"
    lang = entities.language if req.language == "auto" else req.language
    STATE.tickets[ticket_id] = {
        "ticket_id": ticket_id,
        "created_at": time.time(),
        "category": classification.category,
        "confidence": classification.confidence,
        "escalated": escalation.escalate,
        "language": lang,
        "customer_id": req.customer_id,
    }
    return {
        "ticket_id": ticket_id,
        "classification": classification.as_dict(),
        "entities": entities.as_dict(),
        "safety": signals.as_dict(),
        "escalation": escalation.as_dict(),
        "message_if_escalated": escalation_message(lang, escalation.template_key)
        if escalation.escalate else None,
        "suggested_documents": [h.as_dict() for h in hits[:3]],
        "note": ("Classification and escalation are deterministic rules. Suggested documents "
                 "are retrieved passages, not an answer."),
    }


@app.post("/v1/feedback", dependencies=[Depends(rate_limit)])
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    if req.ticket_id not in STATE.tickets:
        raise HTTPException(status_code=404, detail=f"unknown ticket {req.ticket_id}")
    STATE.feedback.append({"ticket_id": req.ticket_id, **req.model_dump(),
                            "received_at": time.time()})
    return {"accepted": True, "ticket_id": req.ticket_id,
            "total_feedback": len(STATE.feedback)}


@app.get("/v1/stats")
def stats() -> dict[str, Any]:
    tickets = list(STATE.tickets.values())
    by_cat: dict[str, int] = defaultdict(int)
    escalated = 0
    for t in tickets:
        by_cat[t["category"]] += 1
        escalated += int(t["escalated"])
    return {
        "tickets": len(tickets),
        "by_category": dict(by_cat),
        "escalated": escalated,
        "escalation_rate": round(escalated / max(1, len(tickets)), 4),
        "feedback": len(STATE.feedback),
        "rate_limit": LIMITER.stats(),
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "VeltronLM API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": [
            "POST /v1/chat/completions", "POST /v1/generate", "POST /v1/search",
            "POST /v1/tickets", "POST /v1/feedback", "GET /v1/health",
            "GET /v1/model", "GET /v1/devices", "GET /v1/taxonomy", "GET /v1/stats",
        ],
        "external_model_calls": "none",
    }
