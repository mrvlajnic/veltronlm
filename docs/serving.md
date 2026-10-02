# Serving

FastAPI application. The chat UI is mounted on the same app, so the demo is one process on
one port with no CORS configuration.

```bash
python -m veltron.serve --checkpoint checkpoints/micro-pretrain
#  http://127.0.0.1:8000/chat     chat UI
#  http://127.0.0.1:8000/docs     OpenAPI
```

Docker: `docker compose -f docker/docker-compose.yml up --build`

---

## 1. Endpoints

| Method | Path | Purpose | Auth |
|---|---|---|---|
| GET | `/` | Service banner + endpoint list | none |
| GET | `/chat` | Chat UI (HTML) | none |
| GET | `/v1/health` | Status, load state, rate-limit stats | none |
| GET | `/v1/model` | Loaded model metadata | none |
| GET | `/v1/devices` | Compute backends | none |
| GET | `/v1/taxonomy` | The 14 support categories | none |
| GET | `/v1/stats` | Ticket counts, escalation rate | none |
| POST | `/v1/chat/completions` | Grounded chat completion | none |
| POST | `/v1/generate` | Raw generation (streaming via SSE) | none |
| POST | `/v1/search` | Raw retrieval | none |
| POST | `/v1/tickets` | Classify + extract + escalate | none |
| POST | `/v1/feedback` | Helpfulness rating | none |

### 1.1 Response shape for `/v1/chat/completions`

```json
{
  "id": "chatcmpl-3f9a2b1c8d7e6f50",
  "object": "chat.completion",
  "model": "veltronlm-micro",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "content_filter"
  }],
  "usage": {"prompt_tokens": 812, "completion_tokens": 96,
            "tokens_per_second": 14.2, "time_to_first_token": 0.21,
            "citation_check": {"claimed": [1], "grounded": [1], "fabricated": []}},
  "veltron": {
    "refused": false,
    "escalated": true,
    "citations": [{"index": 1, "title": "Warranty and RMA",
                   "section_path": "Warranty and RMA > Turnaround", "score": 0.87}],
    "classification": {"category": "refund_request", "confidence": 0.5},
    "escalation": {"escalate": true, "reason": "model_low_answer_confidence"},
    "latency_seconds": 0.31,
    "synthetic_data_notice": true,
    "retrieved": []
  }
}
```

`finish_reason: "content_filter"` signals an escalation, so OpenAI-compatible clients
treat it as a refusal rather than a normal completion.

## 2. Degradation without a checkpoint

The API starts and reports `"status": "degraded"`, `"model_loaded": false`. Retrieval and
ticket endpoints **keep working**; generation endpoints return `503` with a message naming
`VELTRON_CHECKPOINT`.

This is deliberate: the support triage layer is the most useful part during development and
does not need a model. A test asserts the 503.

## 3. Security

### 3.1 What is actually enforced

| Control | Implementation |
|---|---|
| **No authentication** | **None.** Documented in three places. The server warns on a non-loopback bind. |
| **Rate limiting** | Per-client-IP sliding window, 60 requests / 60 s by default. Returns 429 with `Retry-After`. |
| **Input bounds** | Pydantic `max_length` on every string field; `top_k` bounded to 1–25; `temperature` to [0, 2]; `max_new_tokens` to [1, 4096]. |
| **Log redaction** | `RedactingFilter` strips API keys, bearer tokens and email addresses at the handler, so a new `logger.info` cannot leak by accident. |
| **Error containment** | Unhandled errors return `{"error": "internal_error", "request_id": ...}`. Stack traces and internal paths never reach the client. |
| **Request correlation** | Every response carries `x-request-id` and `x-latency-ms`. Bodies are never logged. |
| **Prompt injection** | `detect_safety_signals` runs before generation; injection forces escalation and the model is never called. |
| **RAG isolation** | The knowledge base is a static local corpus, not a customer database, so cross-tenant retrieval leakage is structurally impossible. |
| **Credential requests** | Matched by request-shaped markers and escalated before generation. |
| **Non-root container** | `USER veltron` (uid 10001); models mounted read-only. |

### 3.2 What is NOT enforced

Stated plainly so nobody assumes otherwise:

* **No authentication or authorisation.** Anyone who can reach the port can generate and
  read tickets.
* **No TLS.** Loopback only by default.
* **Rate limiting is per-process.** A multi-replica deployment needs a shared store.
* **Tickets are held in memory** and lost on restart. That is a privacy feature, not a
  feature.
* **No audit log.** `x-request-id` is returned but not persisted.
* **CORS defaults to `*`** with credentials disabled. Tighten for any real deployment.

### 3.3 Running it anywhere but loopback

```bash
VELTRON_CHECKPOINT=... python -m veltron.serve --host 0.0.0.0
```

prints a warning to stderr. Put it behind a reverse proxy with TLS and authentication, or
keep it on loopback. A `VELTRON_AUTH_REQUIRED` environment variable exists as a signal to
deployment tooling but does not itself implement auth.

## 4. Chat UI

Single static HTML file at `veltron/chatbot/static/index.html`. No build step, no CDN, no
external assets — it works offline.

Features: streaming-friendly request flow, source list per answer, category/confidence/
synthetic-KB badges, escalation and refusal flags, latency and tokens/sec, a debug toggle
that dumps the full `veltron` payload, language selector (en/sr), clickable sample
questions, and periodic `/v1/health` polling.

The UI opens with a banner stating that Veltron Industries is fictional and the policies
are synthetic — the same disclosure the API carries in `synthetic_data_notice`.

## 5. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `VELTRON_CHECKPOINT` | *(unset)* | Checkpoint directory. Unset ⇒ degraded mode. |
| `VELTRON_TOKENIZER` | auto | Overrides the tokenizer resolved from checkpoint metadata. |
| `VELTRON_RAG_INDEX` | `models/rag_index` | Retrieval index. Built on startup if absent. |
| `VELTRON_DEVICE` | `auto` | `auto` / `cuda` / `directml` / `cpu` |
| `VELTRON_DTYPE` | `auto` | `auto` / `fp32` / `fp16` / `bf16` |
| `VELTRON_HOST` | `127.0.0.1` | |
| `VELTRON_PORT` | `8000` | |
| `VELTRON_RATE_LIMIT` | `60` | Requests per window per IP |
| `VELTRON_RATE_WINDOW` | `60` | Window seconds |
| `VELTRON_CORS_ORIGINS` | `*` | Comma-separated |
| `VELTRON_LOG_LEVEL` | `INFO` | |
| `VELTRON_LOG_JSON` | `0` | `1` for JSON logs |

## 6. Performance

Single request at a time. Generation throughput is bounded by the decoder, not the server;
`reports/inference_benchmark.json` has measured numbers. The KV cache is reused across
calls within a `Generator`, so a chat session avoids re-prefilling the system prompt only
if the caller reuses the cache — the `/v1/chat/completions` endpoint does **not**, because
caching across independent requests would leak context between them.

## 7. Known gaps

1. **No authentication.** Blocking for anything beyond localhost.
2. **No streaming for chat completions.** `/v1/generate` streams via SSE; chat
   completions does not, because the RAG pipeline needs the full retrieval before the first
   token. Streaming chat would require streaming retrieval.
3. **No request queue or backpressure.** Two concurrent generations contend for the GPU.
4. **No persistent ticket storage.** By design for privacy, but it means `/v1/stats` resets
   on restart.
5. **No metrics exporter.** `x-latency-ms` per request, but no Prometheus endpoint.
6. **The RAG index is built at startup** if missing, which adds ~2 s to first boot.