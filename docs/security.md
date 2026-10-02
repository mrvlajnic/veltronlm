# Security

Threat model, what is enforced, and what is explicitly not.

---

## 1. Trust boundaries

```
Internet / LAN
    │
    │  ← NO authentication, NO TLS. Loopback only by default.
    ▼
┌──────────────────────────────────────────────┐
│ FastAPI app                                 │
│  rate limiter (per-IP, in-process)           │
│  request-id + latency headers                │
├──────────────────────────────────────────────┤
│ Safety gate  ← customer message             │
│  detect_safety_signals()                     │
│  ├ prompt injection      → escalate, no LLM  │
│  ├ credential request    → escalate, no LLM  │
│  └ third-party data      → escalate, no LLM  │
├──────────────────────────────────────────────┤
│ Triage (deterministic)                       │
│  classify_ticket / extract_entities          │
│  decide_escalation                           │
├──────────────────────────────────────────────┤
│ Retrieval                                   │
│  static local knowledge base                 │
│  (NOT a customer database — see §4)          │
├──────────────────────────────────────────────┤
│ VeltronLM (local weights)                    │
│  grounded on retrieved context only          │
├──────────────────────────────────────────────┤
│ Post-generation checks                       │
│  verify_citations → fabricated citations     │
│  numeric claim audit → unsupported claims    │
└──────────────────────────────────────────────┘
    │
    ▼
In-memory ticket store (lost on restart)
```

## 2. Enforced controls

| Control | Where | Test |
|---|---|---|
| Rate limiting, 60 req/60 s per IP, 429 + `Retry-After` | `veltron/serving/app.py` | — |
| Input length bounds on every field | Pydantic schemas | `test_api_rejects_bad_request` |
| Parameter ranges (`temperature`, `top_p`, `top_k`, `max_new_tokens`) | Pydantic schemas | `test_api_rejects_bad_request` |
| Secret redaction in logs | `RedactingFilter` | — |
| Stack traces never returned | request middleware | — |
| Request correlation | `x-request-id`, `x-latency-ms` | — |
| Prompt injection → escalation before generation | `detect_safety_signals` | `test_prompt_injection_forces_escalation`, `test_api_ticket_escalates_injection` |
| Credential / third-party data → security category | `classify_ticket` | `test_credential_request_is_security_incident` |
| Fabricated citations detected | `verify_citations` | `test_fabricated_citation_is_detected` |
| Unsupported numeric claims detected | `score_support_item` | — |
| PII filtering in the training corpus | 12 patterns, drop/redact split | 20 tests in `test_data.py` |
| Non-root container, read-only model mounts | `docker/Dockerfile` | — |
| CI secret scan | `.github/workflows/ci.yml` | — |
| Contact details masked on entity extraction | `extract_entities` | `test_extract_entities_masks_email` |
| Agent prompt forbids requesting 2FA codes | `policy-security` KB document | — |

## 3. NOT enforced

Stated plainly. Each of these is a blocker for any deployment beyond localhost.

| Gap | Consequence | Required before |
|---|---|---|
| **No authentication** | Anyone reaching the port can generate and read tickets | Internet exposure |
| **No TLS** | Traffic in clear text | Internet exposure |
| **Rate limiting is per-process** | N replicas = N× the limit | Horizontal scaling |
| **Tickets in memory only** | Lost on restart; also means no durable audit trail | Any compliance requirement |
| **No audit log** | `x-request-id` is returned, not persisted | Incident response |
| **CORS defaults to `*`** | Any origin may call the API (credentials disabled) | Any browser client |
| **No request size limit at the socket level** | Bounded by Pydantic only after body read | Public exposure |
| **No output filtering** | Generated text is returned verbatim | Any customer-facing use |

## 4. RAG isolation

The retrieval layer reads a **static local knowledge base** of 15 synthetic policy documents.
It is not a customer database, so:

* **Cross-tenant leakage is structurally impossible** — there are no tenants to leak
  between. This is a property of the architecture, not a control that could be misconfigured.
* **No customer data enters the index.** Ingestion reads files from `knowledge_base/`.
* **Knowledge-base content is treated as data, not instructions.** The system prompt states
  this explicitly and the adversarial suite includes a document-borne injection
  (`adv-011`: quoted troubleshooting text that tries to override the refund policy).

If a real customer database were ever indexed, **cross-tenant isolation would become a
requirement with no current implementation.** That gap is recorded in `docs/roadmap.md`.

## 5. Prompt injection

### 5.1 Detection

12 markers covering instruction override (`ignore all previous instructions`), role
reassignment (`you are now`, `act as`, `pretend to be`), prompt extraction (`system prompt`,
`print your prompt`, `reveal your instructions`), and jailbreak framing (`developer mode`,
`jailbreak`).

### 5.2 Response

Injection **short-circuits before the model runs**. The pipeline returns an escalation
message and the generator is never called. This matters: a model asked to ignore its
instructions may comply, so the defence cannot depend on the model refusing.

### 5.3 Residual risk

Markers are a blocklist. An attacker who paraphrases without any known marker will reach
the model. The defence-in-depth layers that remain are the grounded-only prompt, the
escalation check on retrieval confidence, and citation validation — but no claim is made
that this is sufficient against a determined attacker. A red-team evaluation is an open
action in `docs/roadmap.md`.

## 6. Secret management

* All secrets come from the environment. No `.env` file is committed, and
  `.gitignore` excludes `.env*`.
* `RedactingFilter` is attached to the **handler**, so a new `logger.info(f"{token}")`
  cannot leak by accident. Patterns cover `api_key`, `secret`, `password`, `token`,
  `authorization`, `bearer`, and email addresses.
* **CI scans tracked files** for credential-shaped strings: AWS keys, GitHub tokens, Slack
  tokens, PEM private keys and JWTs.
* Checkpoints contain only weights, optimizer state and metadata. **No secret is ever
  written to a checkpoint**, because none is ever read into one.

## 7. Data protection

| Control | Detail |
|---|---|
| PII filtering | 12 patterns; credentials and unique financial identifiers **drop** the document; bulk contact details are **redacted** |
| No private data in training | Only public-domain and permissively-licensed corpora plus project-authored synthetic data |
| Synthetic data labelled | Every KB document carries `synthetic: true`; every grounded API answer carries `synthetic_data_notice` |
| Licence discipline | Sources are declared with SPDX ids before download; GPL-2.0 is excluded in code |
| Ticket data in memory only | Lost on restart. Not exported anywhere |

## 8. Reporting a vulnerability

See `SECURITY.md`. Please do not open a public issue for a security report.

## 9. Honest assessment

The controls above are real and tested, and the most important design decision — **the
deterministic gate runs before the model** — is the right one. But this service has **no
authentication**, which makes it unsuitable for anything beyond a trusted local network
regardless of how good the rest is. Treat it as a research demonstration, not a deployable
support product.