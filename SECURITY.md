# Security Policy

## Reporting a vulnerability

**Do not open a public issue for a security report.** Send it privately to the maintainer
via the repository's security advisory page (`Security` → `Report a vulnerability`).

Include: what you found, how to reproduce it, and the impact. Please give reasonable time
for a fix before public disclosure.

## Known limitations of this build

VeltronLM ships as a **research demonstration**. The following are known and unaddressed:

| Limitation | Impact |
|---|---|
| **No authentication** | Anyone who can reach the port can generate text and read tickets |
| **No TLS** | Traffic is clear text |
| Rate limiting is per-process | N replicas multiply the limit |
| No persistent ticket storage | Lost on restart (a privacy property, not a feature) |
| No audit log | Request ids are returned, not persisted |
| CORS defaults to `*` | Any origin may call the API (credentials disabled) |
| Prompt-injection detection is a blocklist | A paraphrased attack can reach the model |

**VeltronLM must not be exposed to an untrusted network.** It binds to `127.0.0.1` by
default for this reason and prints a warning on any other bind address.

The full threat model, including what *is* enforced, is in `docs/security.md`.

## What is implemented

| Control | Notes |
|---|---|
| Deterministic safety gate before generation | Injection and credential requests are escalated **without calling the model** |
| PII filtering in the data pipeline | 12 patterns; credentials dropped, bulk contacts redacted |
| Log redaction | API keys, bearer tokens and emails stripped at the handler |
| Error containment | No stack traces or internal paths returned to clients |
| Input bounds | Pydantic length and range validation on every field |
| Rate limiting | Per-IP sliding window with `Retry-After` |
| CI secret scan | Credential-shaped strings fail the build |
| Non-root container | uid 10001; models mounted read-only |
| Citation validation | Fabricated citation numbers are detected, not displayed |

## Deployment checklist

Before putting this anywhere but loopback:

- [ ] Put a reverse proxy in front with TLS and authentication
- [ ] Set `VELTRON_CORS_ORIGINS` to the specific origins you need
- [ ] Move rate limiting to a shared store
- [ ] Add a request size limit at the proxy
- [ ] Add persistent, access-controlled ticket storage
- [ ] Add request logging with PII redaction and a defined retention period
- [ ] Run an independent red-team review of the RAG path

## Synthetic data

The bundled knowledge base describes a **fictional** company, Veltron Industries. Its
policies are invented for testing and are not legally binding. Every document is flagged
`synthetic: true` and every grounded API response carries `synthetic_data_notice: true`.
Do not present these policies as a real company's terms.
