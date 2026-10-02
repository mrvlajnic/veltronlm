# Changelog

All notable changes. Versions follow `MAJOR.MINOR.PATCH` with a pre-release suffix.

## [0.1.0-alpha] — 2026-09-30

First working end-to-end system. Every component is implemented and tested; one component
is blocked by hardware and is reported as blocked.

### Added

**Model**
- Decoder-only Transformer: RoPE, grouped-query attention, RMSNorm with QK-norm, SwiGLU,
  pre-norm residuals, scaled residual initialisation
- 7 configurations from 19.7M to 4.03B parameters; parameter counts verified two ways
  (analytic algebra and `meta`-device instantiation), agreeing for all tiers
- KV cache with explicit-offset read/write, chunked prefill, growth by doubling
- Batched generation with left padding, verified token-identical to single-prompt
- fp32 logits at decode to avoid fp16 overflow above 65,504
- Sampling: temperature, top-k, top-p, min-p, repetition/presence/frequency penalties,
  no-repeat-ngram, stop tokens and strings, seeded determinism
- Group-wise 4- and 8-bit weight quantization with measured fidelity cost
- safetensors export with 2 GiB sharding and an HF-compatible config

**Tokenizer**
- Byte-level BPE trained from scratch, two vocabularies (32,768 and 57,611)
- GPT-2 byte-pre-tokenizer regex, Serbian Cyrillic and Latin support
- Role/control tokens, chat templates, per-slice compression metrics
- Lossless round trip verified on 12 script/format cases

**Data**
- Seven declared sources with SPDX licences checked before download; GPL excluded in code
- Acquisition: Wikimedia API, category traversal, Project Gutenberg catalogue,
  GitHub raw, curated synthetic knowledge base
- Quality pipeline: normalisation, boilerplate removal, 12-pattern PII drop/redact split,
  quality scoring, exact + MinHash-LSH near-duplicate detection, language ID
- Mixture resampling with upsampling counts reported, document-level splits, packing,
  uint16/uint32 shards with geometry-validated indices
- Versioned manifests with provenance, filter funnel and a content hash

**Training**
- Causal LM loop with gradient accumulation, warmup-cosine LR, weight-decay exclusion for
  norms, gradient clipping, per-token validation, throughput accounting
- Memory-bounded chunked cross-entropy
- Backend-aware precision: autocast on CUDA, fp32 on DirectML (no `aten::embedding` kernel)
- Canary gate refusing to train when the pipeline is broken
- Failure budget tracking OOM, NaN, gradient explosion, corrupt checkpoints, I/O errors
- Checkpoints with atomic writes, `COMPLETE` markers, torn-checkpoint skipping,
  rotation keeping the best, RNG restoration, resume with continuous LR
- Wall-clock budget, emergency checkpoint before an OOM re-raise

**Post-training**
- SFT with assistant-token-only loss masking and length bucketing
- 604-example SFT set: grounded support QA, refusals, escalations, general instructions
- DPO with a frozen reference model and margin/accuracy diagnostics
- Preference pairs generated from six named RAG failure modes

**Support and RAG**
- 15-document synthetic knowledge base, 45 heading-aware chunks with breadcrumbs
- BM25 + hashed n-gram hybrid retrieval with Reciprocal Rank Fusion
- Absolute retrieval confidence (lexical + z-scored embedding) for escalation gating
- Serbian-to-English query expansion
- Citation validation detecting fabricated reference numbers
- 14-category deterministic triage, entity extraction with masked emails, escalation with
  named reasons, English and Serbian refusal templates

**Serving**
- FastAPI: 10 endpoints, per-IP rate limiting, request correlation, JSON/human logging
- Streaming SSE generation, degradation to retrieval-only without a checkpoint
- Static chat UI with sources, badges, language selector and debug view
- Dockerfile and Compose stack, non-root, read-only model mounts

**Evaluation**
- 37 fixed evaluation items across support, adversarial and long-context suites
- Metrics with no learned judge; groundedness labelled as an upper bound
- Benchmark matrix including `not_evaluated` rows with reasons
- Canary, checkpoint, tokenizer and retrieval regression tests

**Engineering**
- 184 tests: 154 unit, 30 integration
- 13 experiment records, including two documenting failed designs
- 19 documents, one model card, one final report
- CI: lint, tests, parameter cross-check, API smoke, retrieval regression, secret scan

### Blocked

- **4B pretraining from random initialisation.** 60.00 GiB of AdamW optimizer state
  against ~10 GiB of usable VRAM (6.0x shortfall), and a ~22-year lower bound at the measured
  matmul peak. Reported with arithmetic in `experiments/EXP-0004.md`.
- **Benchmark comparison against Qwen or any other model.** No baseline was run on this
  host, so no competitive claim is made anywhere in this repository.

### Known limitations

- The trained model is 55.7M parameters on a 19.5M-token corpus — far from
  Chinchilla-optimal, and it does not produce useful open-ended text
- Serbian is ~1% of the corpus
- Retrieval is validated on 15 synthetic documents
- Groundedness is lexical containment
- The API has no authentication
- No toxicity filtering on the pretraining corpus
