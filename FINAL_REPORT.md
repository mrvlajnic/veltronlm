# FINAL REPORT — VeltronLM 0.1.0-alpha

**Date:** 2026-10-01
**Commit:** `3ef258e` (+ `1c66b71`)
**Host:** Windows 11 · Intel i7-7700 · AMD Radeon RX 6700 XT (12 GiB) · PyTorch 2.4.1 via DirectML
**Tests:** 181 passing (151 unit, 30 integration)

---

## 1. Executive summary

A complete ~4B-parameter decoder-only Transformer and customer-support AI system,
implemented from scratch in PyTorch. Every component is real, tested and measured.
Nothing calls an external model API.

**One thing could not be done: train the 4B model from random initialisation.** The
architecture has exactly **4,026,765,312 verified parameters** and no weights, because
AdamW with fp32 master weights needs **60.00 GiB** against **~10 GiB** of usable VRAM. That is a
5.0× shortfall. It is reported with arithmetic rather than as an excuse.

The response was to build everything and train the largest tier that fits:
**`veltronlm-micro`, 55.7M parameters**, which reached **validation perplexity 66.6** over
18.4M tokens.

| Claim | Verdict |
|---|---|
| 4B architecture implemented and verified | **YES** — exact causality, verified twice-over param count |
| 4B weights trained | **NO — BLOCKED** (60 GiB vs 12 GiB) |
| A real model trained | **YES** — 55.7M params, val ppl 66.6 |
| Tokenizer trained from scratch | **YES** — 2 vocabularies, 0 round-trip failures |
| Dataset pipeline with licences | **YES** — 19,485,297 tokens, full provenance |
| RAG retrieval works | **YES** — 100% hit@1, 100% recall@3 |
| SFT/DPO implemented | **YES** — but **no SFT/DPO run completed** |
| Beats Qwen or any model | **NOT CLAIMED** — no baseline was ever run |

---

## 2. Measured results

### 2.1 Pretraining — `veltronlm-micro`

| Step | Tokens seen | Train loss | Val loss | Val perplexity |
|---:|---:|---:|---:|---:|
| 0 | 0 | 9.55 (`ln V`) | — | — |
| 250 | 4,096,000 | 6.087 | 5.762 | 317.9 |
| 500 | 8,192,000 | 4.555 | 5.101 | 164.3 |
| 750 | 12,288,000 | 5.414 | 4.564 | 96.0 |
| 1000 | 16,384,000 | — | **4.198** | **66.6** |
| 1125 | 18,432,000 | — | (running) | — |

Monotone improvement across four evaluations. Gradient norm declined 2.60 → ~0.50 and
stayed bounded; no instability, no NaN.

**Throughput:** ~3,300–3,700 tokens/s (16,384 tokens/step, ~14 s/step).

### 2.2 Architecture — verified two independent ways

`veltronlm-4b-base` = **4,026,765,312** parameters. The analytic tensor algebra and a
`meta`-device instantiation of the real module agree for all 7 registered tiers.

| Component | Parameters | Share |
|---|---:|---:|
| Token embedding | 201,326,592 | 5.000% |
| Attention (q/k/v/o + QK-norm) | 905,968,896 | 22.500% |
| MLP (SwiGLU) | 2,717,908,992 | 67.496% |
| All norms | 233,472 | 0.006% |
| LM head | 201,326,592 | 5.000% |
| **Total** | **4,026,765,312** | **100%** |

### 2.3 Correctness

| Property | Measured |
|---|---|
| Causality — logit delta before an edit | **exactly 0.0** |
| Attention mass beyond the causal window | **exactly 0.0** |
| KV-cache decode vs full forward | **9.239e-07** |
| Batched left-padded vs single-prompt logits | **5.513e-07** |
| Batched vs single-prompt token output | **identical** |
| Gradients received by all 55.7M parameters | **yes, none zero** |
| Seeded sampling reproducibility | **identical** |
| Save/load output identity | **identical** |

### 2.4 Retrieval — 14 held-out support queries

| Metric | Result |
|---|---|
| **hit@1** | **14/14 = 100%** |
| **recall@3** | **14/14 = 100%** |
| Confidence, documented query | 0.88–0.97 |
| Confidence, nonsense query | 0.21–0.24 |

### 2.5 Tokenizer

| Slice | `tok-mini-32k` | Round-trip failures |
|---|---:|---:|
| English prose | 3.788 chars/token | **0** |
| Serbian Cyrillic | 3.484 | **0** |
| Serbian Latin | 2.346 | **0** |
| Python source | 4.032 | **0** |
| JSON / structured | 3.597 | **0** |

The 64k vocabulary saturated at 57,611, and it **is** measurably better than the 32k one
on every measured slice: +3.9% English, +23.6% Serbian Cyrillic, +3.1% Python.

### 2.6 Compute

| | |
|---|---|
| Matmul peak, fp16, **synchronised** | **2.80 TFLOP/s** |
| Matmul peak, CPU fp32 | 0.21 TFLOP/s |
| GPU speedup | **7–35×** |
| Training throughput | ~3,600 tokens/s |
| Accelerator nodes | 1 |

An earlier unsynchronised measurement reported 53.156 TFLOP/s — **19× too high**. All
figures use a blocking device→host read.

### 2.7 Dataset

19,485,297 train tokens · 383,615,628 characters · 1,312 documents kept of 1,380 seen
(95.1% accept) · 3.593 chars/token · manifest hash `272e134cf74c7367`.

---

## 3. The blocker, with arithmetic

### Memory

| Tensor | Size |
|---|---:|
| fp32 parameters | 15.00 GiB |
| fp32 gradients | 15.00 GiB |
| Adam `m` | 15.00 GiB |
| Adam `v` | 15.00 GiB |
| **Minimum optimizer state** | **60.00 GiB** |

Available: **12 GiB**. Shortfall **5.0×**, before activations or fragmentation.

### Time

Chinchilla-optimal (20 tokens/param) for 4B: 8.06e10 tokens = 1.95e21 FLOPs.

| Assumption | Runtime |
|---|---:|
| Raw matmul peak, 100% model FLOPs utilisation | **~22 years** |
| Measured end-to-end efficiency | **~710 years** |

### What would unblock it

| Option | Requirement |
|---|---|
| 8× A100/H100 80GB | ≥ 640 GB aggregate, FSDP/ZeRO-3 |
| 1× A100 80GB + CPU offload | 2–4× slower |
| LoRA/QLoRA from an existing 4B base | An open-weight base; **not** pretraining |
| Gradient accumulation | **Does not help** — this is optimizer *state*, not activations |
| More time on this GPU | **Does not help** — the memory gap is absolute |

### 4B inference is not blocked

| Precision | Weights | Fits 12 GiB |
|---|---:|---|
| bf16 / fp16 | 7.50 GiB | **yes** |
| int8 | 3.75 GiB | **yes** |
| int4 | 1.88 GiB | **yes** |

---

## 4. Bugs found and fixed

Nineteen, all regression-tested. Listed because they are the non-obvious traps.

| Bug | Symptom | Root cause |
|---|---|---|
| Byte tokenizer mangles non-ASCII | Serbian → `"    .  1  ."` | Latin-1 vocab instead of GPT-2 `bytes_to_unicode` |
| Byte tokenizer drops spaces | `"TheVeltronHubX1"` | `add_tokens` bypasses pre-tokenization |
| Shard dtype mismatch | `IndexError: index 2172472 ... size 32768` | Index hardcoded `uint32`, data `uint16` |
| KV cache reads uninitialised memory | Wrong prefill | Length tracked internally, not passed as offset |
| Batched generation diverges | batch ≠ solo | Padding mask dropped on decode steps |
| `normalize` corrupts code | Indents 8 → 2 | Leading whitespace collapsed |
| MinHash irreproducible | Dedupe changes every run | `hash()` randomised per process |
| Empty-matching regex | Words split per character | Alternation with `*` on an empty branch |
| Retrieval confidence always high | Escalation never fires | Rank-normalised score used as absolute |
| Augmentation manufactures evidence | Swallow question scored 0.725 | Augmented query fed the confidence calc |
| `tokenize` drops Cyrillic | `['24']` | ASCII `[a-z]`; `re.IGNORECASE` doesn't widen it |
| `field()` outside a dataclass | `'Field' object does not support item assignment` | `AppState` was a plain class |
| Checkpoint stores train loss as `val_loss` | Best-checkpoint picked by train-vs-train | Interval save had no evaluation |
| SFT summaries are boilerplate | Targets were `[Illustration]` | No boilerplate filtering in extraction |
| `other` reports 0.95 confidence | `"hello"` → certain "Unclassified" | Margin formula with no runner-up |
| "I forgot my password" → `security_incident` | Most common question misrouted | Bare `password` in secret markers |
| Query rewrite deletes the subject | `my VeltronHub X1 is not working` → `working` | Stripped framing, not just filler |
| Canary false failures ×2 | Refused to train correct code | Sampled too widely; then windows too long |
| Acquisition dispatch wrong key | All 4 sources wrote the KB | Compared `name` against function names |

---

## 5. What was implemented but NOT run

Stated so nothing is overclaimed:

| Item | State |
|---|---|
| **SFT run** | Pipeline tested, 604 examples built, **never trained** |
| **DPO run** | Objective unit-verified, **never trained** |
| **Full model evaluation** | Suites exist, **never run against a checkpoint** |
| **4B instantiation benchmark** | Segfaulted under GPU contention, **never completed** |
| **Clean compute benchmark** | Contaminated by concurrent training, **needs re-run** |
| **`exp_lr` / `exp_mixture` / `exp_tokenizer` sweeps** | Scripts written, **never run** |
| **Human evaluation** | Interface designed, **no raters** |
| **Baseline comparison (Qwen etc.)** | **Never run** — so no comparative claim |
| **CI** | Written, **never executed** |
| **Docker image** | Written, **never built** |
| **4B-long (16k context)** | Architecture only, **never trained or evaluated** |

---

## 6. Limitations

### Model

1. **Heavily undertrained.** 55.7M params, 18.4M tokens, against Chinchilla-optimal
   ~1.1B — a factor of ~60 short. Perplexity 66.6 is far from useful open-ended text.
2. **Two-thirds of parameters are embeddings.** At a 32k vocabulary, 33.5M of 55.7M are
   the embedding and output matrices; ~22M is real transformer capacity.
3. **Will confabulate.** RAG mitigates this; the raw model does not.

### Data

4. **Serbian is ~1% of the corpus** (21 documents). The largest quality ceiling. Wikipedia
   rate-limits anonymous clients (HTTP 429).
5. **Corpus is 98.7% Project Gutenberg by character.**
6. **Up-sampling is extreme** — the 15-document support KB repeats ~116 times, so it is
   memorised rather than generalised. Retrieval carries domain behaviour, not weights.
7. **No toxicity filtering** on the pretraining corpus. **Blocker for public release.**
8. **`test` split has 2 documents.**

### Retrieval

9. **Validated on 15 synthetic documents.** Says nothing about 50,000.
10. **Hashed embeddings, not semantic.**
11. **`MIN_RETRIEVAL_SCORE = 0.18` is fitted to this corpus**, no theoretical basis.
12. **No contradiction detection** between retrieved documents.

### Serving

13. **No authentication.** Loopback only.
14. **No streaming chat completions** (retrieval must complete first).
15. **Rate limiting per-process.**

---

## 7. Repository

```
veltron/         55 files  model, tokenizer, data, training, finetuning, alignment,
                         rag, support, inference, serving, chatbot, evaluation, utils
docs/            17 documents
experiments/     13 records + template + index
scripts/         15 reproducible entry points
tests/           184 tests
configs/         8 YAML configs
```

Git: **`3ef258e` on `main`, 154 files, committed but NOT pushed** — no remote is
configured and no `gh` CLI or token exists in this environment. See `HANDOFF.md` §13.

---

## 8. Recommended next actions

1. **Finish the live run, then evaluate it** — the highest-value single step.
2. **Run SFT → DPO → full evaluation** on the trained checkpoint.
3. **More Serbian data.** Largest quality ceiling; needs a permitted bulk source.
4. **Re-benchmark on an idle GPU**, including 4B int8 (3.75 GiB, fits).
5. **Run the three designed sweeps.** No sweep result is currently claimed.
6. **Red-team the RAG path.** Injection detection is a blocklist — a known weakness.
7. **Add toxicity filtering.** Required before any public release.
8. **Baseline comparison.** Only then is any comparative statement defensible.

---

## 9. Verification

```powershell
$env:PYTHONPATH = "E:\Posao\testmaxspace"
python -m pytest tests -q                    # 181 passed
python scripts/verify_param_count.py         # 4b = 4,026,765,312, analytic == meta
python scripts/verify_rag.py                 # hit@1 14/14, recall@3 14/14
python scripts/verify_checkpoints.py checkpoints/micro-pretrain
python -m veltron.cli info
```

Start at [`HANDOFF.md`](HANDOFF.md).