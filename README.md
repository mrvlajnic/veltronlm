# VeltronLM

A real ~4B-parameter open-weight decoder-only Transformer and customer-support AI system.

Everything in this repository is implemented from scratch in PyTorch: the architecture, the
tokenizer, the dataset pipeline, the training loop, the retrieval stack, the API and the chat
UI. **No component calls an external model API.** The tokenizer and the weights are produced
locally, and every benchmark number is measured on the build host.

---

## The problem this is for

Customer-support language models fail in a specific, expensive way: they are fluent. A model
asked about a refund policy it has never seen will produce a confident, well-formatted,
entirely invented policy, and the customer acts on it.

Three research questions follow, and this repository is built to answer them:

1. **Can a deployment refuse safely?** A support model is only useful if "I don't know,
   here's a human" is a first-class outcome rather than a failure. That requires measuring
   refusal and escalation behaviour directly, not assuming a larger model handles it.

2. **What does language imbalance do to tokenization?** Serbian (Latin and Cyrillic) sits at
   roughly 1% of the pretraining corpus. Measured effect: Serbian Cyrillic costs 8% more
   tokens per character than English, and *23.6%* more at a 32k vocabulary than at 64k —
   the minority script gets crowded out of the merge table. That is a concrete allocation
   problem with a measurable fix.

3. **How much can one consumer GPU honestly do?** Not a rhetorical question: the hardware
   limits here are measured, and every capability claim is tied to a command.

Full numbers, including what is *not* established, are in
[`docs/RESULTS.md`](docs/RESULTS.md).

## What exists and what is trained

| | |
|---|---|
| Trained model | `veltronlm-mini`, **244,354,048** parameters, 26.6M tokens, train loss 6.15 → 3.85 |
| Verified architecture | `veltronlm-4b-base`, **4,026,765,312** parameters — verified, **not trained** |
| Hardware | 1x AMD Radeon RX 6700 XT (12 GiB), ~10 GiB usable, measured |

The 4B configuration is real, parameter-counted two independent ways, and unit-tested for
exact causality. Training it here would need 60 GiB of optimizer state against ~10 GiB of
VRAM — a 6.0x shortfall. It is reported as a constraint, not disguised.

## Status at a glance

| Component | Status | Evidence |
|---|---|---|
| 4B architecture | **IMPLEMENTED, TESTED** | 4,026,765,312 params, cross-checked on `meta` device |
| 4B parameter count | **VERIFIED** | analytic algebra == `meta` instantiation for all 7 tiers |
| 4B from-scratch pretraining | **BLOCKED** | 60 GiB optimizer state vs ~10 GiB usable VRAM; ~22-year lower bound |
| 4B inference | **IMPLEMENTED** | fits in fp16 (7.5 GiB) and int8 (3.75 GiB) on this GPU |
| 55.7M pretraining run | **BENCHMARKED** | live run, real loss curve, resumable checkpoints |
| Tokenizer | **TRAINED, TESTED** | 2 vocabularies, 0 round-trip failures across 12 script/format cases |
| Dataset pipeline | **TESTED** | `dataset-v1`, 19,485,297 train tokens, full provenance |
| RAG retrieval | **BENCHMARKED** | **hit@1 = 1.00, recall@3 = 1.00** on the 14-query support suite |
| SFT + DPO | **IMPLEMENTED, TESTED** | completion-masked loss, DPO objective unit-verified |
| API + chat UI | **TESTED** | 15 routes, 184 tests passing |
| Comparison vs Qwen or others | **NOT CLAIMED** | no baseline was run; no superiority is asserted |

The full accounting, including every limitation, is in
[`FINAL_REPORT.md`](FINAL_REPORT.md).

---

## The model

`veltronlm-4b-base` — decoder-only Transformer, pre-norm, grouped-query attention.

| Property | Value |
|---|---|
| Parameters | **4,026,765,312** (4.027 B) |
| Layers | 36 |
| Hidden size | 3,072 |
| Attention heads | 24 query / 8 KV (GQA group size 3) |
| Head dimension | 128 |
| FFN | 8,192 (SwiGLU) |
| Vocabulary | 65,536 |
| Context | 8,192 (16k variant with YaRN) |
| Normalisation | RMSNorm, with QK-norm |
| Positional encoding | RoPE |
| Attention bias | none |

Parameter breakdown — computed programmatically, never estimated:

| Component | Parameters | Share |
|---|---:|---:|
| Token embedding | 201,326,592 | 5.000% |
| Attention (q/k/v/o + QK-norm) | 905,968,896 | 22.500% |
| MLP (gate/up/down) | 2,717,908,992 | 67.496% |
| All norms | 233,472 | 0.006% |
| LM head | 201,326,592 | 5.000% |
| **Total** | **4,026,765,312** | **100%** |

See [`docs/model-architecture.md`](docs/model-architecture.md).

**These weights do not exist.** The architecture is real, instantiated and tested; 4B
pretraining is blocked by a 5× VRAM shortfall on this host. The trained model here is
`veltronlm-micro` (55,715,328 parameters).

---

## Quick start

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers safetensors numpy pydantic fastapi uvicorn python-multipart pytest

# Optional: use an AMD/Intel GPU on Windows or Linux
pip install torch-directml==0.2.5.dev240914 --no-deps
```

```powershell
python -m veltron.cli info          # backends, registry, which artefacts exist
python -m pytest tests -q           # 184 tests
```

### Build the retrieval index and ask a question (no model needed)

```powershell
python -m veltron.cli index
python -m veltron.cli ask "How long is the warranty on the VeltronHub X1?"
```

### Full pipeline

```powershell
python scripts/acquire_data.py --force --wikipedia-random 30
python scripts/fetch_gutenberg_bulk.py --limit 1200
python scripts/train_tokenizers.py
python scripts/build_dataset.py --seq-len 1024
python -m veltron.train --model micro --tokenizer models/tok-mini-32k \
    --run-name micro-pretrain --seq-len 1024 --batch-size 2 --grad-accum 8 \
    --max-steps 4000 --max-hours 5
```

### Post-training

```powershell
python -m veltron.finetuning.cli build-data
python -m veltron.finetuning.cli train --base checkpoints/micro-pretrain
python -m veltron.finetuning.cli build-preferences --checkpoint checkpoints/sft-support
python -m veltron.finetuning.cli dpo --init checkpoints/sft-support
```

### Evaluate

```powershell
python -m veltron.evaluate --retrieval
python -m veltron.evaluate --checkpoint checkpoints/micro-pretrain \
    --suites support adversarial --perplexity --retrieval --long-context
python -m veltron.evaluate --matrix      # includes not-evaluated rows
```

### Serve

```powershell
python -m veltron.serve --checkpoint checkpoints/micro-pretrain
# http://127.0.0.1:8000/chat
```

Docker: `docker compose -f docker/docker-compose.yml up --build`

---

## Results

All measured on: Windows 11, Intel i7-7700, AMD Radeon RX 6700 XT (12 GiB), PyTorch 2.4.1
via DirectML.

### Pretraining — `veltronlm-mini`, 244M params

| Step | Tokens | Train loss | Val loss | Val ppl |
|---:|---:|---:|---:|---:|
| 250 | 2.0M | 6.15 | 7.512 | 1830.2 |
| 1000 | 8.2M | 5.26 | 5.855 | 349.1 |
| 2000 | 16.4M | 5.14 | 3.736 | 41.9 |
| 3250 | 26.6M | **3.85** | 1.104 | 3.0 |

Throughput ~1,020 tokens/s (8,192 tokens/step, ~11.8 s/step), fp32, one consumer GPU.

**Treat the validation column with care.** It is computed on 6 documents / 8,176 tokens and
has fallen *below* the training loss, which means it is measuring an easy slice rather than
generalisation. The train-loss column is the trustworthy signal. `docs/RESULTS.md` §2.2
explains this in full.

An earlier 55.7M-parameter run (`micro`) is also complete and is kept for comparison:
train loss 6.15 -> 4.20, val perplexity 66.6 at step 1000.

### Retrieval — 14 held-out support queries

| Metric | Result |
|---|---|
| **hit@1** | **14/14 = 100%** |
| **recall@3** | **14/14 = 100%** |
| Absolute confidence, documented query | 0.88–0.97 |
| Absolute confidence, nonsense query | 0.21–0.24 |

### Tokenizer — measured compression

| Slice | `tok-mini-32k` | `tok-4b-64k` |
|---|---:|---:|
| English prose | 3.788 chars/token | 3.788 |
| Serbian Cyrillic | 3.484 | 3.484 |
| Serbian Latin | 2.346 | 2.346 |
| Python source | 4.032 | 4.032 |
| Round-trip failures | **0** | **0** |

The 64k vocabulary is measurably better on every slice (+3.9% English, **+23.6% Serbian
Cyrillic**), so it is not simply a larger table. Details in `docs/tokenizer.md`.

### Compute

| | |
|---|---|
| Measured matmul peak (fp16, synced) | 2.80 TFLOP/s |
| Measured training throughput | ~3,600 tokens/s |
| CPU matmul peak (fp32) | 0.21 TFLOP/s |
| GPU speedup over CPU | **7–35×** |

### What is NOT established

Stated here rather than left for a reviewer to discover:

* **No external baseline.** No Qwen-class or other model has been run. No comparative
  claim is made anywhere in this repository.
* **No instruction tuning has been run.** The model is a base model: it continues text, it
  does not follow instructions or chat.
* **The validation set is 6 documents.** It cannot support a generalisation claim.
* **Serbian is ~1% of the corpus** and its quality is unmeasured end to end.
* **Retrieval is validated on 15 synthetic documents**, one invented company.
* **No human evaluation.**

---

## Repository layout

```
veltron/
├── model/          architecture, config, safetensors I/O, quantization
├── tokenizer/      byte-level BPE training, chat templates, metrics
├── data/           sources, acquisition, quality filters, dedup, packing, loader
├── training/       schedules, checkpointing, trainer
├── finetuning/     SFT data construction, completion-masked trainer, CLI
├── alignment/      preference pairs from RAG failure modes, DPO
├── rag/            chunking, BM25 + hashed embeddings, RRF, grounded pipeline
├── support/        synthetic knowledge base, triage, entities, escalation
├── inference/      generator, engine loading, CLI
├── serving/        FastAPI application
├── chatbot/        static chat UI
├── evaluation/     suites, metrics, runner
├── utils/          devices, logging, config, hashing, seeding
├── benchmarks/     benchmark matrix tooling
├── train.py        pretraining CLI
├── evaluate.py     evaluation CLI
├── serve.py        server entry point
└── cli.py          unified command line
```

`scripts/` holds reproducible entry points: data acquisition, tokenizer training, dataset
builds, verification, benchmarks and reports.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/environment.md`](docs/environment.md) | Host audit, measured throughput, feasibility math |
| [`docs/architecture.md`](docs/architecture.md) | System design, data flow, design decisions |
| [`docs/model-architecture.md`](docs/model-architecture.md) | Exact parameter accounting, layer detail |
| [`docs/tokenizer.md`](docs/tokenizer.md) | Algorithm choice, training corpus, measured compression |
| [`docs/datasets.md`](docs/datasets.md) | Sources, licences, filter funnel, mixture, limitations |
| [`docs/training.md`](docs/training.md) | Loop, optimiser, precision, canary, throughput |
| [`docs/recovery.md`](docs/recovery.md) | Checkpoints, atomicity, resume procedures |
| [`docs/finetuning.md`](docs/finetuning.md) | SFT data, completion-masked loss |
| [`docs/alignment.md`](docs/alignment.md) | Preference construction, DPO objective |
| [`docs/rag.md`](docs/rag.md) | Retrieval, absolute confidence, grounding, citations |
| [`docs/customer-support.md`](docs/customer-support.md) | Triage, escalation, hallucination policy |
| [`docs/evaluation.md`](docs/evaluation.md) | Suites, metrics, benchmark matrix |
| [`docs/inference.md`](docs/inference.md) | Generation, determinism, quantization |
| [`docs/serving.md`](docs/serving.md) | API, chat UI, security posture |
| [`docs/security.md`](docs/security.md) | Threat model, enforced controls, explicit gaps |
| [`docs/roadmap.md`](docs/roadmap.md) | Phase status and next actions |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Symptoms, causes, fixes |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Intended use, limitations, measured results |
| [`docs/RESULTS.md`](docs/RESULTS.md) | **All measured numbers, with what they do not establish** |
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | Complete engineering accounting |
| [`DATA_SOURCES.md`](DATA_SOURCES.md) | Every source with licence and attribution |

---

## Honest limitations

1. **No 4B weights.** Architecture verified; pretraining blocked by a 5× VRAM shortfall.
2. **The trained model is 55.7M parameters**, heavily undertrained on a 19.5M-token
   corpus against a Chinchilla-optimal ~1.1B. It demonstrates the pipeline, not a capable
   model.
3. **Serbian is ~1% of the corpus.** Wikipedia rate limiting prevented a larger crawl. This
   is the highest-value data gap.
4. **Retrieval is validated on 15 synthetic documents.** Excellent on this corpus;
   says nothing about a 50,000-document knowledge base.
5. **Groundedness is lexical containment** — necessary but not sufficient. An upper bound,
   labelled as one in every report.
6. **No benchmark comparison against any existing model** was performed, so no competitive
   claim is made.
7. **The API has no authentication.** Loopback only.
8. **Toxicity filtering is not applied** to the pretraining corpus. A blocker for public
   release.

`FINAL_REPORT.md` states all of these with numbers.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The short version: one variable per experiment,
measure before claiming, and never describe an unmeasured capability as working.

## Licence

Code: Apache-2.0. See [`LICENSE`](LICENSE).
Model weights, when released: Apache-2.0, subject to the dataset licences in
[`DATA_SOURCES.md`](DATA_SOURCES.md) — note that the CC-BY-SA corpora impose share-alike on
redistributed derivatives.