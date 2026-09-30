# VeltronLM

A real ~4B-parameter open-weight decoder-only Transformer and customer-support AI system.

Everything in this repository is implemented from scratch in PyTorch: the architecture, the
tokenizer, the dataset pipeline, the training loop, the retrieval stack, the API and the chat
UI. **No component calls an external model API.** The tokenizer and the weights are produced
locally, and every benchmark number is measured on the build host.

---

## Status at a glance

| Component | Status | Evidence |
|---|---|---|
| 4B architecture | **IMPLEMENTED, TESTED** | 4,026,765,312 params, cross-checked on `meta` device |
| 4B parameter count | **VERIFIED** | analytic algebra == `meta` instantiation for all 7 tiers |
| 4B from-scratch pretraining | **BLOCKED** | 60 GiB optimizer state vs 12 GiB VRAM; ~22-year lower bound |
| 4B inference | **IMPLEMENTED** | fits in fp16 (7.5 GiB) and int8 (3.75 GiB) on this GPU |
| 55.7M pretraining run | **BENCHMARKED** | live run, real loss curve, resumable checkpoints |
| Tokenizer | **TRAINED, TESTED** | 2 vocabularies, 0 round-trip failures across 12 script/format cases |
| Dataset pipeline | **TESTED** | `dataset-v1`, 19,485,297 train tokens, full provenance |
| RAG retrieval | **BENCHMARKED** | **hit@1 = 1.00, recall@3 = 1.00** on the 14-query support suite |
| SFT + DPO | **IMPLEMENTED, TESTED** | completion-masked loss, DPO objective unit-verified |
| API + chat UI | **TESTED** | 15 routes, 181 tests passing |
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
python -m pytest tests -q           # 181 tests
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

### Pretraining — `veltronlm-micro`, 55.7M params

| Step | Tokens seen | Train loss | Val loss | Val perplexity |
|---:|---:|---:|---:|---:|
| 0 | 0 | 9.55 (`ln(V)`) | — | — |
| 250 | 4,096,000 | 6.087 | 5.762 | 317.9 |
| 500 | 8,192,000 | 4.555 | 5.101 | 164.3 |

Throughput ~3,600 tokens/s (16,384 tokens per optimiser step, ~14 s/step).

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

The two vocabularies are byte-identical on this corpus: 9.4 Mcharacters is not enough text
to distinguish a 32k vocabulary from a 64k one. That is reported rather than hidden.

### Compute

| | |
|---|---|
| Measured matmul peak (fp16, synced) | 2.80 TFLOP/s |
| Measured training throughput | ~3,600 tokens/s |
| CPU matmul peak (fp32) | 0.21 TFLOP/s |
| GPU speedup over CPU | **7–35×** |

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