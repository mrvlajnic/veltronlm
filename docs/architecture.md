# System Architecture

How the pieces fit together and why each boundary is where it is.

---

## 1. Layered design

```
┌──────────────────────────────────────────────────────────────────────┐
│  veltron/cli.py          unified command surface                      │
├──────────────────────────────────────────────────────────────────────┤
│  serving/app.py          HTTP API  ·  chatbot/  static UI             │
├──────────────────────────────────────────────────────────────────────┤
│  rag/pipeline.py         retrieval → grounding → citations            │
│  support/triage.py       classify → extract → escalate  (deterministic)│
├──────────────────────────────────────────────────────────────────────┤
│  inference/generator.py  sampling, KV cache, streaming                │
│  inference/engine.py     checkpoint → model+tokenizer+generator       │
├──────────────────────────────────────────────────────────────────────┤
│  alignment/              DPO                                          │
│  finetuning/             SFT, completion-masked loss                  │
├──────────────────────────────────────────────────────────────────────┤
│  training/               pretraining loop, schedules, checkpoints     │
├──────────────────────────────────────────────────────────────────────┤
│  data/                   acquire → filter → mix → split → pack → load  │
│  tokenizer/              byte-level BPE                               │
├──────────────────────────────────────────────────────────────────────┤
│  model/                  architecture, config, safetensors, quantize   │
├──────────────────────────────────────────────────────────────────────┤
│  utils/                  devices, logging, config, hashing, seeding   │
└──────────────────────────────────────────────────────────────────────┘
```

Dependencies point downward only. `model/` knows nothing about training; `training/` knows
nothing about RAG; the model layer imports no HTTP code.

## 2. The central design decision

**The deterministic rule layer runs before the model, and the model only phrases.**

```
customer message
  ↓  detect_safety_signals          ← prompt injection? credentials? → escalate NOW
  ↓  classify_ticket                ← 14 categories, weighted keywords
  ↓  extract_entities               ← product, serial, urgency, language
  ↓  RAG retrieval                  ← absolute confidence, not rank
  ↓  decide_escalation               ← one function, named reasons
  ↓  VeltronLM                       ← grounded generation, context is the only authority
  ↓  verify_citations                ← fabricated reference numbers detected
  ↓  numeric claim audit             ← invented policy/price/deadline detected
```

Why this matters: a language model asked to ignore its instructions may comply, so a
prompt-level defence is insufficient. Classification, entity extraction and escalation are
keyword rules that cannot be talked into a wrong answer, and they are inspectable by an
agent who needs to know *why* a ticket was routed where it went.

The cost is that the deterministic layer cannot do anything a model could: it cannot answer
"will this work in my house?". Retrieval covers that, and the model covers phrasing.

## 3. Device abstraction

Three backends in priority order: CUDA → DirectML → CPU. `veltron/utils/device.py` wraps
each so the rest of the codebase is written against one interface.

This was not gold-plating. It was forced by three DirectML gaps discovered in
[EXP-0001](../experiments/EXP-0001.md):

| Gap | Consequence for the design |
|---|---|
| No `aten::embedding` under autocast | `torch.autocast` only on CUDA; fp16 on DirectML = `model.half()` |
| Cannot materialise a computed 4-D bool tensor | Additive float attention masks instead of `masked_fill` |
| `aten::lerp` falls back to host | Optimiser is partly host-bound; measured throughput is below the matmul peak |

`DeviceInfo` carries a `torch_device_type` separate from the human-readable `kind`, because
torch reports DirectML as `privateuseone:0` and `torch.device("directml")` raises.

## 4. Checkpoint protocol

A checkpoint is a **directory**, not a file:

```
step-00000500/
  model.safetensors  optimizer.pt  scheduler.pt  rng.pt
  metadata.json      COMPLETE
```

`COMPLETE` is written **last**, inside a staging directory that is renamed into place. That
ordering is what lets `latest_valid()` distinguish a finished checkpoint from one torn by a
crash, and fall back to the previous step instead of dying.

See `docs/recovery.md`.

## 5. Reproducibility chain

Every artefact carries a hash, and the chain is complete from weights back to sources:

```
checkpoint metadata.json
  ├─ model_config         full architecture
  ├─ dataset_version      "dataset-v1"
  ├─ dataset_manifest_hash  "272e134cf74c7367"
  ├─ tokenizer_version    "tok-mini-32k"
  ├─ tokenizer_sha256     "27f1b702f6f567af"
  ├─ trainer_config       every TrainConfig field
  ├─ git_commit
  └─ checkpoint_hash      SHA-256 over the rest

datasets/dataset-v1/manifest.json
  ├─ totals, filter funnel, dedup stats
  ├─ mixture targets/achieved/upsampled
  ├─ per_source counts
  └─ provenance[] {source, license, license_url, retrieval_date, example url}
```

A checkpoint can be traced to the exact documents, tokenizer bytes and source revisions
that produced it.

### 5.1 What is not reproducible

* **Wikipedia's random-article generator** returns different articles each run. The category
  crawler and the Gutenberg ID selection are deterministic for a given `--limit`.
* **GPU floating-point** is not bit-identical across runs without
  `torch.use_deterministic_algorithms`, which DirectML does not fully honour.
* **Data acquisition** must be reused from the committed `datasets/raw/*.jsonl` for a
  bit-identical dataset.

## 6. Deliberate trade-offs

| Decision | Alternative | Why this way |
|---|---|---|
| Keyword classification | Trained intent classifier | Auditability: an agent can see why a ticket routed where it did. A classifier cannot explain itself. |
| Hashed n-gram retrieval | Neural bi-encoder | No model download, deterministic, and it cannot hallucinate a semantic relation. Costs paraphrase recall. |
| Extractive SFT targets | Teacher-model generations | A verifiable target beats an unverifiable one, and no external model API is used. |
| Additive float masks | `masked_fill` boolean | Required by DirectML; also one fewer kernel and identical numerics. |
| Rule-based escalation | Model self-assessment | A model asked whether it is confident is not a reliable confidence source. |
| Dtype-geometry shard check | Trust the index | A wrong dtype reinterprets every token and surfaces as an `IndexError` that reads like a model bug. |

## 7. Where the boundaries are wrong

Honest assessment of what should be restructured:

1. **RAG and triage are coupled through `RAGPipeline`.** Triage should be an independent
   service so a non-RAG surface (email, phone) can use it. The logic is separable; only the
   wiring is not.
2. **`support/knowledge_base.py` mixes content and code.** Content should live in
   `knowledge_base/*.md` with the Python file only providing loaders. A content edit should
   not be a code review.
3. **No abstraction over the model backend.** Only one model family exists, so the concrete
   class is used directly. Adding a second architecture would require extracting an
   interface.
4. **The evaluator is coupled to the RAG pipeline.** A model-only evaluation requires
   standing up retrieval, which conflates two things that should be measurable separately.
5. **No multi-tenant concept anywhere.** Tickets, feedback and the index are single-tenant.
   Correct for a research demo; a real deployment needs isolation at every level.