# Roadmap

Phase status with the blocking condition for each incomplete phase. Last updated
2026-09-30.

Legend: **VERIFIED** (implemented, tested, measured) · **IMPLEMENTED** (code complete, not
fully measured) · **IN PROGRESS** · **BLOCKED** (external resource required) ·
**NOT STARTED**

---

## Phase summary

| Phase | Status | Notes |
|---|---|---|
| 0 Environment | **VERIFIED** | Audit and feasibility math published |
| 1 Repository | **VERIFIED** | Structure, CLI, configs, CI, 184 tests |
| 2 Tokenizer | **VERIFIED** | Two vocabularies trained, 0 round-trip failures |
| 3 Data | **VERIFIED** | 19,485,297 train tokens, full provenance |
| 4 Architecture | **VERIFIED** | Causality exact; 4B params verified two ways |
| 5 Tiny pretraining | **VERIFIED** | `micro` run, val loss 5.76 → 4.56 |
| 6 Scaling | **BLOCKED** | 60 GiB optimizer state vs ~10 GiB usable VRAM |
| 7 Base model | **BLOCKED** | Depends on phase 6 |
| 8 Instruction tuning | **IMPLEMENTED** | Pipeline tested; run pending base model |
| 9 Customer support | **VERIFIED** | 14/14 hit@1 retrieval; triage tested |
| 10 RAG | **VERIFIED** | 100% hit@1, citations validated |
| 11 Alignment | **IMPLEMENTED** | DPO objective verified; no run yet |
| 12 Evaluation | **IMPLEMENTED** | 37 items, metrics, matrix |
| 13 Optimization | **NOT STARTED** | Needs phase 6 |
| 14 Release | **NOT STARTED** | Needs phases 6–8 |

---

## 0. Environment — VERIFIED

`docs/environment.md` publishes the host audit, synchronised throughput measurements and the
feasibility arithmetic.

## 1. Repository — VERIFIED

Structure, unified CLI, `pyproject.toml`, configs, CI with six jobs, 184 tests.

## 2. Tokenizer — VERIFIED

Both vocabularies trained from scratch. Measured compression per language and format; 0
round-trip failures. Three real bugs found and fixed
([EXP-0006](../experiments/EXP-0006.md)).

**Open:** retrain on a 40 Mcharacter sample to test whether 32k vs 64k diverges. Currently
measurably better on every slice (+23.6% on Serbian Cyrillic).

## 3. Data — VERIFIED

`dataset-v1`: 1,312 documents, 383,615,628 characters, 19,485,297 train tokens. Full licence
register, filter funnel, provenance.

## 4. Architecture — VERIFIED

4,026,765,312 parameters, analytic == `meta`-device for all 7 tiers. Causality exact to 0.0.
KV cache within 9.2e-07. Batched generation token-identical to single-prompt.

## 5. Tiny pretraining — VERIFIED

`veltronlm-micro`, 55,715,328 parameters:

| Step | Tokens | Val loss | Val perplexity |
|---:|---:|---:|---:|
| 0 | 0 | — | — (init 9.55 ≈ ln V) |
| 250 | 4,096,000 | 5.762 | 317.9 |
| 500 | 8,192,000 | 5.101 | 164.3 |
| 750 | 12,288,000 | **4.564** | **96.0** |

## 6. Scaling — **BLOCKED**

**Blocker:** AdamW with fp32 master weights requires **60.00 GiB** (params + grads + m + v)
against **~10 GiB** usable VRAM. A 6.0x shortfall before activations or fragmentation.

Secondary: at the measured 2.80 TFLOP/s matmul peak, Chinchilla-optimal 4B training is a
**~22-year** single-GPU run; at the measured end-to-end efficiency, ~710 years.

**What would unblock it**, in increasing order of cost:

| Option | Requirement |
|---|---|
| 8-16x A100/H100 nodes | ≥ 640 GB aggregate VRAM; standard FSDP/DeepSpeed |
| 1x A100 80GB + CPU offload | Optimizer states in host RAM; 2-4x slower |
| LoRA/QLoRA fine-tuning of an existing 4B | An open-weight base to start from; **not** pretraining |
| Consumer GPU + gradient accumulation | **Does not help** — this is optimizer *state*, not activations |
| Longer runtime on this GPU | **Does not help** — the memory shortfall is absolute |

## 7. Base model — **BLOCKED**

Depends entirely on phase 6. `veltronlm-4b-base` exists as a verified architecture with no
weights.

## 8. Instruction tuning — IMPLEMENTED

Pipeline tested end to end; 604-example dataset built with refusals; completion-masked loss
verified by a perturbation test. **Run pending** a usable base checkpoint.

## 9. Customer support — VERIFIED

| Metric | Result |
|---|---|
| Retrieval hit@1 | **14/14 = 100%** |
| Retrieval recall@3 | **14/14 = 100%** |
| Classification | 14 categories, 20+ tests |
| Citation validation | fabricated references detected |
| Absolute confidence, nonsense query | 0.21–0.24 (below the 0.18 threshold band) |

## 10. RAG — VERIFIED

Hybrid retrieval, absolute confidence, bilingual expansion, heading-aware chunking with
breadcrumbs, multi-format ingestion.

## 11. Alignment — IMPLEMENTED

DPO objective verified by three tests, including that a tie is not counted as a win and that
rejectors do not fire on correct answers. Preference pairs buildable from six named failure
modes. **No alignment run completed**, so no improvement is claimed.

## 12. Evaluation — IMPLEMENTED

37 items across three suites; metrics with no learned judge; benchmark matrix including
`not_evaluated` rows.

## 13. Optimization — NOT STARTED

Needs phase 6.

## 14. Release — NOT STARTED

Needs phases 6–8.

---

## Priority queue

Ordered by value per unit of effort. Items 1–5 are doable on this host.

### 1. Get more Serbian data — **highest value**

Serbian is ~1% of the corpus. This is the single largest quality ceiling.

* Category crawl with exponential backoff and a longer budget
* Serbian Wikisource (declared, not yet crawled)
* A permitted parallel-corpus download

Success metric: Serbian ≥ 5% of the mixture; the Serbian compression slice covers ≥ 20
documents instead of 2.

### 2. Complete the SFT → DPO → evaluation chain

Use the finished `micro` checkpoint as the base. Produces the first real end-to-end
post-training result and populates `reports/eval-micro.json`.

Success metric: evaluation report with measured support, adversarial, perplexity and
long-context numbers; DPO accuracy delta reported honestly whatever its sign.

### 3. Run the three designed sweeps

| Script | Question |
|---|---|
| `scripts/exp_lr.py` | Is 6e-4 right for 55.7M on this corpus? |
| `scripts/exp_mixture.py` | Does the 10% literature / 22% code split help or hurt? |
| `scripts/exp_tokenizer.py` | Does 32k vs 64k matter on more text? |

The scripts exist; no sweep has run. No sweep result is claimed.

### 4. Red-team the RAG path

Independent adversarial review of prompt injection, retrieval poisoning and citation
fabrication. The current blocklist is a known-weakness (`docs/security.md` §5.3).

### 5. Re-benchmark on an idle GPU

All throughput numbers must be measured with nothing else running
([EXP-0013](../experiments/EXP-0013.md)). Also measure the 4B tier in int8, which fits in
3.75 GiB and does not contend with a training job.

### 6. Add toxicity filtering — **required before public release**

No toxicity classifier is applied to the pretraining corpus. `HARASSMENT_MARKERS` exists in
the *inference* path only.

### 7. Fused attention

`scores = q kᵀ + mask` materialises `B × H × T × T`. At 8k context and 24 heads that is
3.1 GiB in fp16 for the scores alone. FlashAttention or a chunked implementation would cut
attention memory from quadratic to linear in a chunked form. Not available on DirectML.

### 8. Authenticated serving

The API has no authentication. Required for anything beyond loopback.

### 9. Distributed training (phase 6 unblock)

FSDP2 or DeepSpeed with ≥ 8 accelerator nodes. Out of scope for this host.

### 10. Baseline comparison

Run a Qwen-class model of comparable size on the same suites with the same prompts. Only
then is any comparative statement defensible. **Currently no baseline has been run, so none
is claimed.**

---

## Explicitly not planned

* **Kubernetes.** One GPU and one process do not need an orchestrator. Adding it would be
  theatre.
* **A public model release before phase 6.** Releasing a 55.7M model's weights as "VeltronLM"
  would misrepresent the project.
* **Teaching more than one new language.** Serbian is the requirement; adding more would
  dilute it.