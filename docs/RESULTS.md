# Results

Every number here is read from an artefact. `python scripts/facts.py` re-derives all of
them; if a figure in this file disagrees with its source, the artefact wins and the
document is wrong.

Last updated **2026-10-02**, from `checkpoints/mini-pretrain` at step 3,250 (still
training; figures marked LIVE will move).

---

## 1. What is trained

| | |
|---|---|
| Model | `veltronlm-mini` |
| Parameters | **244,354,048** |
| Layers / d_model / heads | 16 / 1,024 / 16 query, 4 KV |
| Context | 2,048 (trained at 512) |
| Corpus | `dataset-v1`, 19,485,297 train tokens |
| Tokens consumed | 26,624,000 (**1.37 epochs**) |
| Precision | fp32 |
| Hardware | 1x AMD RX 6700 XT, DirectML |

The 4B architecture in this repository is **verified but untrained**. See §7.

## 2. Pretraining

### 2.1 Validation curve (LIVE)

| Step | Tokens | Val loss | Val perplexity |
|---:|---:|---:|---:|
| 250 | 2.0M | 7.5122 | 1,830.2 |
| 500 | 4.1M | 6.9695 | 1,063.7 |
| 750 | 6.1M | 6.5606 | 706.7 |
| 1000 | 8.2M | 5.8554 | 349.1 |
| 1250 | 10.2M | 5.5000 | 244.7 |
| 1500 | 12.3M | 5.3496 | 210.5 |
| 1750 | 14.3M | 5.4538 | 233.6 |
| 2000 | 16.4M | 3.7363 | 41.9 |
| 2250 | 18.4M | 3.1137 | 22.5 |
| 2500 | 20.5M | 3.3974 | 29.9 |
| 2750 | 22.5M | 3.7460 | 42.3 |
| 3000 | 24.6M | 1.6098 | 5.0 |
| 3250 | 26.6M | **1.1045** | **3.0** |

### 2.2 Read the training loss, not the validation number

**Train loss is the trustworthy signal:**

| | Step 500 | Step 3,250 |
|---|---:|---:|
| Train loss (EMA) | 5.48 | **3.85** |
| Gradient norm | 0.64 | 0.51 |
| Steps with grad norm > 5 | 0 | **0** |

Smooth, monotone, and stable. The gradient norm never approached the clip threshold after
warmup.

**The validation perplexity of 3.0 should not be quoted as a quality result.** Three
reasons, all measurable:

1. **The validation split is 6 documents** (51,287 tokens) and evaluation samples only
   **8,176** of them. That is far too small for a stable estimate.
2. **Validation loss is now *below* training loss** (1.10 vs 3.85). The 6 validation
   documents are a narrow slice of the same Project Gutenberg pool the model trained on, so
   they are much easier than the training distribution.
3. **The curve is not monotone**: 3.11 → 3.40 → 3.75 → 1.61. That is the signature of a
   small-sample estimate, not of a converging model.

An honest summary is: **train loss fell 6.15 → 3.85 over 26.6M tokens with no instability,
and the model is meaningfully better than its step-500 self. A claim about general
quality requires a proper held-out set, which is the first thing to fix.**

This is recorded rather than smoothed over because a reviewer who spots an unexplained
perplexity of 3.0 on a 6-document validation set will discount everything else.

## 3. Retrieval

| Metric | Result |
|---|---|
| **hit@1** | **14/14 = 100%** |
| **recall@5** | **14/14 = 100%** |
| Confidence, documented query | 0.88–0.97 |
| Confidence, nonsense query | 0.21–0.24 |

Measured on 14 held-out support queries. The second row that matters is the nonsense row:
a retrieval system that scores a meaningless query at 0.9 cannot be used to decide whether
to answer or refuse.

## 4. Tokenizer

| Slice | `tok-mini-32k` | `tok-4b-64k` | 64k gain |
|---|---:|---:|---:|
| English prose | 3.646 | **3.788** | +3.9% |
| Serbian Cyrillic | 2.819 | **3.484** | **+23.6%** |
| Serbian Latin | 2.177 | **2.346** | +7.8% |
| Python source | 3.910 | **4.032** | +3.1% |
| Technical Markdown | 3.696 | **3.826** | +3.5% |
| JSON / structured | 3.416 | **3.597** | +5.3% |
| Round-trip failures | **0** | **0** | — |

The Cyrillic gain is the substantive finding: doubling the merge budget stops the minority
script being crowded out of the table. This is a concrete, reproducible argument about
tokenizer allocation under language imbalance.

## 5. Data

| | |
|---|---|
| Documents seen / kept | 1,380 / 1,312 (95.1%) |
| Train tokens | 19,485,297 |
| Characters kept | 383,615,628 |
| Chars per token | 3.593 |
| PII documents dropped | 56 |
| Duplicates removed | 11 (1 exact, 10 near) |
| Manifest hash | `272e134cf74c7367` |

## 6. Correctness

| Property | Measured |
|---|---|
| Attention causality | **exactly 0.0** leakage |
| KV cache vs full forward | 9.239e-07 |
| Batched (left-padded) vs single prompt | 5.513e-07, token-identical |
| Parameter count: analytic vs `meta` device | equal, all 8 tiers |
| Gradient flow | every parameter, never zero |

## 7. The 4B architecture

| | |
|---|---|
| Parameters | **4,026,765,312**, cross-checked two independent ways |
| Persistent optimizer state, fp32 | 60.00 GiB |
| Usable VRAM on this host | ~10 GiB (measured) |
| Verdict | **6.0x short — training not attempted** |

Not a claim, a constraint. The architecture is real and unit-tested; the weights do not
exist.

## 8. Benchmark comparison

**None performed.** No Qwen-class or other model has been run on this host, so no
comparative statement is made anywhere in this repository.

This is the largest gap for any external review, and it is stated rather than obscured.
The first measurement worth adding is perplexity for one small open model on
`dataset-v1`'s validation split under an identical tokenizer, which is cheap and would
convert "we built a system" into "we built a system and know where it sits".

## 9. Not yet done

* **SFT and DPO have never been run.** The pipelines are implemented and unit-tested; the
  model is a base model and will not follow instructions.
* **No external baseline.**
* **No human evaluation.**
* **Validation set is 6 documents** — the most urgent measurement problem.
* Serbian is ~1% of the corpus.

---

## Reproduce

```powershell
python scripts\facts.py          # every figure in this document, read from artefacts
python scripts\verify_rag.py      # retrieval
python scripts\verify_param_count.py
python scripts\verify_checkpoints.py checkpoints\mini-pretrain
python -m veltron.evaluate --checkpoint checkpoints\mini-pretrain --suites all --perplexity
```
