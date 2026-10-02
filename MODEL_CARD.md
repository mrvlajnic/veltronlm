# Model Card — VeltronLM

## Model details

**Name:** VeltronLM
**Architecture:** decoder-only Transformer (pre-norm, GQA, RoPE, RMSNorm, SwiGLU)
**Parameters:** **4,026,765,312** (4.027 B) for `veltronlm-4b-base`
**Trained variant in this repository:** `veltronlm-micro`, **55,715,328** parameters

> ### The 4B weights do not exist
>
> `veltronlm-4b-base` is a **verified architecture with no weights**. Its parameter count,
> tensor shapes, memory footprint and causality are verified and unit-tested. Training it from
> random initialisation requires **60.00 GiB** of AdamW optimizer state against the
> **~10 GiB** of usable VRAM (measured) on the build host — a **6.0x** shortfall — and a ~22-year
> single-GPU runtime at the measured matmul peak.
>
> The only trained weights in this repository are `veltronlm-micro` (55.7M parameters,
> 12.3M tokens, validation perplexity 96.0). See `experiments/EXP-0004.md`.

---

## Architecture

| Property | `veltronlm-4b-base` | `veltronlm-micro` (trained) |
|---|---|---|
| Layers | 36 | 8 |
| Hidden size | 3,072 | 512 |
| Query / KV heads | 24 / 8 | 8 / 2 |
| Head dimension | 128 | 64 |
| FFN (SwiGLU) | 8,192 | 1,376 |
| Vocabulary | 65,536 | 32,768 |
| Context | 8,192 | 1,024 |
| QK-norm | yes | yes |
| Tied embeddings | no | no |

## Intended use

**The customer-support assistant is the intended use.** VeltronLM exists to answer questions
about a product's documentation with citations, refuse when the documentation does not
contain the answer, and escalate to a human when appropriate.

Appropriate:

* Grounded question answering over a retrieval corpus
* Ticket triage and entity extraction
* Drafting responses that a human agent reviews
* Multilingual (English / Serbian) support in a controlled knowledge base
* Research on decoder-only architectures and DirectML training

## Out of scope

**Do not use VeltronLM for:**

* Any decision that affects a person's money, health, safety or legal status
* Medical, legal or financial advice
* Generating content for production without human review
* Open-ended creative writing at any scale
* Any task requiring factual recall beyond its retrieval corpus — **it will confabulate**
* Deployment without authentication on an untrusted network
* Any use not covered by the training data licences in `DATA_SOURCES.md`

## Training data

`dataset-v1`, manifest hash `272e134cf74c7367`:

| | |
|---|---|
| Documents seen / kept | 1,380 / 1,312 (95.1%) |
| Characters kept | 383,615,628 |
| **Train tokens** | **19,485,297** |
| Validation tokens | 51,287 |
| Chars per token | 3.593 |
| Mixture | 34% general, 22% code, 12% technical, 12% Serbian, 10% literature, 10% support |

Sources: Wikipedia (CC BY-SA 4.0), Project Gutenberg (public domain), CPython (PSF-2.0),
Pallets/requests/urllib3/NumPy (BSD-3-Clause / Apache-2.0 / MIT), and a project-authored
synthetic knowledge base (CC0-1.0). GPL-2.0 material is excluded **in code**.

### Data filtering

Applied in order, with the measured funnel:

| Stage | Rejected |
|---|---:|
| PII (12 patterns; credentials and bulk contacts dropped) | 56 |
| Exact deduplication (SHA-256) | 1 |
| Near deduplication (MinHash 64-perm, LSH 16×4) | 10 |
| Too short | 1 |
| Malformed JSON line | 1 |
| **Accept rate** | **95.1%** |

Language identification is script-ratio based with a stopword discriminator. **Not applied:**
a toxicity classifier. That is a blocker for public release.

## Training procedure

### Pretraining

Causal next-token prediction, shift-free cross-entropy, chunked for bounded memory.

```
veltronlm-micro · seq 1024 · batch 2 × accum 8 = 16,384 tokens/step
AdamW · lr 6e-4 cosine · 120 warmup steps · weight decay 0.1 (excluded from norms)
grad clip 1.0 · fp32 · seed 1234
```

### Post-training

604-example SFT set with **completion-masked loss** — loss only on assistant tokens.
21 refusal/escalation examples, deliberately included so the model does not learn that every
question has an answer.

DPO against six named failure modes, with a frozen reference model and held-out
preference-accuracy tracking.

## Measured results

### Pretraining — `veltronlm-micro`

| Step | Tokens seen | Train loss | Val loss | Val perplexity |
|---:|---:|---:|---:|---:|
| 0 | 0 | 9.55 (`ln V`) | — | — |
| 250 | 4,096,000 | 6.087 | 5.762 | 317.9 |
| 500 | 8,192,000 | 4.555 | 5.101 | 164.3 |
| 750 | 12,288,000 | 5.414 | **4.564** | **96.0** |

Training was still in progress at the 5-hour wall-clock budget. These are intermediate
numbers, not a converged result.

### Retrieval (model-independent)

| Metric | Result |
|---|---|
| **hit@1** | **14/14 = 100%** |
| **recall@3** | **14/14 = 100%** |
| Confidence, documented query | 0.88–0.97 |
| Confidence, nonsense query | 0.21–0.24 |

### Tokenizer

| Slice | Chars/token | Round-trip failures |
|---|---:|---:|
| English prose | 3.788 | 0 |
| Serbian Cyrillic | 3.484 | 0 |
| Serbian Latin | 2.346 | 0 |
| Python source | 4.032 | 0 |
| JSON / structured | 3.597 | 0 |

### Hardware and compute

Windows 11 · Intel i7-7700 · AMD Radeon RX 6700 XT (12 GiB) · PyTorch 2.4.1 via DirectML.

| | |
|---|---|
| Measured matmul peak (fp16, synchronised) | 2.80 TFLOP/s |
| Measured training throughput | ~3,600 tokens/s |
| Wall clock, pretraining to step 750 | ~3.2 hours |
| Accelerator nodes | 1 |
| Accelerator type | consumer AMD GPU via DirectML |

## Comparison with other models

**None performed. No comparative claim is made anywhere in this repository.**

No Qwen-class or other baseline was run on this host, so any statement that VeltronLM is
better or worse than an existing model would be unsupported. `python -m veltron.evaluate
--matrix` prints the benchmark matrix with every unmeasured cell marked `not_evaluated` and
the reason.

The 55.7M model trained here is roughly 25x smaller than the smallest commonly compared
Qwen model and is not a meaningful comparison in either direction.

## Evaluation methodology

No learned judge. Every metric is computed by string/sequence operations so a score can be
reproduced and argued about.

37 fixed items across three suites: `support` (16), `adversarial` (12), `long_context` (9).

### Groundedness is an upper bound

Groundedness is **lexical containment** of required facts — necessary but not sufficient. An
answer can contain "24 months" and also invent a refund policy. The complementary negative
signal is `unsupported_claims`: a number-plus-unit absent from the retrieved context. Every
report carries the caveat inline.

## Limitations

### Model

1. **Heavily undertrained.** 55.7M parameters, 12.3M tokens consumed, against a
   Chinchilla-optimal ~1.1B. A factor of ~90 short. Validation perplexity 96.0 is far from
   useful for open-ended generation.
2. **Two-thirds of the parameters are embedding tables.** At a 32k vocabulary, 33.5M of
   55.7M are the embedding and output matrices, leaving ~22M of real transformer capacity.
3. **Will confabulate.** Like any decoder-only LM it produces fluent, plausible and wrong
   text when asked for facts it was not given. The RAG layer mitigates this; the raw model
   does not.
4. **No safety training beyond refusals.** 21 refusal examples is a demonstration, not a
   safety posture.

### Data

5. **Serbian is ~1% of the corpus** (21 documents). The highest-value gap in the project.
6. **The corpus is 98.7% Project Gutenberg by character.** Resampling handles the document
   mix, not the register.
7. **Up-sampling is extreme**: the 15-document support KB is repeated ~116 times.
8. **No toxicity filtering** on the pretraining corpus.
9. **The `test` split has 2 documents** — disjoint, but far too small to be useful.

### Retrieval

10. **Validated on 15 synthetic documents.** 100% hit@1 says nothing about a
    50,000-document knowledge base.
11. **Hashed embeddings, not semantic.** A question sharing no vocabulary with the document
    and containing no mapped Serbian term will fail.
12. **`MIN_RETRIEVAL_SCORE = 0.18` is fitted to this corpus** and has no theoretical
    justification.
13. **No contradiction detection** between retrieved documents.

### Serving

14. **No authentication.** Loopback only.
15. **No streaming for chat completions**, because retrieval must complete first.
16. **Rate limiting is per-process.**

## Ethics and intended deployment

* **Veltron Industries is fictional.** All bundled policies are synthetic, marked
  `synthetic: true` on every document, and `synthetic_data_notice: true` on every grounded
  API response. They must never be presented as a real company's terms.
* **Do not deploy for decisions affecting customers' money, access or safety** without
  human review and authentication.
* **Bias.** The corpus is 98.6% English and about 1% Serbian. The model's coverage of other
  languages is untested and likely poor. It will perform least well on the users it was
  explicitly designed for.
* **Privacy.** No private data is in the training corpus. Ticket data is held in memory only
  and is lost on restart.

## Licence

Code: **Apache-2.0** (`LICENSE`).

Weights, when released: **see `DATA_SOURCES.md` before release.** The CC-BY-SA 4.0
Wikipedia material imposes **share-alike on redistributed derivatives**, which most
interpretations extend to model weights. A release trained on a public-domain-only mixture
could be Apache-2.0 alone. **Obtain legal advice.**

## Reproducing

```powershell
python scripts/acquire_data.py --force --wikipedia-random 30
python scripts/fetch_gutenberg_bulk.py --limit 1200
python scripts/train_tokenizers.py
python scripts/build_dataset.py --seq-len 1024
python -m veltron.train --model micro --tokenizer models/tok-mini-32k \
    --run-name micro-pretrain --seq-len 1024 --batch-size 2 --grad-accum 8 \
    --max-steps 4000 --max-hours 5
python -m veltron.evaluate --checkpoint checkpoints/micro-pretrain \
    --suites support adversarial --perplexity --retrieval --long-context
```

Wikipedia's random-article generator is not deterministic; reuse the committed raw corpus
for a bit-identical dataset.

## Citation

```bibtex
@software{veltronlm,
  title  = {VeltronLM: a 4B-parameter decoder-only Transformer and customer-support system},
  note   = {Research release. 4B architecture verified; weights untrained.
            55.7M variant trained. Compute-limited single AMD GPU via DirectML.},
  year   = {2026},
  url    = {https://github.com/veltronlm/veltronlm}
}
```