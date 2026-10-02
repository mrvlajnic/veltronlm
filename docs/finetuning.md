# Supervised Fine-Tuning

Entry point: `python -m veltron.finetuning.cli`

---

## 1. The dataset

604 examples, built entirely from repository artefacts. **No external teacher model is
used.** Reproduce with `python -m veltron.finetuning.cli build-data`.

| Category | Count | Source | Grounded |
|---|---:|---|---|
| `grounded_support_qa` | 84 | knowledge-base chunks | yes |
| `general_summary` | 135 | corpus documents | yes (extractive) |
| `general_explanation` | 129 | corpus documents | yes (extractive) |
| `general_extraction` | 128 | corpus documents | yes (extractive) |
| `general_classification` | 107 | corpus documents | yes (rule label) |
| `refusal` | 14 | constructed | **no** |
| `escalation` | 7 | constructed | **no** |

583 grounded, 21 refusals. 594 English / 10 Serbian.

### 1.1 Three families, and the one usually forgotten

1. **Grounded support QA.** The answer is the knowledge-base chunk's own text, lightly
   cleaned. Questions are template variations over the section heading. This **cannot**
   hallucinate: the target is copied from the document that retrieval will later surface.
2. **Refusal and escalation.** Questions with no documentation answer, paired with the
   correct refusal. **Without these, SFT teaches that every question has an answer** — the
   single most damaging failure mode for a support assistant.
   `test_sft_dataset_contains_refusals_and_grounded` fails if they disappear.

   Half the refusals carry no context at all and half carry `Context: (no documentation
   found)`, so *absence of context* itself is trained to produce a refusal.
3. **General instructions.** Summarisation, classification, extraction, explanation, with
   **deterministic extractive references**.

### 1.2 Why the references are extractive

A generated target would need a stronger model than the one being trained, and an
unverifiable target is worse than a dull but correct one. So:

| Task | Reference |
|---|---|
| summarisation | first 3 real sentences |
| classification | `classify_ticket()` output |
| extraction | `extract_entities()` serialised as JSON |
| explanation | first 6 substantive lines, numbered |

### 1.3 A data-quality failure, fixed

The first version's summaries were raw Project Gutenberg front matter:

```
[Illustration]
HEWET'S HOUSEHOLD STORIES FOR LITTLE FOLKS
ILLUSTRATED BY THE BEST ARTISTS. CINDERELLA
Credit for this e-text: Internet Archive; University of Florida, ...
```

Training on that teaches the model that a summary is a list of scanning artefacts — worse
than having no summary task. Fixed by `clean_for_sft()` (removes illustration markers,
e-text credits, Gutenberg headers), `is_boilerplate_heavy()` (drops documents whose opening
is dominated by them) and `_sentences()` (requires length and a terminator, which excluded
the fragmented front matter). See [EXP-0010](../experiments/EXP-0010.md).

## 2. Completion-masked loss

**Loss is computed only on assistant tokens.** This is the single most consequential detail
in instruction tuning.

```python
for text, is_target in parts:
    ids.extend(tokenizer.encode(text))
    mask.extend([1 if is_target else 0] * len(ids))
```

Including prompt tokens spends most of the gradient teaching the model to reproduce its own
input — the most common reason a fine-tune looks healthy while failing to follow
instructions.

### 2.1 The assistant's opening marker is unmasked

The `<|assistant|>` marker is part of the target region, because the model must learn to
*start* its own turn, not merely continue one.

### 2.2 Truncation keeps the answer

Sequences longer than `max_seq_len` are truncated **from the left**, so the assistant's
response always survives. Cutting the tail would train on a half-finished answer.

`test_sft_masks_non_target_tokens` asserts the first target position is at or after the
prompt boundary.

### 2.3 The loss really is masked

`test_sft_loss_ignores_masked_positions` perturbs the logits under masked positions by +5.0
and asserts the loss is unchanged to 1e-5. A leakage bug would fail this immediately.

```python
def masked_loss(logits, labels, chunk_tokens=0):
    flat_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    flat_labels = labels[:, 1:].reshape(-1)
    keep = flat_labels != -100
    ...
```

Positions with label `-100` are ignored entirely, so the loss is the mean over **target**
tokens, not over the padded batch. Chunking bounds the fp32 log-softmax memory as in
pretraining.

## 3. Length bucketing

Batches are formed from length-sorted windows, then shuffled. At this dataset size padding
could otherwise double the cost of every step. Combined with length bucketing the loader is
`O(n log n)` in Python but the batches are near-uniform in length.

## 4. Hyperparameters

| Setting | Value | Why |
|---|---|---|
| Base LR | 1e-5 | 60x lower than pretraining. Fine-tuning a pretrained model at the pretraining LR destroys it. |
| Schedule | cosine, 5% warmup | |
| Epochs | 3 | 604 examples is small; more epochs memorise the knowledge base. |
| Weight decay | 0.0 | The model is already regularised by pretraining. |
| Grad clip | 1.0 | |
| Max sequence | 640 | Fits a grounded QA example with context. |
| Micro-batch / accum | 2 / 4 | 8 sequences per step. |
| Eval fraction | 8% | 48 held-out examples. |

## 5. Measured results

`python -m veltron.finetuning.cli train --base checkpoints/micro-pretrain`

Results are recorded in `checkpoints/sft-support/summary.json` and reproduced in
`FINAL_REPORT.md`. The summary reports final validation loss and perplexity **broken down by
category**, which is the useful part: a mean over 604 examples would hide a refusal-category
regression behind summarisation gains.

## 6. Limitations

1. **604 examples is very small.** Enough to change format-following behaviour, not enough
   to teach new capability.
2. **The support KB appears ~116 times** in pretraining and again in SFT. The fine-tuned
   model memorises it. Retrieval carries domain behaviour; the weights do not.
3. **594 en / 10 sr.** The fine-tuned model's Serbian ability is essentially unchanged from
   the base model, and there is no evidence either way.
4. **Extractive references make summarisation trivially easy.** The model may learn to copy
   the opening rather than to summarise.
5. **No held-out support evaluation inside the SFT loop.** Per-category SFT loss is a weak
   proxy; the real test is the evaluation suite.
6. **No multi-turn conversations.** Every example is system/user/assistant. Conversation
   handling is unmeasured.
7. **Learning rate was not swept.** 1e-5 is the standard default, not a measured optimum.