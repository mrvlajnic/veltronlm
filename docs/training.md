# Training

Pretraining objective: **next-token prediction** with shift-free causal cross-entropy.

Entry point: `python -m veltron.train`

---

## 1. The live run

```
python -m veltron.train --model micro --tokenizer models/tok-mini-32k \
  --dataset datasets/dataset-v1 --run-name micro-pretrain \
  --seq-len 1024 --batch-size 2 --grad-accum 8 \
  --max-steps 4000 --max-hours 5.0 \
  --lr 6e-4 --warmup 120 --schedule cosine \
  --eval-interval 250 --save-interval 250 --log-interval 25 --resume none --seed 1234
```

Configuration as executed:

| Setting | Value | Why |
|---|---|---|
| Model | `veltronlm-micro` | 55,715,328 params — largest tier that trains in hours here |
| Sequence length | 1024 | Matches the packed dataset |
| Micro-batch | 2 | Micro-batch 4 OOMs: the logit tensor at 32k vocab is ~0.54 GiB per micro-batch before autograd buffers |
| Grad accumulation | 8 | Effective batch 16 sequences = 16,384 tokens/step |
| Tokens per step | 16,384 | |
| Max steps | 4,000 | Wall-clock budget of 5.0 h cuts it earlier, at ~1,150 steps |
| Peak LR | 6e-4 | Standard for a 55M model with cosine decay |
| Warmup | 120 steps | |
| Schedule | cosine to 10% of peak | |
| Weight decay | 0.1 | Excluded from norms and embeddings |
| Grad clip | 1.0 | Pre-clip norm is logged; a rising trend is the earliest divergence signal |
| Precision | fp32 | DirectML has no autocast for `aten::embedding` (see below) |
| Evaluator | — | Canary runs first: 200-step memorisation of 32 windows |
| Seed | 1234 | |

### 1.1 Measured results

See `checkpoints/micro-pretrain/train_log.jsonl` for the full record and
`experiments/EXP-*.md` for the analysis. Headline numbers are reproduced in
`FINAL_REPORT.md`.

## 2. Optimiser and parameter groups

AdamW, `betas=(0.9, 0.95)`, `eps=1e-8`, weight decay 0.1.

Weight decay is **excluded from RMSNorm gains, biases and embeddings**:

```python
if p.ndim < 2:                     no_decay.append(p)      # norms, biases
elif "embed" in name or name.startswith("lm_head"):
                                      emb_decay.append(p)
else:                                 decay.append(p)
```

Decaying the norm gains is harmful: they are direction-sensitive vectors whose magnitude
carries meaning, so shrinking them toward zero degrades the model without acting as a
useful regulariser. `test_optimizer_excludes_norms_from_weight_decay` asserts the split.

## 3. Learning-rate schedule

`WarmupCosineSchedule` is a function of **total optimiser steps**, not wall-clock or tokens
consumed. A run that is preempted and resumed therefore lands on exactly the rate it would
have used without the interruption — verified by `test_resume_restores_step_and_optimizer`.

Linear warmup over `warmup_steps`, then cosine decay to `min_lr_ratio × peak`.
`test_lr_schedule_is_monotone_after_warmup` asserts monotonicity after warmup and that the
final rate is below 20% of the post-warmup peak.

## 4. Gradient accumulation

Micro-batches accumulate into one optimiser step; the loss of each micro-batch is divided by
`grad_accum_steps` **before** `backward()`, so gradients sum to the mean over the full
effective batch.

`test_gradient_accumulation_matches_larger_batch` asserts that 4× (batch 2) is
gradient-equivalent to 1× (batch 8) to `atol=1e-4`.

## 5. Mixed precision

`_resolve_precision()` returns `(name, autocast_dtype, use_grad_scaler)`.

| Backend | `precision=auto` | Why |
|---|---|---|
| CUDA | bf16 if supported, else fp16, with `GradScaler` | Standard autocast path |
| DirectML | **fp32** | See below |
| CPU | fp32 | |

**DirectML constraint.** `torch.autocast` on this backend has no `aten::embedding` kernel,
so entering an autocast region raises `NotImplementedError` on the very first op of the
model. `precision=fp16` on DirectML is therefore expressed by casting the parameters
(`model.half()`) and letting each matmul accumulate in fp32 — never by entering an autocast
region. This is recorded in the run config as `precision: fp16_weights` so a log is never
ambiguous about what actually ran.

## 6. Memory management

### 6.1 Chunked cross-entropy

`compute_loss(..., chunk_tokens=N)` accumulates the loss over slices of the flattened token
dimension, bounding the fp32 log-softmax buffer at `N × V` instead of `batch·seq × V`.

This is what makes the 4B configuration trainable at all in principle: at 8k context and a
65k vocabulary, the undivided logit tensor alone is ~2 GiB per micro-batch, and the fp32
softmax plus its backward doubles or triples that. `test_chunked_loss_matches_unchunked`
asserts the chunked and unchunked losses agree to `1e-4`.

### 6.2 Batch sizing is memory-bound, not compute-bound

Observed OOM threshold on this host (`micro`, 32k vocab, seq 1024, fp32):

| Micro-batch | Logit tensor | Result |
|---:|---:|---|
| 8 | ~1.07 GiB | OOM (`Could not allocate tensor with 3218341696 bytes`) |
| 4 | ~0.54 GiB | OOM (`1609301920 bytes`) |
| 2 | ~0.27 GiB | works, ~3,600 tok/s |
| 1 | ~0.13 GiB | works, ~2,400 tok/s |

Batch 2 with accumulation 8 gives the same effective batch at half the peak memory.

## 7. The canary

Before any training step, `Trainer._canary()` builds a **fresh** model and tries to
memorise 32 fixed windows of ≤128 tokens over 200 steps at lr 2e-3.

```
CANARY PASSED: loss 8.2158 -> 1.8608 (4.4x reduction)
```

If the loss cannot drop 4× in that budget, the run **refuses to start**. The canary exists
because a model with shifted labels, detached activations or a misread shard trains without
error and produces fluent nonsense; failing fast costs 90 seconds and saves hours.

### 7.1 Canary design notes

Two earlier designs produced **false failures** on the 55M model, both instructive:

* *256 windows × 60 steps at batch 1 → 1.22× reduction.* It sampled widely across the
  corpus, so it was measuring generalisation rather than wiring. Reduced to 32 fixed
  windows.
* *32 windows × 120 steps at 1024-token windows.* A 55M model barely moves in 120 steps at
  batch 1 over long sequences. Truncated to 128-token windows, which lets the batch grow.

The current threshold is `best_loss < 0.25 × first_loss` with all 200 losses finite.

## 8. Validation

Every `eval_interval` steps (250) the trainer evaluates on `eval_batches` validation
batches, accumulating **per-token** rather than per-batch:

```python
n = int((labels[:, 1:] != -100).sum())
total_loss += float(out.loss) * n
total_tokens += n
```

Averaging per batch would weight batches of unequal token count incorrectly. Perplexity is
reported as `exp(min(loss, 20.0))` — computed in log space because `exp()` of a large loss
overflows to `inf` and would hide the real value.

## 9. Logging

`checkpoints/<run>/train_log.jsonl`, one JSON object per line:

```json
{"step": 250, "loss": 6.087371855974197, "loss_ema": 6.0, "lr": 0.000598,
 "grad_norm": 0.57, "tokens_seen": 4096000, "tokens_per_step": 16384,
 "tokens_per_second": 3629.5, "seconds_per_step": 14.09,
 "precision": "fp32", "epoch": 0}
{"step": 250, "eval": true, "val_loss": 5.761687517166138,
 "val_perplexity": 317.88431178349396, "val_tokens": 40920}
```

`grad_norm` is the **pre-clip** norm, logged deliberately: a rising pre-clip trend is the
earliest reliable divergence signal, and the post-clip value is always exactly 1.0.

## 10. Failure classification

Every failure increments a counter in the checkpoint manager, persisted in `summary.json`:

| Counter | Trigger | Response |
|---|---|---|
| `oom` | `"out of memory"` in a `RuntimeError` | Emergency checkpoint, then re-raise |
| `nan_loss` | Non-finite gradient detected at clip time | Skip the optimiser step, count it |
| `gradient_explosion` | Skipped step with pre-clip norm > 50× the clip value | Counted for diagnosis |
| `corrupt_checkpoint` | Unreadable optimizer/RNG state on load | Warn and continue with what is loadable |
| `io_error` | Checkpoint write failed | Clean up staging, re-raise |

Non-OOM `RuntimeError`s propagate immediately. They are real bugs and must not be
swallowed by a broad `except`.

The run aborts if `consecutive_nonfinite > nan_patience` (default 3).

## 11. Wall-clock budget

`--max-hours` sets a hard budget. The loop checks it every step and stops with
`stop_reason: "wall_clock_budget"` after writing a final checkpoint. This is what makes a
long run safe to launch unattended on a workstation.

## 12. Hyperparameter sweeps

`scripts/exp_*.py` provide controlled single-variable experiments. The method is always:
one variable at a time, same seed, same data, compare validation loss.

| Script | Varies |
|---|---|
| `scripts/exp_lr.py` | learning rate |
| `scripts/exp_mixture.py` | dataset mixture weights |
| `scripts/exp_tokenizer.py` | tokenizer vocabulary size |

**No sweep result is claimed in this repository.** The scripts exist; the runs have not been
performed to completion, and `experiments/` records only what actually ran.

## 13. Measured throughput on this host

`veltronlm-micro`, seq 1024, batch 2, accumulation 8, fp32 on RX 6700 XT via DirectML:

| Metric | Value |
|---|---|
| Seconds per optimiser step | ~14.0 |
| Tokens per optimiser step | 16,384 |
| **Tokens per second** | **~3,600** |

Throughput drops when other GPU work runs concurrently. The numbers in this document were
taken with the machine otherwise idle.