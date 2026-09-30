# Checkpointing and Recovery

Training must survive preemption, crashes, power loss and disk pressure. This document
describes exactly what is written, what is validated, and how to resume.

---

## 1. Layout

```
checkpoints/micro-pretrain/
├── run_config.json          full TrainConfig + model config + device + library versions
├── canary.json              overfit canary result
├── train_log.jsonl          one JSON object per log event
├── summary.json             final summary, including the failure budget
├── manifest.json            every checkpoint with step/tokens/losses/hash
└── step-00000250/
    ├── model.safetensors    weights
    ├── optimizer.pt         AdamW moments + param groups
    ├── scheduler.pt         LR schedule position + base LRs
    ├── rng.pt               torch, numpy and python RNG states
    ├── metadata.json        step, epoch, tokens, dataset/tokenizer hashes, git commit
    └── COMPLETE             written LAST
```

`step-00000250/` means **250 optimiser steps completed**, not 250 tokens or 250 batches.

## 2. Atomicity

The write order is:

1. Stage into `.staging-00000250/` (never directly into the target).
2. Write weights, optimizer, scheduler, RNG, metadata.
3. Write the `COMPLETE` marker.
4. Rename `.staging-00000250/` → `step-00000250/`.

Because `COMPLETE` is written before the rename, a crash at any point leaves either a
complete directory or a `.staging-*` directory that `is_valid()` rejects. A reader can
never observe a half-written checkpoint as valid.

`scripts/` sweeps `.staging-*` on every rotation, so abandoned staging directories do not
accumulate.

## 3. Validation

```python
def is_valid(self, path) -> bool:
    return (path.is_dir()
            and (path / COMPLETE_MARKER).exists()
            and any(path.glob("*.safetensors")))
```

`latest_valid()` walks newest → oldest and **skips invalid directories with a warning**,
returning the first valid one. A torn newest checkpoint therefore degrades to the previous
step rather than killing the run.

`verify_checkpoint(path)` is the independent auditor, used by tests and by hand:

```python
from veltron.training.checkpoint import verify_checkpoint
print(verify_checkpoint("checkpoints/micro-pretrain/step-00000250"))
```

It reports tensor count, byte size, presence of optimizer/scheduler/RNG, metadata fields, a
content hash, and a list of problems. A checkpoint is `valid` only when `errors` is empty.

## 4. Metadata

Every checkpoint records enough to explain and reproduce the run:

| Field | Purpose |
|---|---|
| `step`, `epoch`, `tokens_seen` | Position |
| `train_loss`, `val_loss`, `val_perplexity` | Metrics at this point |
| `tokens_per_second` | Throughput |
| `learning_rate`, `grad_norm` | Optimiser state at save time |
| `tokens_per_step`, `micro_batch_size`, `grad_accum_steps` | Effective batch reconstruction |
| `dataset_version`, `dataset_manifest_hash` | Corpus provenance |
| `tokenizer_version`, `tokenizer_sha256` | Vocabulary provenance |
| `model_config` | Full architecture |
| `trainer_config` | Every `TrainConfig` field |
| `git_commit` | Source revision (or `unavailable`) |
| `peak_memory_bytes`, `device` | Hardware record |
| `created_at` | Timestamp |
| `checkpoint_hash` | SHA-256 over the rest of the payload |

The `dataset_manifest_hash` is what lets you prove a checkpoint was trained on
`dataset-v1` specifically.

## 5. Resuming

```powershell
# automatic: newest valid checkpoint in the run directory
python -m veltron.train --model micro --run-name micro-pretrain ...          # default --resume auto

# explicit
python -m veltron.train --run-name micro-pretrain --resume step-00000250 ...

# from scratch (keeps existing checkpoints on disk)
python -m veltron.train --run-name micro-pretrain --resume none ...
```

What is restored:

* model weights (mandatory — a failure here propagates)
* optimizer state (Adam moments, param groups)
* scheduler position and base learning rates
* `torch`, `numpy` and `python` RNG states
* step, epoch, tokens seen, wall clock, best validation loss

After loading, the trainer forces the scheduler to the restored step:

```python
self.scheduler.step_count = self.state.step
self._set_lr(self.scheduler._lr_at(self.state.step))
```

Without this the LR curve would restart from step 0 while the weights are already at step
N, producing a discontinuity that a long resume would then have to train through.

### 5.1 What is tolerated

| Missing state | Behaviour |
|---|---|
| `optimizer.pt` | Warn, resume weights only. The first steps run with fresh Adam moments — a measurable but non-fatal degradation. |
| `scheduler.pt` | Warn, LR restarts. Prefer restoring the checkpoint. |
| `rng.pt` | Warn, continue. The run is no longer bit-reproducible. |
| `metadata.json` | Raise unless `strict=False`. |
| `model.safetensors` | Raise. Refuse to silently "resume" from a checkpoint with no weights. |
| corrupt optimizer state | Warn, continue with fresh state, increment `corrupt_checkpoint`. |

Missing auxiliary state degrades to the strongest available signal instead of aborting.
Corrupt *weights* always abort — there is nothing to fall back to.

## 6. Rotation

Defaults: `keep_last=4`, `keep_best=2`.

Rotation deletes checkpoints outside both windows, **except** the best-by-monitor checkpoint,
which is always retained even when it falls outside the keep-last window — the best weights
are usually what a later run needs to continue from. `manifest.json` is rewritten after each
rotation.

## 7. Emergency checkpoint

On OOM the trainer writes a checkpoint **without rotating**, so the previous good state
cannot be destroyed by a crash:

```python
def _emergency_checkpoint(self) -> None:
    self.ckpt.save(self.model, self.optimizer, self.scheduler,
                   self._metadata(float("nan"), 0.0), scaler=self.scaler)
```

It does not raise if it fails; it logs. Losing the emergency checkpoint must not mask the
original OOM.

## 8. Diagnosing a damaged checkpoint

```powershell
python scripts/verify_checkpoints.py --run checkpoints/micro-pretrain
```

For each step directory this prints validity, tensor count, byte size, and the specific
problems found. Typical diagnoses:

| Symptom | Cause | Fix |
|---|---|---|
| `no COMPLETE marker` | Crash during write | Delete the directory; resume from the previous step |
| `no safetensors weights` | Crash before weights were written | Delete |
| `weight file contains zero tensors` | Disk-full during write | Free space, delete |
| `metadata is not valid JSON` | Crash during metadata write | Delete |
| `loaded N/M tensors` + warnings | Config/weights mismatch | Check `model_config` in metadata |

## 9. Disk pressure

| Artifact | Size |
|---|---|
| `micro` checkpoint (55.7M params, fp32) | ~213 MiB |
| `4b` checkpoint (4.03B params, bf16, sharded) | ~7.5 GiB |
| 4 retained `micro` checkpoints | ~0.9 GiB |

With `keep_last=4, keep_best=2`, a long run holds at most ~6 checkpoints. For the 4B model
on a 67 GiB disk, reduce to `keep_last=2, keep_best=1` or budget explicitly.

## 10. Long-running behaviour

The trainer is designed to be interrupted at any point:

* **Wall-clock budget** (`--max-hours`) stops cleanly at a step boundary.
* **Epoch boundaries** reshuffle the loader with a seeded, reproducible generator.
* **Canary** runs before the first step, so a broken pipeline costs 90 seconds.
* **Every failure mode above** leaves at least one valid checkpoint.

The one thing that is *not* resumable is a partial download during acquisition — see
`docs/datasets.md` §9.