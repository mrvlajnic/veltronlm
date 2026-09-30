# Troubleshooting

Symptoms, causes, and what to actually do. Every command is runnable as written.

---

## 1. Environment and installation

### `ModuleNotFoundError: No module named 'veltron'`

The package is not on the path. Either install it (`pip install -e .`) or set the path:

```powershell
$env:PYTHONPATH = "E:\Posao\testmaxspace"
```

### `torch_directml` not found / DirectML unusable

```powershell
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-directml==0.2.5.dev240914 --no-deps
```

`torch-directml` **pins** `torch==2.4.1`. Installing it with dependencies will silently
downgrade or upgrade torch. Check with `python -m veltron.cli devices`.

### GPU detected but `total_memory_gb` shows 8.0 on a 12 GiB card

Windows `Win32_VideoController.AdapterRAM` reports a 32-bit truncated value (4 GiB for the
RX 6700 XT, whose real capacity is 12 GiB). `veltron.utils.device` treats that field as
untrusted and falls back to a documented 8 GiB assumption for DirectML planning. This is a
Windows reporting bug, not a VeltronLM bug.

---

## 2. Training

### `CANARY FAILED: loss did not collapse`

The trainer refused to start because a fresh model could not memorise 32 windows in 200
steps. Something is wrong with the model or the data.

```powershell
Get-Content checkpoints\<run>\canary.json
```

Likely causes, in order of frequency:

1. **Shard dtype mismatch.** The loader reads uint16 bytes as uint32 and produces garbage
   token ids. `PackedDataset` now catches this from file geometry, but a previously built
   dataset may still have a stale index:
   ```powershell
   python scripts/repair_shard_index.py datasets\dataset-v1
   ```
2. **Labels misaligned.** `compute_loss` owns the shift; a caller that pre-shifts labels
   breaks the objective silently.
3. **Shard file truncated** by a disk-full crash during `build_dataset.py`.

### `RuntimeError: Could not allocate tensor with N bytes`

VRAM exhaustion. Reduce `--batch-size` first; a lower `--batch-size` with higher
`--grad-accum` gives the same effective batch at lower peak memory.

Measured thresholds for `micro` (32k vocab, seq 1024, fp32):

| Micro-batch | Result |
|---:|---|
| 8 | OOM |
| 4 | OOM |
| 2 | works |

Also check no other process is using the GPU. On this host a background Wikipedia crawler
reduced throughput from 3,600 to 295 tok/s before being killed.

### `IndexError: index N is out of bounds for dimension 0 with size 32768`

Shard dtype mismatch. See above — run `scripts/repair_shard_index.py`.

### `NotImplementedError: Could not run 'aten::embedding' with arguments from the 'AutocastPrivateUse1' backend`

DirectML has no `aten::embedding` kernel under autocast. Use `--precision fp32` (the
default on DirectML). `precision=fp16` on DirectML casts parameters directly and never
enters an autocast region, so it does not trigger this.

### Training is much slower than expected

| Cause | Check |
|---|---|
| Another GPU process | `Get-Process python \| Select-Object Id,CPU` |
| Thread oversubscription | `--threads 4` |
| Batch size too small | tokens/s is reported per log line |
| Loss not decreasing | compare `loss_ema` over 50-step windows |

### `repeated_nonfinite_loss` abort

Gradients went non-finite 4 times in a row. Check `grad_norm` in the log just before the
divergence — a rising pre-clip norm is the signature. Lower `--lr` by 2×, or raise
`--warmup`.

### How do I resume after the machine restarted?

```powershell
python -m veltron.train --run-name <run> ...   # --resume auto is the default
```

The newest **valid** checkpoint is used. Torn directories are skipped with a warning.
See `docs/recovery.md`.

---

## 3. Checkpoints

### `checkpoint <dir> is not a complete, resumable checkpoint`

No `COMPLETE` marker, or no weights. The write was interrupted.

```powershell
python scripts/verify_checkpoints.py checkpoints\<run>
```

Delete the damaged directory; resume from the previous one.

### `optimizer state unrestorable; continuing with fresh state`

The optimizer file is corrupt or from an incompatible run. Training continues from the
weights; the first steps have no Adam momentum. Not fatal, but the loss curve will show a
small discontinuity. If it matters, resume from an earlier checkpoint.

### Disk filling up during a 4B run

One 4B fp16 checkpoint with optimizer state is ~24 GiB. Set `--save-interval` higher and
pass fewer retained checkpoints. `CheckpointManager(save_every, keep_last, keep_best)`
controls this; the CLI exposes `--save-interval`.

### `latest valid: NONE`

No checkpoint has a `COMPLETE` marker. If the run was interrupted before the first save,
raise `--save-interval` so checkpoints are written more often.

---

## 4. Generation and inference

### `tokenizer has N tokens but the model was built with vocab_size=M`

The checkpoint and tokenizer do not match. `load_engine` refuses rather than producing
index errors.

```powershell
python -c "import json,pathlib; print(json.loads(pathlib.Path('checkpoints/<run>/step-*/metadata.json').read_text())['tokenizer_version'])"
```

Then point `--tokenizer` at the matching directory.

### Generation is gibberish

1. **Is there a checkpoint?** `python -m veltron.cli info` shows completed runs. An
   untrained model produces fluent-looking nonsense; that is expected.
2. **Is it the right tier?** A `nano` checkpoint served by a `micro` config will not load.
3. **Is greedy sampling being used on an undertrained model?** Try `--temperature 0.8`.

### Generation crashes the process (exit -1073741819)

Access violation, almost always VRAM exhaustion during weight allocation. On this host the
4B fp16 model (7.5 GiB) plus any concurrent GPU job exceeds 8 GiB. Run the benchmark
alone, or use `--quantize 8`.

### Decode throughput is low

DirectML has no fused attention kernel, so decode materialises `B × H × T × T` scores.
This is a backend limitation, not a bug. A CUDA backend would be several times faster.

### Batched output differs from single-prompt output

It should not — that was a real bug, fixed. If you see it, the padding mask is not being
re-supplied on decode steps. `test_explicit_positions_support_left_padding` and the
token-identity check in `scripts/smoke_inference.py` cover this.

---

## 5. Retrieval and RAG

### Serbian queries return nothing relevant

Check that `expand_query` has the term. The bilingual map covers ~50 domain terms.

```powershell
python -c "from veltron.rag.retriever import expand_query; print(expand_query('Koliko traje garancija?'))"
```

### Every query scores 1.0

Rank fusion normalises to 1.0 by construction. Use `confidence`, not `score`, for
thresholds. `test_confidence_is_absolute_not_normalised` guards this.

### The system escalates everything

`MIN_RETRIEVAL_SCORE = 0.18` is tuned for this knowledge base. Inspect the actual
confidence:

```powershell
python -m veltron.inference.cli ask --checkpoint checkpoints\micro-pretrain "your question" --json
```

If a documented question scores below 0.18, retrieval is failing, not the threshold.

### Retrieval returns the wrong document for "how much is the Plus plan"

Known and fixed by the title boost. If it regressed, rebuild the index:

```powershell
python -m veltron.cli index models/rag_index
python scripts/verify_rag.py
```

### Citations show a number that was never retrieved

That is the point of `verify_citations`. It appears in `usage.citation_check.fabricated`.

---

## 6. Data pipeline

### `Reusing existing raw` when I expected a re-download

`collect_raw` reuses `datasets/raw/*.jsonl` unless `--force` or `--re-acquire` is passed.

### The dataset is 98% Project Gutenberg

Expected: `gutenberg.jsonl` is 378 of 384 Mcharacters. The mixture stage resamples it down
to 10%. If the mixture appears wrong, check `manifest.json → mixture.achieved`.

### Wikipedia fetch returns HTTP 429

Wikimedia rate-limits anonymous clients. The acquisition code backs off exponentially. Reduce
`--wikipedia-random` or fetch in smaller batches.

### `worst of the failure budget` table

| Failure | Counter in `summary.json` | Immediate action |
|---|---|---|
| OOM | `oom` | lower `--batch-size` |
| NaN loss | `nan_loss` | lower `--lr`, raise `--warmup` |
| Gradient explosion | `gradient_explosion` | lower `--lr` |
| Corrupt checkpoint | `corrupt_checkpoint` | resume from an earlier step |
| Disk write failure | `io_error` | free space; check permissions |

---

## 7. Tests and CI

### A test fails only on the GPU

GPU-dependent behaviour is marked. Run CPU-only:

```powershell
python -m pytest tests -q -m "not gpu"
```

### `test_4b_adamw_state_does_not_fit_in_12gb` fails on a bigger machine

Working as intended: it asserts the *current* blocker. When the project moves to hardware
with more than 60 GiB of VRAM it should start failing, which is the signal to revisit the
BLOCKED note in `docs/environment.md` and `FINAL_REPORT.md`.

### `mypy` reports many errors

Mypy is not a CI gate yet; the codebase is mid-migration. Track in `docs/roadmap.md`.

---

## 8. Getting help

1. `python -m veltron.cli info` — backends, registry, which artefacts exist.
2. `python scripts/verify_checkpoints.py <run>` — checkpoint integrity.
3. `python scripts/verify_rag.py` — retrieval regression.
4. `reports/*.json` — every benchmark writes a machine-readable result.
5. `FINAL_REPORT.md` — what was measured and what remains blocked.