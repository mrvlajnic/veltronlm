# HANDOFF - VeltronLM

> ### CURRENT STATE -- 2026-10-02 22:40
>
> **Nothing is training. The `mini` run COMPLETED cleanly at 22:37.**
>
> | | |
> |---|---|
> | Latest checkpoint | `checkpoints/mini-pretrain/step-00004134` |
> | Best checkpoint | `checkpoints/mini-pretrain/step-00003250` |
> | Model | `veltronlm-mini`, 244,354,048 params |
> | Tokens trained | 33,865,728 (1.74 epochs) |
> | **Train loss** | **6.15 -> 3.42** -- the number to quote |
> | Stop reason | `wall_clock_budget`, clean, 8 h 1 m |
> | Failure budget | **all zero** (no OOM / NaN / grad explosion / corruption) |
> | Checkpoints | 5, all valid, 17.59 GiB; 44 GiB free disk |
> | Scheduled tasks | none -- the keepalive was unregistered |
> | Git | clean, pushed to `github.com/mrvlajnic/veltronlm` |
> | Tests | 184 passing |
>
> The model is a **BASE** model: it continues text, it does not follow instructions and
> does not chat. SFT and DPO are implemented but have never been run.
>
> **Resume:** `powershell -ExecutionPolicy Bypass -File scripts\windows\start_veltron_training.ps1`
> (verified to resume from step 4,134, not restart)
> **Status:** `powershell -ExecutionPolicy Bypass -File scripts\windows\status_veltron_training.ps1`
> **Safe stop:** `powershell -ExecutionPolicy Bypass -File scripts\windows\stop_veltron_training.ps1`
>
> ### Do not quote val perplexity 3.0
>
> It sits *below* the training loss (1.18 vs 3.42), which is impossible for real
> generalisation. The validation split is **6 documents** and evaluation samples **8,176**
> of its 51,287 tokens. The curve is non-monotone and flattened while training loss kept
> falling. Building a proper validation split is the highest-value next action.
>
> ### Everything below the banner is historical session log
>
> Sections 1-4 and the dated entries record what was true at the time and have been
> superseded. Keep them for the reasoning and for the incident record, but do not read
> "training is running" there as current.


---

**Read this first.** Written 2026-10-01 ~00:15, mid-project. This document is the entry
point for anyone continuing the work, including a future session that has no memory of the
project.

> ### Why this file exists
>
> AI coding sessions do **not** carry memory between conversations. If you open a new
> session, that session knows nothing about VeltronLM, this machine, the measured numbers,
> the bugs found, or the decisions made. **This file plus the linked documents are the
> entire handoff.**
>
> Do not rely on conversational history. Read this, then `FINAL_REPORT.md`, then the
> specific document for the area you are working on.

---

## 1. TL;DR

A real ~4B-parameter open-weight decoder-only Transformer plus a customer-support AI
system, built from scratch in PyTorch. Everything is implemented, tested and measured
except the single thing the hardware cannot do: **train the 4B model from scratch.**

| | |
|---|---|
| Repo root | `E:\Posao\testmaxspace` |
| Git | **committed `3ef258e` on `main`, NOT pushed** — no remote, no `gh`, no token (§13) |
| Python | 3.10.11 at `C:\Users\Gamer\AppData\Local\Programs\Python\Python310\python.exe` |
| **Required env var** | `$env:PYTHONPATH = "E:\Posao\testmaxspace"` (package is not pip-installed) |
| Tests | `python -m pytest tests -q` → **181 passed** |
| GPU | AMD Radeon RX 6700 XT, 12 GiB, usable via `torch-directml` |
| CUDA | none |
| Live training | PID 5260, `micro` pretraining, 5-hour wall-clock budget |

### The one hard blocker

`veltronlm-4b-base` has **4,026,765,312 verified parameters and no weights.** AdamW with
fp32 master weights needs **60.00 GiB** of optimizer state against **12 GiB** of VRAM — a
6.0x shortfall. Chinchilla-optimal 4B training is a ~22-year single-GPU run at the measured
2.80 TFLOP/s matmul peak.

This is reported as a blocker **with arithmetic** in `experiments/EXP-0004.md`, not as an
excuse. The response was to build everything and train the largest tier that fits.

---

## 2. Read in this order

| # | Document | Why |
|---|---|---|
| 1 | **`HANDOFF.md`** (this file) | Where things stand right now |
| 2 | [`FINAL_REPORT.md`](FINAL_REPORT.md) | Complete engineering accounting |
| 3 | [`docs/environment.md`](docs/environment.md) | Host audit, measured throughput, feasibility math |
| 4 | [`docs/roadmap.md`](docs/roadmap.md) | Phase status + prioritised next actions |
| 5 | Area-specific doc | listed in §6 |

**Do not** trust conversational summaries from previous sessions. Verify with the artefacts.

---

## 3. What works right now

Verify these claims before relying on them:

```powershell
$env:PYTHONPATH = "E:\Posao\testmaxspace"
python -m pytest tests -q                              # expect: 181 passed
python scripts/verify_param_count.py                   # expect: PASS, 4b = 4,026,765,312
python scripts/verify_rag.py                           # expect: hit@1 14/14, recall@3 14/14
python scripts/verify_checkpoints.py checkpoints\micro-pretrain
python -m veltron.evaluate --retrieval                 # expect: hit@1 = 1.0
python -m veltron.cli info                             # artefact inventory
```

| Subsystem | Status | Evidence |
|---|---|---|
| 4B architecture | VERIFIED | causality exact (0.0), KV cache within 9.2e-07, batched == single-prompt |
| Parameter counts | VERIFIED | analytic == `meta`-device for all 7 tiers |
| 4B pretraining | **BLOCKED** | 60 GiB vs 12 GiB VRAM |
| Tokenizer | TRAINED | 2 vocabularies, 0 round-trip failures / 12 script cases |
| Dataset | BUILT | `dataset-v1`, 19,485,297 train tokens, full provenance |
| `micro` pretraining | RUNNING | val loss 5.762 → 4.564, ppl 317.9 → 96.0 |
| SFT pipeline | TESTED | 604 examples, completion-masked loss verified |
| DPO | TESTED | objective unit-verified, **no run completed** |
| RAG retrieval | BENCHMARKED | **hit@1 100%, recall@3 100%** |
| API + chat UI | TESTED | 15 routes |
| Comparison vs Qwen/etc | **NOT CLAIMED** | no baseline was ever run |

---

# 2026-10-02 14:20 — UNATTENDED RUN LIVE

Read this section first; it supersedes everything above about running state.

## Training is RUNNING right now

```
PID 6084   checkpoints/mini-pretrain   8-hour budget, ends about 20:15
python -m veltron.train --config configs/pretrain_mini_evening.yaml \
        --set stop_file=<run>/STOP_REQUEST
```

Resumed from **step-00000500**. At the time of writing: step ~650, loss ~5.59,
~1,090 tok/s, ~11.8 s/step. Expect roughly step 3,000 / 24M tokens by the end,
about 1.3 epochs.

PID and parameters: `checkpoints/mini-pretrain/AUTORUN_INFO`

## Why batch size changed to 1 (this is important)

The run had been dying at **step ~550 every time**, reproducibly, with:

```
RuntimeError: Could not allocate tensor with 134217728 bytes.
```

`134217728` is exactly `2 x 512 x 32768 x 4` — **the vocabulary projection**. It is not a
random tensor. `mini` holds 3.72 GiB of persistent fp32 optimizer state
(params + grads + Adam m + v for 244M parameters), which leaves too little headroom for a
134 MiB transient plus activations.

Fix: `micro_batch_size` 2 -> 1, `grad_accum_steps` 8 -> 16. Same 8,192-token effective
batch, half the projection (67 MiB) and half the activations. Measured ~1,100 tok/s, no
OOM. `configs/pretrain_mini_evening.yaml` is updated.

**fp16 weights were tried and are NOT viable.** `precision=fp16_weights` resumed correctly
and then logged **zero steps in 45 minutes** before being killed, with no error. AdamW's
moving average already falls back to the host for `aten::lerp` on DirectML; in fp16 it is
slower still. The config pins `precision: fp32` with a comment saying so.

## Safety net is installed

**Scheduled task "VeltronLM Keepalive"** — every 20 minutes for 24 h, starting 2 minutes
after registration.

It calls `start_veltron_training.ps1`, which refuses to start when a trainer is already
running. So the repeated firings do nothing while training is healthy, and a crashed run is
restarted within 20 minutes. Verified by firing it manually during a live run: **no
duplicate started, the running trainer was untouched.**

```powershell
Get-ScheduledTask -TaskName "VeltronLM Keepalive" | Get-ScheduledTaskInfo   # status
Start-ScheduledTask  -TaskName "VeltronLM Keepalive"                        # run now
Disable-ScheduledTask -TaskName "VeltronLM Keepalive"                       # pause
powershell -ExecutionPolicy Bypass -File scripts\windows\install_scheduled_task.ps1 -Remove
```

Wake timers are irrelevant here: **sleep is disabled on this machine**, so the task always
finds the PC awake. It still sets `WakeToRun`, which is a no-op in that case.

## Desktop shortcuts created

Location: **`C:\Users\Gamer\OneDrive\Desktop`** (OneDrive-redirected Desktop)

* `VeltronLM - Resume Training.lnk`  — 8-hour window, stops cleanly
* `VeltronLM - Stop Training.lnk`    — safe stop, never a raw kill
* `VeltronLM - Status.lnk`           — read-only dashboard, `-Watch`

## When you get home (~19:30-20:00)

Training will still be running. To let it finish and keep the checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\shutdown_after_training.ps1
```

That waits, verifies, prints the preserved checkpoint, and **does not shut down** unless
you pass `-ShutdownWhenDone`. Or just leave it; the 8-hour budget ends ~20:15 on its own.

To stop it now instead:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\stop_veltron_training.ps1
```

To look at progress:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\status_veltron_training.ps1
```

## Disk note

Each `mini` checkpoint is **3.77 GiB**. Config keeps `keep_last: 3, keep_best: 2`, so up to
5 checkpoints = ~19 GiB. There was 54 GiB free, so this fits, but check `free disk` in the
status output. If it gets tight, lower `keep_last` to 2.

---

# 2026-10-02 01:35 — WINDOWS AUTOMATION BUILT; mini run needs resuming

Read this section before anything else. It supersedes section 4.

## State right now

**No training is running.** The GPU is free.

| | |
|---|---|
| Latest valid checkpoint | `checkpoints/mini-pretrain/step-00000500` |
| Step / tokens | 500 / 4,096,000 |
| Model | `veltronlm-mini`, 244,354,048 params |
| val loss / perplexity | 6.9695 / 1063.7 (at step 250 it was 7.5122 / 1830.2) |
| Checkpoints on disk | step-00000250, step-00000500 — both valid, 7.03 GiB total |
| Free disk | 54 GiB |

## How to resume (one command)

```powershell
$env:PYTHONPATH = "E:\Posao\testmaxspace"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\start_veltron_training.ps1
```

Verified to resume from step 500, NOT from step 0. `configs/pretrain_mini_evening.yaml`
already contains `resume: auto`.

## INCIDENT — how the mini run stopped, and why

The run died at ~step 550 with:

```
RuntimeError: Could not allocate tensor with 134217728 bytes.
There is not enough GPU video memory available!
```

**Cause: I caused it.** While testing the automation I launched a second, isolated test
trainer on a different config. Two trainers on one 12 GiB card exhausted VRAM between them
and the real run OOM'd.

Per-config duplicate detection had allowed it, because the second trainer used a different
config file. That check has been replaced: `start_veltron_training.ps1` now refuses if
**any** `veltron.train` process exists, whatever its config, unless `-ForceConcurrent` is
passed. Exit code 11 means "another trainer is running".

**Nothing was lost.** Checkpoints are written atomically (staging dir, then rename, then a
COMPLETE marker), so a crash mid-write cannot corrupt one. Step 500 verified valid
immediately afterwards. This is the exact scenario `docs/recovery.md` describes.

**Rule going forward: never run a second trainer, even a test one, while the real run is
active.** Use `status_veltron_training.ps1` first.

## Automation scripts (all under scripts/windows/)

| Script | Purpose |
|---|---|
| `status_veltron_training.ps1` | Read-only dashboard. Safe while training. `-Watch` to poll |
| `start_veltron_training.ps1` | Start/resume. Refuses duplicates. `-Background`, `-DryRun`, `-MaxHours` |
| `stop_veltron_training.ps1` | Cooperative stop via stop file. `-Force` waits for the next checkpoint first |
| `run_veltron_training_window.ps1` | `-Hours N` then stops cleanly and prints a summary |
| `shutdown_after_training.ps1` | Wait, verify, report. Shuts down ONLY with `-ShutdownWhenDone` |
| `install_desktop_shortcut.ps1` | Creates the three desktop shortcuts |
| `install_scheduled_task.ps1` | Creates "VeltronLM Auto Resume". `-Remove` cancels it |
| `veltron_common.ps1` | Shared helpers. Dot-sourced, not run directly |

Also: `scripts/veltron_state.py` — JSON training state, reusing the trainer's own
`CheckpointManager` so PowerShell cannot disagree with the trainer about what is resumable.

Supporting change: `TrainConfig.stop_file` (default `""`, so no behaviour change for
existing runs). When set, the trainer polls it once per step and stops at a step boundary —
after the in-flight step, after any due evaluation, after any due checkpoint — then runs
its normal final-eval and final-save path.

## What was NOT done yet

* Desktop shortcuts were never installed (installer written and parse-checked only)
* The scheduled task was dry-run (`-WhatIf`) only; no task exists
* Wake timers are still DISABLED
* These scripts are uncommitted only in the sense that they are not yet pushed

## Wake-from-sleep findings (measured, not assumed)

```
Modern Standby (S0 low power idle) : present
Standby S3                         : available
Hibernate                          : available
active power scheme                : My Custom Plan 1
wake timers (AC)                   : DISABLED
```

* Waking from **S3 sleep**: possible once the wake-timer setting is on —
  `install_scheduled_task.ps1 -EnableWakeTimers`
* Waking from **hibernation**: needs an RTC alarm in BIOS/UEFI. Windows cannot verify
  this; firmware settings were not touched
* Waking from **full shutdown**: not possible via Task Scheduler alone; needs BIOS RTC
  alarm or Wake-on-LAN from an always-on device

## Bugs found by testing the automation

Each was found by running the scripts, not by reading them:

1. `-MaxHours` multiplied by 3600, handing the trainer 108 *hours* for a 108-second window
2. Process detection matched the run name, which never appears in the command line — only
   the config filename does. It reported "not running" while the trainer held 10.8 GiB
3. `Split-Path -Leaf $null` threw whenever a run directory did not exist — a normal state
   on a first run
4. Merging stderr into the success stream turned a harmless pynvml deprecation warning
   into a terminating `NativeCommandError`, so a successful run reported failure
5. Number formatting used the machine locale: perplexity 1063.7 rendered as `1.063,7`
6. Duplicate detection was per-config, which is what allowed the incident above

All fixed. 184 tests pass.

## Verified by test

| Test | Result |
|---|---|
| status while training active | PASS, found PID 17860, GPU breakdown, evals |
| duplicate protection, same config | PASS, exit 10, refused |
| duplicate protection, different config | PASS, exit 11, refused |
| checkpoint discovery + validity | PASS, uses repo verifier |
| manual resume | PASS, `resumed at step 26`, LR curve continued |
| bounded window, 1.9 min | PASS, `stop reason: stop_file_requested`, checkpoint kept |
| bounded window, 1.3 min | PASS after fixing the 3600x bug |
| foreground start/resume | PASS, exit 0 |
| graceful stop via stop file | PASS |
| paths containing spaces | PASS |
| logs written | PASS, `logs/mini_train_*.log` |
| scheduled task dry run | PASS, `-WhatIf` |
| tests still pass | PASS, 184 |

---

## 4a. LIVE RUN: veltronlm-mini (244,354,048 params)

Started 2026-10-01 ~23:47. **Leave it running overnight.**

`
PID 17860   checkpoints/mini-pretrain   11-hour wall-clock budget
python -m veltron.train --config configs/pretrain_mini_evening.yaml
`

| | |
|---|---|
| Parameters | 244,354,048 (16 layers, d_model 1024, 16/4 heads, SwiGLU 2752) |
| seq / batch / accum | 512 / 2 / 8 = **8,192 tokens per optimiser step** |
| Measured throughput | **~1,690 tok/s** |
| loss_chunk_tokens | 256 |
| Evaluations | every 250 steps |
| Checkpoints | every 250 steps, keep_last 3, keep_best 2, resume auto |
| Expected steps in 11h | ~5,800 of 6,000 |
| One epoch of dataset-v1 | 3.2 h, so ~3.4 epochs expected |

### Three fixes were needed to get mini past step ~2

Each was found by measuring, not reasoning, and each produced a wrong diagnosis first:

1. **The canary ran in-process.** DirectML's allocator grows its heap and never returns
   it, so the canary's several GiB stayed reserved and training OOMed inside
   cross_entropy almost immediately. Fixed by running the canary in a subprocess
   (eltron/training/canary.py). Symptom looked like a memory-budget problem; the
   identical config survived 40 micro-steps standalone (scripts/stress_accum.py).

2. **The canary rewrite used random token ids.** Uniformly random ids over a 32k
   vocabulary are incompressible, so ln(V) is the loss floor and memorisation is
   impossible. It scored a 2.28x reduction and the gate correctly rejected it -- which is
   the gate working as intended. Fixed to read real dataset windows.

3. **The NaN gradient check allocated a bool copy of every gradient.**
   	orch.isfinite(g).all() materialises a full-size temporary, ~256 MiB for this model,
   and failed on a nearly-full heap. Replaced with 	orch.linalg.vector_norm, a fused O(1)
   reduction that propagates NaN and Inf into the scalar result.

Plus loss_chunk_tokens 4096 -> 256: at a 32k vocabulary one chunk needs ~1.5 GiB for a
single cross_entropy call.

### Canary threshold

Set to 2.5x reduction, chosen from measured history rather than picked:

| Reduction | Case | Verdict |
|---|---|---|
| 1.22x | canary v1, 256 windows x 1024 tokens | fail |
| 1.39x | canary v2, 32 windows x 1024 tokens | fail |
| 2.28x | canary on random token ids | fail (correctly rejected its own bad input) |
| 3.13x | canary on real windows, subprocess | **pass** |
| 399x | earlier in-process canary | pass |

### Check on it

`powershell
Get-Content checkpoints\mini-pretrain\train_log.jsonl | Select-String '"eval"' | Select-Object -Last 5
python scripts\verify_checkpoints.py checkpoints\mini-pretrain
`

## 4. Previous run: veltronlm-micro (complete)

**Do not benchmark or start other GPU work while this is running** — see
`experiments/EXP-0013.md`. A background crawler once cut throughput from 3,600 to 295
tok/s.

```
PID 5260   checkpoints/micro-pretrain   5-hour wall-clock budget
python -m veltron.train --model micro --tokenizer models/tok-mini-32k \
  --dataset datasets/dataset-v1 --run-name micro-pretrain \
  --seq-len 1024 --batch-size 2 --grad-accum 8 --max-steps 4000 --max-hours 5.0 \
  --lr 6e-4 --warmup 120 --schedule cosine --eval-interval 250 \
  --save-interval 250 --log-interval 25 --resume none --seed 1234
```

### Measured so far

| Step | Tokens seen | Train loss | Val loss | Val perplexity |
|---:|---:|---:|---:|---:|
| 0 | 0 | 9.55 (`ln V`) | — | — |
| 250 | 4,096,000 | 6.087 | 5.762 | 317.9 |
| 500 | 8,192,000 | 4.555 | 5.101 | 164.3 |
| 750 | 12,288,000 | 5.414 | **4.564** | **96.0** |

At step 825 / 13,516,800 tokens when this file was written. ~3,300–3,700 tok/s sustained.

### Run status

The micro run was **terminated externally** at step 1175 (no torn staging directory, no
traceback — a clean external kill, most likely the session ending). summary.json was
therefore never written, but all four checkpoints are valid and step-00001000 is
intact with **validation loss 4.198 / perplexity 66.6**.

Resume it, or start the bigger tier instead:

`powershell
# resume micro where it stopped
python -m veltron.train --model micro --run-name micro-pretrain --resume auto ...

# or train the 117m tier (see below)
python -m veltron.train --config configs/pretrain_117m.yaml
`

### The 117m tier — verified trainable

Added after the micro run finished, in response to "can we hit 117 million".
**117,027,592 parameters**, found by grid search (scripts/feasibility_117m.py) rather
than hand-tuned.

| | |
|---|---|
| Layers / d_model | 11 / 704 |
| Heads | 16 query / 4 KV, head_dim 44 |
| FFN (SwiGLU) | 2,464 |
| Optimizer state | **1.74 GiB** |
| Measured throughput | **847 tok/s** at b=2 seq=512; **764 tok/s** at b=1 seq=1024 |
| Effective TFLOP/s | 0.595 (21.3% of the 2.80 TFLOP/s matmul peak) |
| One epoch of dataset-v1 | **6.4 hours** |
| =1 seq=2048 / =2 seq=1024 | **OOM** — exceeds the 3.62 GiB ceiling |

**Answer: yes.** The blocker for 4B is optimizer state, not GPU class. At 117M the
optimizer state is 1.74 GiB against a measured 3.62 GiB ceiling.

### Measured VRAM ceiling: 3.62 GiB, not 12 GiB

scripts/measure_vram.py binary-searches the largest allocation DirectML will satisfy:
**3,712 MiB**. This GPU drives the desktop, so the compositor holds a share of the 12 GiB
that compute cannot reclaim. Every memory figure in this project is measured against that
ceiling, not the nameplate.

| Tier | Optimizer state | Fits 3.62 GiB? |
|---|---:|---|
| 4b | 60.00 GiB | **no — 16.6x over** |
| small | 11.26 GiB | **no — 3.1x over** |
| mini | 3.91 GiB | **no — over before activations** |
| **117m** | **1.74 GiB** | **yes** |
| micro | 0.89 GiB | yes |

This corrects the earlier docs/environment.md figure of 12 GiB usable.

### When the mini run finishes

1. `checkpoints/micro-pretrain/summary.json` appears.
2. **Evaluate immediately** — this is the highest-value next step:
   ```powershell
   python -m veltron.evaluate --checkpoint checkpoints/micro-pretrain \
       --suites support adversarial --perplexity --retrieval --long-context \
       --max-items 40 --out reports/eval-micro.json
   ```
3. **Run SFT** on the best checkpoint:
   ```powershell
   python -m veltron.finetuning.cli train --base checkpoints/micro-pretrain
   ```
4. **Build preferences and run DPO**:
   ```powershell
   python -m veltron.finetuning.cli build-preferences --checkpoint checkpoints/sft-support
   python -m veltron.finetuning.cli dpo --init checkpoints/sft-support
   ```
5. **Re-benchmark with an idle GPU** — all throughput numbers must be measured with nothing
   else running:
   ```powershell
   python scripts\bench_compute.py
   python scripts\bench_inference.py --checkpoint checkpoints/micro-pretrain \
       --tiers micro 4b --quantize 8 4
   ```
   The 4B tier needs ~7.5 GiB in fp16 and **will fail if any other process holds the
   GPU**. Run it alone; report a `FAILED` entry with the reason if it does not fit.
6. **Regenerate the status report and update the numbers** in `README.md`,
   `MODEL_CARD.md`, `experiments/EXP-0008.md` and `FINAL_REPORT.md`.

---

## 5. Repository map

```
E:\Posao\testmaxspace\
├── veltron/                     the package
│   ├── model/                   architecture, config, safetensors IO, quantization
│   ├── tokenizer/               byte-level BPE training + chat templates
│   ├── data/                    sources, acquire, filters, pipeline, loader
│   ├── training/                schedule, checkpoint, trainer
│   ├── finetuning/              SFT data + trainer + CLI
│   ├── alignment/               preference pairs + DPO
│   ├── rag/                     ingest, retriever, pipeline
│   ├── support/                 knowledge_base.py (content), triage.py (rules)
│   ├── inference/               generator, engine, CLI
│   ├── serving/app.py           FastAPI
│   ├── chatbot/static/          chat UI (single HTML file)
│   ├── evaluation/              datasets, metrics, runner
│   ├── utils/                   device, logging, config, hashing, seed
│   ├── train.py  evaluate.py  serve.py  cli.py
│   └── benchmarks/
├── tests/unit/                  151 tests
├── tests/integration/           30 tests
├── scripts/                     reproducible entry points
├── configs/                     8 YAML configs
├── docs/                        16 documents
├── experiments/                 13 records + template + README
├── datasets/
│   ├── raw/*.jsonl              ~510 MB (gitignored; re-acquirable)
│   └── dataset-v1/              manifest, provenance, train/val/test shards
├── models/
│   ├── tok-mini-32k/            trained tokenizer
│   ├── tok-4b-64k/              trained tokenizer
│   └── rag_index/               retrieval index
├── checkpoints/micro-pretrain/  live run
├── reports/                     machine-readable benchmark outputs
├── knowledge_base/              overlay dir for extra KB docs
├── docker/                      Dockerfile + compose
├── .github/workflows/ci.yml
├── README.md  MODEL_CARD.md  FINAL_REPORT.md  DATA_SOURCES.md
├── CHANGELOG.md  CONTRIBUTING.md  SECURITY.md  LICENSE
└── pyproject.toml
```

### Artefacts that must not be deleted

| Path | Why |
|---|---|
| `models/tok-mini-32k/` | Required by every training run and by `micro` inference |
| `models/rag_index/` | Retrieval; rebuildable with `python -m veltron.cli index` but takes 12 min |
| `datasets/dataset-v1/manifest.json` | Provenance chain back to `dataset_manifest_hash` in checkpoints |
| `datasets/sft/sft.jsonl` | SFT data; 73 s to rebuild |
| `checkpoints/micro-pretrain/step-*/` | **The only trained weights in the project** |

`datasets/raw/` (510 MB) is gitignored and re-acquirable, but re-acquisition is **not
deterministic** for Wikipedia's random-article generator. Reuse it if you need a
bit-identical dataset.

---

## 6. Where to read for each area

| Working on | Read |
|---|---|
| Architecture, params | `docs/model-architecture.md` |
| Training, hyperparameters | `docs/training.md` |
| Checkpoints, resume, crashes | `docs/recovery.md` |
| Data pipeline, sources, licences | `docs/datasets.md`, `DATA_SOURCES.md` |
| Tokenizer | `docs/tokenizer.md` |
| Retrieval, grounding, citations | `docs/rag.md` |
| Support triage, escalation | `docs/customer-support.md` |
| SFT | `docs/finetuning.md` |
| DPO | `docs/alignment.md` |
| Evaluation, metrics | `docs/evaluation.md` |
| Generation, quantization | `docs/inference.md` |
| API, chat UI | `docs/serving.md` |
| Threat model | `docs/security.md` |
| Anything is broken | `docs/troubleshooting.md` |

---

## 7. Environment setup

```powershell
$env:PYTHONPATH = "E:\Posao\testmaxspace"
$env:PYTHONIOENCODING = "utf-8"    # Serbian output crashes cp1252 otherwise
```

`torch-directml==0.2.5.dev240914` **pins `torch==2.4.1`**. Installing it with dependencies
will change the torch version.

```powershell
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-directml==0.2.5.dev240914 --no-deps
pip install tokenizers safetensors numpy pydantic fastapi uvicorn python-multipart pytest ruff
```

### DirectML gotchas — these are not optional workarounds

| Limitation | Consequence |
|---|---|
| No `aten::embedding` under autocast | `torch.autocast` **only** on CUDA. On DirectML fp16 = `model.half()`. `--precision fp32`. |
| Cannot materialise a computed 4-D bool tensor | Attention uses **additive float masks**, never `masked_fill` with a computed mask. |
| `aten::lerp` falls back to host | Optimiser is partly host-bound. |
| No fused attention | `B × H × T × T` scores are materialised. |
| `Win32_VideoController.AdapterRAM` lies | Reports 4 GiB for a 12 GiB card. Treated as untrusted; falls back to 8 GiB. |

---

## 8. Bugs found and fixed

All regression-tested. Listed because they are the non-obvious traps in this codebase.

| Bug | Symptom | Root cause |
|---|---|---|
| Byte tokenizer mangles non-ASCII | Serbian → `"    .  1  ."` | Latin-1 vocab instead of GPT-2 `bytes_to_unicode` |
| Byte tokenizer drops spaces | `"TheVeltronHubX1"` | `add_tokens` bypasses pre-tokenization |
| Shard dtype mismatch | `IndexError: index 2172472 ... size 32768` | Index hardcoded `uint32`, data written `uint16` |
| KV cache reads uninitialised memory | Wrong prefill | Length tracked internally, not passed as offset |
| Batched generation diverges | batch ≠ solo | Padding mask dropped on decode steps |
| `normalize` corrupts code | Indents 8 → 2 spaces | Leading whitespace collapsed |
| MinHash irreproducible | Dedupe changes every run | `hash()` randomised per process |
| Empty-matching regex | Words split per character | Alternation with `*` on an empty branch |
| Retrieval confidence always high | Escalation never fires | Rank-normalised score used as absolute |
| Augmentation manufactures evidence | Swallow question scored 0.725 | Augmented query fed the confidence calc |
| `tokenize` drops Cyrillic | `['24']` only | ASCII-only `[a-z]`; `re.IGNORECASE` does not widen it |
| `field()` outside a dataclass | `'Field' object does not support item assignment` | `AppState` was a plain class |
| Checkpoint stores train loss as `val_loss` | Best-checkpoint selection compared train vs train | Interval save had no evaluation |
| SFT summaries are boilerplate | Targets were `[Illustration]` | No boilerplate filtering in extraction |
| `other` category reports 0.95 confidence | `"hello"` → certain "Unclassified" | Margin formula with no runner-up |
| "I forgot my password" → `security_incident` | Most common legitimate question misrouted | Bare `password` in the secret markers |
| Query rewrite deletes the subject | `my VeltronHub X1 is not working` → `working` | Stripped the framing, not just filler |
| `canary` false failures (×2) | Refused to train correct code | Sampling too widely; then windows too long |

---

## 9. Critical invariants — do not break these

```
1. Analytic param algebra == meta-device instantiation, for all 7 tiers.
   test_analytic_param_count_matches_meta_device

2. Attention is exactly causal.
   test_attention_is_causal -> 0.0 delta before an edit

3. KV-cache decode == full forward.
   test_kv_cache_matches_full_forward -> < 1e-4

4. Batched generation == single-prompt, token-identical.
   test_explicit_positions_support_left_padding + scripts/smoke_inference.py

5. SFT loss is masked to assistant tokens only.
   test_sft_loss_ignores_masked_positions (perturb logits under mask, loss unchanged)

6. Shard dtype is validated against file geometry.
   test_loader_rejects_dtype_mismatch_with_data

7. Retrieval confidence is ABSOLUTE, not rank-normalised.
   test_confidence_is_absolute_not_normalised

8. Fabricated citations are detected.
   test_fabricated_citation_is_detected

9. Refusal examples exist in the SFT set.
   test_sft_dataset_contains_refusals_and_grounded
```

---

## 10. House rules

1. **Never describe an unmeasured capability as working.** If a number appears anywhere, a
   command produced it.
2. **One variable per experiment.** Record failures — `EXP-0007` (two false canary
   failures) and `EXP-0010` (summaries of Gutenberg boilerplate) are the most useful
   records in the repo precisely because they went wrong.
3. **Never add an external model API.** The whole point is that the tokenizer, the weights
   and the logits are produced locally.
4. **Benchmark only on an idle GPU.**
5. **Comments explain *why*, including the rejected alternative.** The most valuable comments
   in this repo record a measurement that decided a design.
6. **No `except Exception: pass`.** Catch the specific error, fix the cause, or let it
   propagate.
7. **`Not evaluated` is a valid result.** Writing it is mandatory where a benchmark could
   not run; guessing is not.

---

## 11. Prioritised next actions

Full detail in `docs/roadmap.md`.

1. **Wait for the training run, then evaluate it** (§4). Highest value.
2. **Complete SFT → DPO → evaluation** on the trained checkpoint.
3. **More Serbian data.** ~1% of the corpus (21 documents) is the largest quality ceiling.
   Wikipedia rate-limits anonymous clients (HTTP 429).
4. **Re-benchmark on an idle GPU**, including the 4B tier in int8 (3.75 GiB, fits).
5. **Run the three designed sweeps**: `exp_lr.py`, `exp_mixture.py`, `exp_tokenizer.py`.
   The scripts exist; **no sweep has been run and no result is claimed.**
6. **Red-team the RAG path.** Injection detection is a blocklist, a known weakness.
7. **Add toxicity filtering.** Required before any public release; currently absent from
   the pretraining corpus.
8. **Baseline comparison.** Run a Qwen-class model on the same suites. Only then is any
   comparative statement defensible.

---

## 12. If something is wrong

1. `python -m veltron.cli info` — what exists on this machine
2. `python scripts/verify_checkpoints.py checkpoints/micro-pretrain` — checkpoint integrity
3. `python scripts/verify_rag.py` — retrieval regression
4. `python -m pytest tests -q` — 184 tests
5. `docs/troubleshooting.md` — symptom-indexed fixes
6. `reports/*.json` — every benchmark wrote a machine-readable result
7. **`git log`** — the repo is **not** a git repository yet. See §13.

---

## 13. Git state — READ THIS FIRST

**Work is committed. It is NOT pushed. There is no remote.**

```
commit  3ef258e   feat: VeltronLM 0.1.0-alpha ...
branch  main
files   154 tracked
```

The repo was created in this session (it did not exist before). `.gitignore` excludes
`datasets/raw/` (510 MB), all ~19,000 dataset shard files, `checkpoints/` (all weights),
`models/*/tokenizer.json` and the duplicate `.venv-gpu/`.

### To publish

There is **no `gh` CLI and no `GITHUB_TOKEN`/`GH_TOKEN` in the environment**, and no remote
is configured, so the push was not possible from this session. To finish it:

```powershell
# 1. Create the empty repo on github.com (or wherever), then:
git remote add origin https://github.com/<you>/veltronlm.git
git push -u origin main

# or with a token, non-interactively:
git remote add origin https://github.com/<you>/veltronlm.git
git -c http.extraHeader="AUTHORIZATION: basic $env:PAT" push -u origin main
```

Alternatively install `gh` (`winget install GitHub.cli`), run `gh auth login`, then
`gh repo create veltronlm --public --source . --push`.

Nothing else is required — the working tree is clean and ready.

### Also outstanding

| Gap | Action |
|---|---|
| 10 ruff warnings remain | All cosmetic (`F841` unused locals, `B007` unused loop vars). The three real ones — `F821 undefined 'asdict'`, `F821 undefined 'k'`, `F601 duplicate "rs" key` — **are fixed**. |
| `mypy` not a CI gate | `mypy veltron --ignore-missing-imports` and triage. |
| CI never executed | `.github/workflows/ci.yml` references `httpx`, needed for `TestClient`. Verify the install list. |
| Docker never built | `docker compose -f docker/docker-compose.yml build`. |
| `git prune` warning | "too many unreachable loose objects" from the interrupted `git add`. `git gc --prune=now` when convenient. |
| README results table | From step 750. Update after the run completes. |

---

## 14. Quick verification script

```powershell
$env:PYTHONPATH = "E:\Posao\testmaxspace"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=== tests ==="
python -m pytest tests -q -p no:warnings

Write-Host "`n=== parameter counts ==="
python scripts/verify_param_count.py | Select-Object -Last 3

Write-Host "`n=== retrieval ==="
python scripts/verify_rag.py | Select-String "hit@1|recall@3"

Write-Host "`n=== checkpoints ==="
python scripts/verify_checkpoints.py checkpoints/micro-pretrain

Write-Host "`n=== training state ==="
if (Test-Path checkpoints/micro-pretrain\train_log.jsonl) {
  Get-Content checkpoints/micro-pretrain\train_log.jsonl |
    Where-Object { $_ -match '"eval"' } | Select-Object -Last 5
}

Write-Host "`n=== artefacts ==="
python -m veltron.cli info
```

---

## 15. Sign-off

Written 2026-10-01. State at that moment:

* **184 tests passing**
* **4B architecture verified, 4,026,765,312 parameters, weights blocked** (60 GiB vs 12 GiB)
* **`micro` pretraining at step 825/4000**, val loss 4.564, perplexity 96.0, ~3,600 tok/s
* **Retrieval at 100% hit@1 and 100% recall@3** on the 14-query support suite
* **Two tokenizers trained**, 0 round-trip failures
* **dataset-v1**: 19,485,297 train tokens, full licence provenance
* **13 experiment records**, including 2 documenting failed designs
* **~24 documents**, one model card, one final report
* **Not a git repository** — fix this first
---

# 2026-10-02 19:40 — documentation audit (done while mini trains)

No configuration was changed. Training PID 6084 continued throughout.

## Documentation errors found and corrected

**1. The VRAM budget was wrong by 2.7x, in a way that understated the machine.**

The docs claimed an "effective compute budget of ~3.62 GiB". That figure is the
**single-allocation ceiling**, not the total. Measured by `scripts/vram_shape.py`:

| | |
|---|---|
| Largest single allocation | 3.707 GiB |
| Total holdable across many allocations | **10.00 GiB** |
| What Windows reports in use | 1.11 GiB |

Proof it was wrong: the live `mini` run holds **6.98 GiB** of VRAM. A 3.62 GiB budget
cannot hold that. The 4B blocker arithmetic is corrected from "60 GiB vs 12 GiB, 5.0x
short" to **"60 GiB vs ~10 GiB usable, 6.0x short"**, which is a *worse* number for the
project and therefore the honest one.

**2. The tokenizers were described as byte-identical. They are not.**

`docs/tokenizer.md` and `FINAL_REPORT.md` both claimed the 32k and 64k vocabularies
produced identical compression. That came from reading one table twice. The authoritative
per-tokenizer reports show the 64k tokenizer is better on **every** slice:

| Slice | 32k | 64k | Gain |
|---|---:|---:|---:|
| English prose | 3.646 | 3.788 | +3.9% |
| **Serbian Cyrillic** | 2.819 | **3.484** | **+23.6%** |
| Serbian Latin | 2.177 | 2.346 | +7.8% |
| Python | 3.910 | 4.032 | +3.1% |
| Markdown | 3.696 | 3.826 | +3.5% |
| JSON | 3.416 | 3.597 | +5.3% |

This is now presented as a finding rather than a non-result: doubling the merge budget stops
the minority script being crowded out of the table. `experiments/EXP-0006.md` keeps the
original wrong conclusion visible with a correction above it, because deleting it would
hide the mistake.

**3. Test count** was 181 everywhere; the real number is **184**.

## Added

* `scripts/facts.py` — reads every documented figure from the artefact it came from, so
  prose cannot drift from reality. `--check` compares claims.
* `docs/RESULTS.md` — all measured numbers in one place, each with what it does *not*
  establish.
* `README.md` — restructured around the research problem rather than the component list.

## The most important correction for any reviewer

The live run reports **validation perplexity 3.0**, which looks excellent and is *not* a
quality claim. The validation split is **6 documents**; evaluation samples 8,176 of its
51,287 tokens; and validation loss has fallen *below* training loss (1.10 vs 3.85), which
means it is measuring an easy slice of the same Gutenberg pool rather than generalisation.
The validation curve is also non-monotone (3.11 → 3.40 → 3.75 → 1.61), the signature of a
small-sample estimate.

The trustworthy number is **train loss 6.15 → 3.85 with gradient norm never exceeding 2.74**.

`docs/RESULTS.md` §2.2 states this plainly instead of quoting 3.0. A reviewer who catches
an unexplained perplexity of 3.0 on 6 documents would discount the whole repository; better
to name it first.

## Still not done, and named as such

* No external baseline has ever been run.
* SFT and DPO have never been executed.
* Human evaluation has not been done.
* The validation set needs to be larger than 6 documents before any quality claim.

## Verification

184 tests pass; zero broken cross-links; `scripts/facts.py` reproduces every figure quoted
in `docs/RESULTS.md`.

---

# 2026-10-02 22:35 — mini run COMPLETE, automation retired

## The run finished

```
stop_reason : wall_clock_budget      (self-imposed, clean)
steps       : 4,134 of 6,000
tokens      : 33,865,728  (1.74 epochs of dataset-v1)
wall clock  : 8 h 1 m 18 s
train loss  : 6.15 -> 3.42
best val    : 1.1045 at step 3,250 (see the caveat below)
```

**Failure budget: all zero.** No OOM, no NaN loss, no gradient explosion, no corrupt
checkpoint, no I/O error. First completely clean run in the project.

### Checkpoints (all valid, 17.59 GiB, 44 GiB disk free)

| Step | Val loss |
|---:|---:|
| 3250 (best) | 1.1045 |
| 3500 | 1.1303 |
| 3750 | 1.1365 |
| 4000 | 1.1498 |
| 4134 (final) | 1.1788 |

## The validation number still is not a quality claim

Val loss 1.18 is **below** train loss 3.42, on a 6-document split sampling 8,176 tokens, and
the curve flattened from step 3250 to 4134 (1.10 -> 1.18) while train loss kept falling.
That is overfitting to a tiny evaluation set. `docs/RESULTS.md` §2.2 says so explicitly.

**Quote: train loss 6.15 -> 3.42, gradient norm never above 2.74.**

## Automation is now retired

* `VeltronLM Keepalive` scheduled task **removed** (`Unregister-ScheduledTask`). It will not
  restart anything tonight.
* `STOP_REQUEST`, `AUTORUN_PID`, `AUTORUN_INFO` cleared from `checkpoints/mini-pretrain`.
* No python training process is running. GPU is free.

To resume later: `scripts\windows\start_veltron_training.ps1` (resumes from step 4,134).

## What happens next, in priority order

1. **Build a real validation split.** Six documents is the single biggest weakness in the
   results. This is cheap and unblocks every other claim.
2. **Run SFT.** The model is still a base model; it will not chat. The 604-example dataset
   already exists at `datasets/sft/sft.jsonl`.
3. **Run one external baseline.** Converting "we built a system" into "we know where it
   sits" is the largest remaining gap for external review.
4. Rebuild the scheduled task only if another unattended run is wanted.

## Documentation state

Clean. `docs/RESULTS.md` holds the final numbers with the caveats attached;
`scripts/facts.py` re-derives every figure from its artefact; 184 tests pass; no broken
cross-links.
