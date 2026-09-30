# HANDOFF — VeltronLM

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
5.0x shortfall. Chinchilla-optimal 4B training is a ~22-year single-GPU run at the measured
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

## 4. Live training run

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

### When it finishes

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
4. `python -m pytest tests -q` — 181 tests
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

* **181 tests passing**
* **4B architecture verified, 4,026,765,312 parameters, weights blocked** (60 GiB vs 12 GiB)
* **`micro` pretraining at step 825/4000**, val loss 4.564, perplexity 96.0, ~3,600 tok/s
* **Retrieval at 100% hit@1 and 100% recall@3** on the 14-query support suite
* **Two tokenizers trained**, 0 round-trip failures
* **dataset-v1**: 19,485,297 train tokens, full licence provenance
* **13 experiment records**, including 2 documenting failed designs
* **~24 documents**, one model card, one final report
* **Not a git repository** — fix this first