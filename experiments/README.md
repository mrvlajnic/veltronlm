# Experiment Log

Every experiment actually run in this repository. An experiment is recorded here when a
command produced output, regardless of outcome. Planned or imagined experiments appear in
`docs/roadmap.md` instead.

Format follows `EXPERIMENT_TEMPLATE.md`.

| ID | Date | Title | Outcome |
|---|---|---|---|
| [EXP-0001](EXP-0001.md) | 2026-09-30 | GPU backend provisioning and throughput measurement | Partial — see record |
| [EXP-0002](EXP-0002.md) | 2026-09-30 | Data source reachability probe | Pass |
| [EXP-0003](EXP-0003.md) | 2026-09-30 | Architecture correctness and causality verification | Pass |
| [EXP-0004](EXP-0004.md) | 2026-09-30 | 4B parameter accounting and feasibility | Verdict — 4B training BLOCKED |
| [EXP-0005](EXP-0005.md) | 2026-09-30 | Corpus acquisition and mixture | Pass, with a large documented skew |
| [EXP-0006](EXP-0006.md) | 2026-09-30 | Tokenizer training and compression measurement | Pass — with a documented saturation |
| [EXP-0007](EXP-0007.md) | 2026-09-30 | Pretraining canary calibration | Pass after two false failures |
| [EXP-0008](EXP-0008.md) | 2026-09-30 | Pretraining run — `veltronlm-micro` | In progress |
| [EXP-0009](EXP-0009.md) | 2026-09-30 | Retrieval hybrid tuning | Pass — hit@1 100% |
| [EXP-0010](EXP-0010.md) | 2026-09-30 | SFT dataset construction | Pass — boilerplate bug found and fixed |
| [EXP-0011](EXP-0011.md) | 2026-09-30 | Additive vs boolean attention masks (DirectML) | Pass — boolean masks rejected |
| [EXP-0012](EXP-0012.md) | 2026-09-30 | Chunked cross-entropy for memory bound | Pass |
| [EXP-0013](EXP-0013.md) | 2026-09-30 | Concurrent GPU workload interference | Measured degradation |

## Summary of findings

**What worked**

* The architecture is correct: causality exact to 0.0, KV cache within 9.2e-07 of a full
  forward, batched generation token-identical to single-prompt.
* Hybrid retrieval reaches 14/14 hit@1 once confidence is made absolute.
* A 55.7M-parameter model trains stably on a 12 GiB AMD GPU at ~3,600 tokens/s.
* DirectML is a viable GPU path on Windows for AMD hardware, with three documented
  limitations handled by design.

**What did not work, and why**

* **4B pretraining**: 60 GiB of optimizer state against 12 GiB of VRAM. A 5× shortfall with
  no single-GPU remedy.
* **Canary design**: two successive false failures before the test became meaningful. Both
  are recorded rather than deleted.
* **Tokenizer 64k**: saturated at 57,611 merges. 9.4 Mcharacters is not enough text.
* **Cross-lingual retrieval**: required a hand-written bilingual map; a zero-shot approach
  does not work.
* **4B instantiation under contention**: segfaulted while a training job held the GPU.

**Bugs found and fixed** (all regression-tested)

| Bug | Symptom | Root cause |
|---|---|---|
| Byte tokenizer mangles non-ASCII | Serbian decoded to `"    .  1  ."` | Latin-1 vocabulary instead of the GPT-2 byte↔unicode alphabet |
| Byte tokenizer drops spaces | `"TheVeltronHubX1"` | `add_tokens` bypasses pre-tokenization |
| Shard dtype mismatch | `IndexError: index 2172472 ... size 32768` | Index hardcoded `uint32`, data written `uint16` |
| KV cache reads uninitialised memory | Wrong prefill output | Length tracked internally instead of passed as an offset |
| Batched generation diverges | Batch ≠ solo output | Padding mask dropped on decode steps |
| `normalize` corrupts code | Python indents 8→2 spaces | Leading whitespace collapsed |
| MinHash irreproducible | Dedupe changed every run | Python `hash()` is randomised per process |
| Empty-matching regex | Every word split per character | Alternation with a `*` on an empty branch |
| Retrieval confidence always high | Escalation never fired | Rank-normalised score used as an absolute |
| Taxonomy augmentation manufactures evidence | Swallow question scored 0.725 | Augmented query fed the confidence calculation |
| Serialiser drops Cyrillic | `tokenize` returned `['24']` | ASCII-only `[a-z]` class; `re.IGNORECASE` does not widen it |
| `field()` in a plain class | `TypeError: 'Field' object does not support item assignment` | `dataclasses.field` outside a dataclass |
| Checkpoint stores train loss as `val_loss` | Best-checkpoint selection compared train vs train | Interval save had no evaluation |
| SFT summaries are Gutenberg boilerplate | Targets were `[Illustration]` and e-text headers | No boilerplate filtering in extraction |

## Experiments that were designed but not run

These scripts exist; the runs did not happen. No result is claimed for them.

* `scripts/exp_lr.py` — learning-rate sweep
* `scripts/exp_mixture.py` — dataset mixture sweep
* `scripts/exp_tokenizer.py` — vocabulary-size sweep

`docs/roadmap.md` lists them as open actions with the specific question each should answer.