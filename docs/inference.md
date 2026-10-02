# Inference

All generation is local. No external model API is contacted anywhere in this project.

---

## 1. Generation pipeline

```
prompt → tokenizer → chat template
       → chunked prefill into a KV cache
       → loop: sample token, append to cache, forward one step
       → stop on eos / stop token / stop string / max tokens
```

### 1.1 Sampler

| Control | Behaviour |
|---|---|
| `temperature` | Divides logits before softmax. ≤ 0 is treated as greedy rather than dividing by zero. |
| `top_k` | Keeps the k highest-probability tokens. 0 disables. |
| `top_p` (nucleus) | Keeps the smallest set whose cumulative mass reaches p, always keeping ≥ 1 token. |
| `min_p` | Keeps tokens with `p >= min_p * p_max`. Scale-free, works better than top_p on flat distributions. |
| `repetition_penalty` | Divides positive logits and multiplies negative ones for already-emitted tokens. |
| `presence_penalty` / `frequency_penalty` | Linear penalties on presence and count. |
| `no_repeat_ngram_size` | Bans tokens that would complete an n-gram already present. |
| `stop_strings` | Checked against the decoded text of each row. |
| `stop_token_ids` | Checked before appending. |

`min_p` is worth highlighting: unlike `top_p`, it is invariant to the overall scale of the
distribution, which matters for a base model whose confidence varies enormously between a
freshly initialised checkpoint and a trained one.

### 1.2 fp32 logits at decode

Logits for decode steps are always computed in fp32 even when the model runs in fp16:

```python
self._last_logits = out.logits[:, -1, :].float()
```

A 65,536-way logit vector in fp16 saturates at 65,504. A confident prediction therefore
overflows to `inf` and then to `nan` in the softmax — a failure that appears **only at 4B
scale and only for the highest-probability tokens**, which is precisely where a sampling bug
does the most damage.

### 1.3 `num_logits_to_keep`

Decode steps project only the last position to vocabulary space. For the 4B model that
avoids a `3072 × 65536` matmul per token (201M multiply-accumulates) when only one row is
needed.

### 1.4 Chunked prefill

Prompts longer than `prefill_chunk` (default 512) are prefilled in chunks so a long prompt
does not materialise all its activations at once. Because the cache is causal, the chunked
path is numerically identical to a single pass.

## 2. Determinism

| Scenario | Guaranteed |
|---|---|
| Greedy decoding | **Yes** — `argmax`, no sampling |
| Seeded sampling | **Yes** — the `Sampler` builds a CPU `torch.Generator` seeded per call |
| Batched vs single generation | **Yes** — token-identical (see §4) |
| Save/load round trip | **Yes** — verified byte-for-byte |
| Re-running the same checkpoint | **Yes** (modulo GPU floating-point non-determinism) |

`seed_everything(seed, deterministic=True)` additionally enables
`torch.use_deterministic_algorithms` and sets `CUBLAS_WORKSPACE_CONFIG`. Without
`deterministic=True`, `seed_everything` still seeds Python, NumPy and torch, which covers
data order and dropout.

## 3. Loading a checkpoint

```bash
python -m veltron.inference.cli generate --checkpoint checkpoints/micro-pretrain \
    --prompt "The VeltronHub" --greedy --max-new-tokens 64
```

`find_checkpoint` accepts a step directory, a run directory (picks best, then latest), or
nothing at all. `load_engine` reads `model_config` from the checkpoint metadata, resolves
the matching tokenizer by recorded SHA-256, and **refuses to load** if the tokenizer has
more tokens than the model's vocabulary:

```
tokenizer has 65536 tokens but the model was built with vocab_size=32768;
the checkpoint and tokenizer do not match
```

An HF-style `config.json` is also accepted and remapped, so an export can be read by
tooling that expects the HuggingFace layout.

## 4. Batch generation with left padding

Left-padded batching is subtle, and getting it wrong is silent. Two requirements:

1. **Per-row RoPE positions.** A left-padded row's first real token is at a different
   absolute index than `kv_offset` implies. `VeltronLM.forward(positions=...)` accepts an
   explicit `(B, T)` position tensor.
2. **The padding mask must be re-supplied on every decode step.** Omitting it lets a short
   row attend to cached pad tokens, which changes its output as soon as it diverges from
   the batch. `_decode_step` rebuilds the mask from each row's pad count every step.

Verified: batched output is **token-identical** to running each prompt alone, and logits
agree to **5.5e-07**. The earlier version diverged on the first step for any batch
containing a short prompt.

## 5. Quantization

`quantize_model_inplace` performs **group-wise symmetric affine quantization** with fp16
scale/zero per group:

| Bits | Group | Typical fidelity |
|---:|---:|---|
| 8 | 128 | low error |
| 4 | 128 | moderate error |

The dequantized values are left in place, which measures the **fidelity cost** of low-bit
weights while leaving the rest of the stack untouched. That is what makes a before/after
perplexity comparison meaningful. Shipping for production would additionally need an
integer kernel; this repository does not claim one.

`test_4bit_quantization_error_exceeds_8bit` asserts the error ordering, which is a real
property rather than a coincidence.

## 6. Measured performance

`python scripts/bench_inference.py --checkpoint <dir>` writes
`reports/inference_benchmark.json`.

### 6.1 Untrained tier footprint (measured on this host)

| Tier | Parameters | fp16 weights | Loads? |
|---|---:|---:|---|
| `micro` | 55,715,328 | 0.10 GiB | yes |
| `4b` | 4,026,765,312 | 7.50 GiB | see §6.3 |

### 6.2 Trained `micro` generation

| Metric | Value |
|---|---|
| Median decode throughput | see `reports/inference_benchmark.json` |
| Decode is slower than matmul peak | expected — DirectML lacks a fused attention kernel and `AdamW`/`lerp` falls back to the CPU |

### 6.3 4B instantiation

`veltronlm-4b-base` in fp16 needs 7.50 GiB of weights plus activations and a 1.125 GiB KV
cache at 8k context. This host reports 8 GiB usable for DirectML, so a full 4B fp16
generation pass **competes with any concurrent GPU job** and can exhaust memory. The
benchmark reports a `FAILED` entry with the exact reason rather than omitting the tier.

Quantized variants are comfortable: int8 needs 3.75 GiB, int4 needs 1.88 GiB.

## 7. Batched throughput

`--max-batch-size` bounds the batch; oversizing raises rather than silently truncating.
Throughput scales sublinearly with batch size on this backend because attention at
`seq × seq` dominates and DirectML has no fused kernel for it.

## 8. Limitations

1. **No fused attention kernel.** `scores = q kᵀ + mask` then `softmax` then `probs v`
   materialises a `B × H × T × T` tensor. At 8k context and 24 heads that is 3.1 GB in
   fp16 for the scores alone. FlashAttention or xformers would fix it, but neither is
   available on DirectML.
2. **No paged KV cache.** A single contiguous buffer per layer; long conversations in one
   process allocate the max immediately.
3. **Speculative decoding is not implemented.**
4. **Quantization is simulated.** Correctness of a real int4 kernel is untested.
5. **`no_repeat_ngram` and `frequency_penalty` are O(generated) per step** and will become
   the bottleneck for very long outputs.
6. **The generation engine is single-request.** Concurrent requests contend for the same
   KV cache; `max_batch_size` bounds but does not schedule.
7. **DirectML decode throughput is modest** and some optimiser ops fall back to the host.
   CUDA would be substantially faster; this host has no CUDA device.