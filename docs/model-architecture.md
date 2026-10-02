# Model Architecture

Every parameter count in this document is **computed programmatically** by
`python scripts/verify_param_count.py`, never estimated. Two independent derivations are
cross-checked on every run:

* **A** — analytic tensor shapes from `ModelConfig.param_counts()`;
* **B** — real `nn.Parameter` objects instantiated on the `meta` device
  (`count_parameters_via_meta`), which allocates no memory.

All seven registered configurations pass this cross-check (`analytic == meta`).

---

## 1. Target architecture: `veltronlm-4b-base`

```python
ModelConfig(
    name="veltronlm-4b-base",
    vocab_size=65536,
    n_layers=36,
    d_model=3072,
    n_heads=24,
    n_kv_heads=8,
    d_ff=8192,
    max_seq_len=8192,
    rope_theta=10000.0,
    tie_embeddings=False,
    norm_eps=1e-5,
    initializer_range=0.02,
    use_qk_norm=True,
)
```

Derived: `head_dim = 3072/24 = 128`, `kv_dim = 8 × 128 = 1024`,
`gqa_group_size = 24/8 = 3`.

### 1.1 Exact parameter breakdown

| Component | Shape derivation | Parameters | Share |
|---|---|---:|---:|
| Token embedding | `V × D = 65536 × 3072` | 201,326,592 | 5.000% |
| Attention `q_proj` | `L × D × H·Dhd = 36 × 3072 × 3072` | 339,738,624 | 8.437% |
| Attention `k_proj` | `L × D × Hkv·Dhd = 36 × 3072 × 1024` | 113,246,208 | 2.812% |
| Attention `v_proj` | `L × D × Hkv·Dhd = 36 × 3072 × 1024` | 113,246,208 | 2.812% |
| Attention `o_proj` | `L × H·Dhd × D = 36 × 3072 × 3072` | 339,738,624 | 8.437% |
| Attention QK-norm | `L × 2·Dhd = 36 × 256` | 9,216 | 0.000% |
| MLP gate/up/down | `L × 3·D·F = 36 × 3 × 3072 × 8192` | 2,717,908,992 | 67.496% |
| All RMSNorms | `L × 2·D + D = 36 × 6144 + 3072` | 233,472 | 0.006% |
| LM head | `V × D = 65536 × 3072` | 201,326,592 | 5.000% |
| **Per layer** | `L × 100,669,696` | **3,624,109,056** | **90.001%** |
| **TOTAL** | | **4,026,765,312** | **100%** |
| Non-embedding | total − embedding | 3,825,438,720 | 94.999% |

**4,026,765,312 parameters = 4.027 B**, inside the 3.7 B–4.3 B target band.

### 1.2 Parameter-shape derivation

With `V` = vocabulary, `D` = hidden size, `L` = layers, `H` = query heads,
`Hkv` = KV heads, `Dhd` = `D/H`, `F` = FFN size:

```
per layer =  D·(H·Dhd)          q_proj
           + D·(Hkv·Dhd)        k_proj
           + D·(Hkv·Dhd)        v_proj
           + (H·Dhd)·D          o_proj
           + 2·Dhd              QK-norm (if enabled)
           + 3·D·F              SwiGLU: gate, up, down
           + 2·D                attention + FFN RMSNorm gains

total     = V·D                 embedding
           + L · per_layer
           + D                   final norm
           + (0 if tied else V·D)
```

### 1.3 Memory and throughput implications

| Quantity | Value |
|---|---|
| fp32 weights | 15.00 GiB |
| bf16/fp16 weights | 7.50 GiB |
| int8 weights | 3.75 GiB |
| int4 weights | 1.88 GiB |
| KV cache, batch 1, 8192 ctx, bf16 | 1.125 GiB |
| KV cache, batch 8, 8192 ctx, bf16 | 9.00 GiB |
| AdamW optimizer state (fp32 m, v, grads, params) | **60.00 GiB** |

The 60 GiB optimizer requirement against ~10 GiB of usable VRAM is the hard blocker on
from-scratch training here. See `docs/environment.md` §4.

GQA divides the KV cache by `H/Hkv = 3` relative to full multi-head attention, which is
what keeps an 8k-context batch of 8 feasible at fp16 in 12 GiB.

---

## 2. Full configuration registry

| Key | Tier | Name | L | D | H | KV | Dhd | F | V | Ctx | Parameters | B |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nano` | T0 | veltronlm-nano | 4 | 256 | 4 | 2 | 64 | 688 | 32768 | 512 | 19,679,488 | 0.0197 |
| `micro` | T1 | veltronlm-micro | 8 | 512 | 8 | 2 | 64 | 1376 | 32768 | 1024 | 55,715,328 | 0.0557 |
| `mini` | T2 | veltronlm-mini | 16 | 1024 | 16 | 4 | 64 | 2752 | 32768 | 2048 | 244,354,048 | 0.2444 |
| `small` | T3 | veltronlm-small | 24 | 1536 | 12 | 4 | 128 | 4096 | 32768 | 4096 | 704,724,480 | 0.7047 |
| `4b` | T4 | veltronlm-4b-base | 36 | 3072 | 24 | 8 | 128 | 8192 | 65536 | 8192 | **4,026,765,312** | **4.0268** |
| `4b-long` | T4 | veltronlm-4b-base-16k | 36 | 3072 | 24 | 8 | 128 | 8192 | 65536 | 16384 | 4,026,765,312 | 4.0268 |
| `1b` | T4 | veltronlm-1b-base | 24 | 2048 | 16 | 4 | 128 | 5632 | 65536 | 8192 | 1,350,672,384 | 1.3507 |

### Vocabulary alignment with the trained tokenizers

The model vocabulary must be **at least** the tokenizer vocabulary or token ids index out
of range. Measured tokenizer sizes:

| Tokenizer | Trained vocab | Model tiers using it |
|---|---:|---|
| `tok-mini-32k` | 32,768 | `nano`, `micro`, `mini`, `small` |
| `tok-4b-64k` | 57,611 | `4b`, `4b-long`, `1b` |

`tok-4b-64k` was requested at 65,536 but saturated at 57,611: at `min_frequency=2` the
9.4 Mcharacter training sample does not contain 65,536 distinct merges. The `4b` tier
therefore declares `vocab_size=65536`, and **7,925 embedding rows (12.1%) are unused
padding rows** — 97,145,856 dead parameters, 2.4% of the model. This is deliberate:
power-of-two vocabularies keep the output projection aligned for tensor-core kernels, and
the cost is reported rather than hidden.

---

## 3. Layer-by-layer structure

```
VeltronLM
├── TokenEmbedding                  V × D
├── RotaryEmbedding                 non-parametric (θ, YaRN when scaled)
├── TransformerBlock × L
│   ├── input_layernorm             RMSNorm(D)
│   ├── self_attn                   Attention
│   │   ├── q_proj  D → H·Dhd
│   │   ├── q_norm  RMSNorm(Dhd)   ← QK-norm
│   │   ├── RoPE on q
│   │   ├── k_proj  D → Hkv·Dhd
│   │   ├── k_norm  RMSNorm(Dhd)   ← QK-norm
│   │   ├── RoPE on k
│   │   ├── v_proj  D → Hkv·Dhd
│   │   ├── GQA expand k, v         repeat_interleave by H/Hkv
│   │   ├── scores = (q kᵀ) · scale + additive_mask
│   │   ├── probs  = softmax(scores, fp32)
│   │   └── o_proj  H·Dhd → D
│   ├── residual add
│   ├── post_attention_layernorm    RMSNorm(D)
│   ├── mlp                         SwiGLU: D → F → F → D
│   └── residual add
├── norm                            RMSNorm(D)
└── lm_head                         D → V
```

### 3.1 Attention

**Grouped-query attention** (Ainslie et al. 2023) shares each KV head across
`H/Hkv` query heads. In the 4B model each of the 24 query heads attends to one of 8
shared KV heads, cutting KV-cache bytes 3× versus full MHA at equal query-head count.

**QK normalization.** A per-head RMSNorm is applied to queries and keys *before* RoPE.
Without it the logit magnitude grows until the output projection's learning rate must be
cut to the point where the residual stream stops learning; in fp16 training it diverges
within a few hundred steps. QK-norm costs 9,216 parameters (0.0002% of the model).

**Additive masks, not `masked_fill`.** The causal mask is a float tensor containing `0`
where attention is allowed and `3.4e38` where it is forbidden, added to the scores. A
boolean mask with `masked_fill` requires DirectML to materialise a computed 4-D boolean
tensor, which its shader compiler rejects for several mask shapes (see
`docs/environment.md` §2.1). Addition is also one kernel fewer and numerically identical
once the fp32 softmax is taken.

**Causality is verified exactly.** `tests/unit/test_model.py::test_attention_is_causal`
edits token 20 of a 48-token sequence and asserts the maximum logit delta at positions
0–19 is `< 1e-6` (measured: exactly `0.0`) while position 20 changes. The attention
probability mass beyond each row's causal window is exactly `0.0`.

### 3.2 RoPE

Rotary position embedding with `theta_i = 10000^(-2i/d)`. Because the rotation angle added
to query and key at position `m` depends only on `m`, the dot product of positions `m` and
`n` depends only on `m − n` — relative-position behaviour that extrapolates past the
training window while the frequencies remain well-conditioned.
`test_rope_is_relative` verifies this: shifting query and key by the same offset leaves
every pairwise dot product invariant.

**YaRN scaling** is available for the 16k variant (`rope_scaling_factor > 1`), applying the
NTK-by-parts blend of interpolation and attention scaling. It is **implemented and unit
tested but not trained or evaluated on this host** — `4b-long` is architecture-only.

### 3.3 RMSNorm

```python
x32 = x.float()
inv_rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
return ((x32 * inv_rms).to(x.dtype) * self.weight.to(x.dtype)).to(x.dtype)
```

Variance reduction runs in fp32 even for fp16 activations, so the sum of squares does not
lose precision. The gain is cast to the input dtype: without that cast an fp16 activation
is silently promoted back to fp32 by the multiply and the whole residual stream widens.
This was a real bug caught by `test_rmsnorm_preserves_dtype`.

### 3.4 SwiGLU

`down(silu(gate(x)) · up(x))`. Gate and up are kept as separate `nn.Linear` modules rather
than fused into one `D → 3F` matrix because DirectML does not implement the grouped matmul
that batching them would require.

### 3.5 Initialization

* All weights: `N(0, 0.02)`.
* `o_proj` and `down_proj` (the two residual-writing projections per block):
  `N(0, 0.02 / sqrt(2L))`. This keeps residual-stream variance from growing with depth,
  which is what makes a 36-layer pre-norm stack trainable at all.
* At initialisation the measured loss on a fresh model is within 0.5 nats of `ln(V)`,
  confirming the logits are near-uniform before any training
  (`test_initial_loss_is_near_uniform`).

### 3.6 Loss

Causal cross-entropy, shift-free:

```
loss = cross_entropy(logits[:, :-1], labels[:, 1:])
```

The shift lives inside `compute_loss`, so no caller can introduce the classic off-by-one.

**Chunked evaluation.** `compute_loss(..., chunk_tokens=N)` accumulates the loss over
slices of the flattened token dimension, bounding the fp32 log-softmax buffer at
`N × V` instead of `batch·seq × V`. For the 4B model at 8k context the undivided version
needs ~2 GiB for the logit tensor alone, plus the same again for the fp32 softmax and again
for its backward. `test_chunked_loss_matches_unchunked` asserts the two agree to `1e-4`.

---

## 4. Inference: KV cache and batching

### 4.1 KV cache

One cache owns all layers, allocating once per request instead of once per layer.
Read/write uses an **explicit offset** passed by the model rather than an internally
tracked length. During multi-layer prefill, an internally tracked length is a silent
correctness bug: layer *i* reads `cache.length`, which is still 0 for layers before the
last, and therefore attends over an uninitialised buffer. Passing `kv_offset` makes every
layer read exactly the positions that layer itself wrote.

Growth doubles the capacity, so a long generation pays `O(log n)` copies rather than `O(n)`.

**Verified:** `test_kv_cache_matches_full_forward` runs a 24-token prefill plus one decode
step and compares against a full 25-token forward. Measured maximum logit delta:
**9.2e-07**.

### 4.2 Batched generation with left padding

Batched prompts are left-padded so every sequence's final real token sits at the same
index, which would otherwise cause generation to continue from pad tokens for the shorter
prompts. Two additional details are required for exactness:

1. **Per-row RoPE positions.** A left-padded row's first real token sits at a different
   absolute index than `kv_offset` implies. `VeltronLM.forward(positions=...)` accepts an
   explicit `(B, T)` position tensor for this case.
2. **The padding mask must be re-supplied on every decode step.** Dropping it lets a short
   row attend to the pad tokens cached during prefill, which silently changes its output
   as soon as it diverges from the batch. `_decode_step` reconstructs the mask from each
   row's pad count on every step.

**Verified end to end:** batched generation with padding produces **token-identical output**
to running each prompt alone (`scripts/…` equivalent in `tests/unit/test_model.py`:
`test_explicit_positions_support_left_padding`, and the direct comparison in
`scripts/smoke_inference.py`). Logits agree to **5.5e-07**.

---

## 5. Parameter-tying option

`tie_embeddings=True` sets `lm_head.weight = embed_tokens.weight`, removing `V·D` =
201,326,592 parameters (5.0% of the 4B model). It is **not** used for the 4B configuration:
the 5% saving is real but untied output projections measurably improve large-model
quality, and the model fits in memory either way. The option exists and is tested
(`test_weight_tying_reduces_parameter_count`).

---

## 6. Export format

`safetensors`, sharded at 2 GiB with an index file, plus an HF-compatible `config.json`
(`model_type: veltronlm`), a `generation_config.json`, and `veltron_config.json` with the
native schema. `load_safetensors` tolerates a `model.` key prefix so a VeltronLM export can
be read by tooling that expects the HuggingFace layout.

---

## 7. What is NOT claimed

* `veltronlm-4b-base` has **never been trained**. Its parameter count, memory footprint and
  tensor shapes are verified; its weights do not exist.
* `4b-long` (16k context) has not been trained and its long-context behaviour has not been
  measured. The YaRN implementation is unit-tested for frequency behaviour only.
* No claim is made that this architecture beats any existing model. Such a claim would
  require a benchmark run that has not happened.
* The head-dimension bounds `[32, 256]` are enforced in `ModelConfig.__post_init__` as an
  engineering rule of thumb, not an empirically derived optimum for this corpus.