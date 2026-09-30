"""Core building blocks: RMSNorm, rotary position embedding, SwiGLU.

Everything here is written with plain tensor ops rather than fused vendor kernels
(``torch.nn.functional.scaled_dot_product_attention``, FlashAttention, xformers) because
the primary GPU backend on this host is DirectML, whose operator coverage is incomplete.
Using primitive ops means one implementation runs identically on CPU, CUDA and DirectML,
so measured numbers are comparable across backends.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich, 2019).

    Unlike LayerNorm this removes the mean-centering and bias subtraction, which is both
    cheaper and empirically better for decoder-only Transformers. The learned gain is
    applied in the input dtype after the variance reduction is computed in fp32, so fp16
    inputs do not lose precision in the sum of squares.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        inv_rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # The gain is cast to the input dtype too, otherwise an fp16 activation is silently
        # promoted back to fp32 by the multiply and the whole residual stream widens.
        gain = self.weight.to(dtype)
        return ((x32 * inv_rms).to(dtype) * gain).to(dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split ``x`` in half along the last dim and rotate: ``[a, b] -> [-b, a]``.

    Only used by the YaRN-style scaling path; the default Llama-style path uses the
    interleaved formulation below.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoFormer, Su et al. 2021) with optional YaRN scaling.

    The default path is the Llama formulation: position ``m`` rotates a pair of
    dimensions ``(2i, 2i+1)`` by angle ``m * theta_i`` where
    ``theta_i = base^(-2i/d)``. Because ``m * theta_i`` is added to the query and key,
    the dot product of position ``m`` and position ``n`` depends only on ``m - n``,
    which gives relative-position behaviour and lets the model extrapolate past the
    training window as long as the rotary frequencies stay well-conditioned.

    YaRN (Peng et al. 2023) is used when ``scaling_factor > 1``: it applies a
    wavelength-based blend of interpolation and NTK-by-parts attention scaling, which
    preserves high-frequency detail that plain linear interpolation destroys.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        original_max_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.scaling_factor = float(scaling_factor)
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.original_max_seq_len = original_max_seq_len or max_seq_len

        inv_freq = self._build_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._scaled_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_inv_freq(self) -> torch.Tensor:
        idx = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (idx / self.head_dim))
        if self.scaling_factor > 1.0:
            inv_freq = self._yarn_rescale(inv_freq)
        return inv_freq

    def _yarn_rescale(self, inv_freq: torch.Tensor) -> torch.Tensor:
        """NTK-by-parts + interpolation blend used by YaRN."""
        factor = self.scaling_factor
        low_freq_factor = (
            self.beta_fast
            + self.beta_slow
            - (2 * math.pi) * self.original_max_seq_len / (self.beta_fast / math.log(factor))
        )
        low_freq_wavelen = 2 * math.pi / low_freq_factor
        wavelen = 2 * math.pi / inv_freq
        inv_freq_extrap = inv_freq / factor
        smooth = (self.original_max_seq_len / wavelen - self.beta_slow) / (
            self.beta_fast - self.beta_slow
        )
        smoothed = (1 - smooth) * inv_freq_extrap + smooth * inv_freq
        # Frequencies with wavelength longer than low_freq_wavelen are interpolated;
        # shorter ones keep the original (high-frequency) rate.
        is_low_freq = wavelen > low_freq_wavelen
        return torch.where(is_low_freq, smoothed, inv_freq)

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        key = seq_len
        cached = self._scaled_cache.get(key)
        if cached is not None and cached[0].device == device and cached[0].dtype == dtype:
            return cached
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        self._scaled_cache[key] = (cos, sin)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor | None = None,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query/key tensors shaped ``(B, H, T, D)``.

        ``positions`` may be an explicit index tensor (used when decoding with a KV
        cache, where the new token sits at absolute position ``offset + T - 1``).
        """
        B, H, T, D = q.shape
        if positions is None:
            cos, sin = self._cos_sin(offset + T, q.device, q.dtype)
            # cos/sin are (L, D) for L = offset + T; slice the live window and add the
            # batch/head broadcast axes: (1, 1, T, D)
            cos = cos[offset : offset + T].unsqueeze(0).unsqueeze(0)
            sin = sin[offset : offset + T].unsqueeze(0).unsqueeze(0)
        else:
            freqs = positions.to(device=q.device, dtype=torch.float32)[:, None] * self.inv_freq.to(
                device=q.device, dtype=torch.float32
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(q.dtype).unsqueeze(1)  # (T, 1, D) -> broadcast over B
            sin = emb.sin().to(q.dtype).unsqueeze(1)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class SwiGLU(nn.Module):
    """Gated feed-forward network with SiLU activation (Shazeer, 2020).

    ``down(silu(gate(x)) * up(x))``. The gate and up projections are kept separate
    rather than fused into one ``3D`` matrix because DirectML does not implement the
    grouped matmul that fusing them into a single batched call requires.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class KVCache:
    """Fixed-size ring-free KV cache for single- and multi-sequence decoding.

    One cache instance owns the whole model: every layer writes into its own slice, so
    allocation happens once per request instead of once per layer. Growth is amortised by
    doubling, and ``seq_len`` is tracked without any host synchronization.
    """

    def __init__(
        self,
        n_layers: int,
        batch: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        self.n_layers = n_layers
        self.batch = batch
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.k = [
            torch.empty((batch, n_kv_heads, max_seq_len, head_dim), dtype=dtype, device=self.device)
            for _ in range(n_layers)
        ]
        self.v = [
            torch.empty((batch, n_kv_heads, max_seq_len, head_dim), dtype=dtype, device=self.device)
            for _ in range(n_layers)
        ]
        self.length = 0
        self._prefilled = 0

    @property
    def seq_len(self) -> int:
        return self.length

    def reset(self) -> None:
        self.length = 0

    def grow_to(self, needed: int) -> None:
        """Double the cache until ``needed`` positions fit.

        Reallocation is proportional to cache size rather than to the number of decode
        steps, so a long generation pays O(log n) copies instead of O(n).
        """
        if needed <= self.max_seq_len:
            return
        new_cap = self.max_seq_len
        while new_cap < needed:
            new_cap *= 2
        new_k = [
            torch.empty((self.batch, self.n_kv_heads, new_cap, self.head_dim), dtype=self.dtype, device=self.device)
            for _ in range(self.n_layers)
        ]
        new_v = [
            torch.empty((self.batch, self.n_kv_heads, new_cap, self.head_dim), dtype=self.dtype, device=self.device)
            for _ in range(self.n_layers)
        ]
        n = self.length
        for i in range(self.n_layers):
            if n:
                new_k[i][:, :, :n].copy_(self.k[i][:, :, :n])
                new_v[i][:, :, :n].copy_(self.v[i][:, :, :n])
        self.k, self.v = new_k, new_v
        self.max_seq_len = new_cap

    def read(self, layer: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the first ``length`` cached positions for ``layer``."""
        return self.k[layer][:, :, :length], self.v[layer][:, :, :length]

    def append(
        self,
        layer: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        offset: int,
    ) -> int:
        """Write ``k_new``/``v_new`` at absolute position ``offset``; return the new length.

        ``offset`` is passed explicitly rather than tracked internally because the model
        owns the position counter. Reading and writing through an explicit offset is what
        keeps multi-layer prefill correct: during prefill every layer writes its own slot
        at offset 0, and no layer can accidentally read another layer's uninitialised
        buffer.
        """
        end = offset + k_new.shape[2]
        self.grow_to(end)
        self.k[layer][:, :, offset:end].copy_(k_new)
        self.v[layer][:, :, offset:end].copy_(v_new)
        self.length = end
        return end

    def copy_from(self, other: KVCache) -> None:
        """Fork this cache, used for n-best sampling and beam search."""
        self.grow_to(other.length)
        for i in range(self.n_layers):
            n = other.length
            self.k[i][:, :, :n].copy_(other.k[i][:, :, :n])
            self.v[i][:, :, :n].copy_(other.v[i][:, :, :n])
        self.length = other.length


def rope_tables_for_positions(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(cos, sin)`` of shape ``(B, 1, T, D)`` for explicit position indices.

    Used for left-padded batches, where each row's first real token sits at a different
    absolute index. A single shared table cannot express that, because rotary frequencies
    depend on the absolute position rather than the offset within the padded row.
    """
    inv = inv_freq.to(device=positions.device, dtype=torch.float32)
    freqs = positions.to(dtype=torch.float32)[:, :, None] * inv[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(1)
    sin = emb.sin().to(dtype).unsqueeze(1)
    return cos, sin


def causal_mask(
    seq_q: int,
    seq_k: int,
    device: torch.device,
    dtype: torch.dtype,
    kv_offset: int = 0,
    min_value: float | None = None,
) -> torch.Tensor:
    """Additive attention mask of shape ``(1, 1, seq_q, seq_k)``.

    Returns ``0.0`` where attention is allowed and a large negative number where it is
    masked, so the caller simply adds it to the scores.

    An additive float mask is used rather than a boolean mask with ``masked_fill`` because
    the boolean path requires DirectML to materialise a computed 4-D boolean tensor, which
    its shader compiler rejects for some mask shapes. Addition is supported everywhere, is
    one kernel fewer, and gives identical numerics because the masked entries collapse to
    the same effective value inside the fp32 softmax.

    ``kv_offset`` is the absolute position of the first query, which is nonzero during
    incremental decoding with a KV cache.
    """
    neg = torch.finfo(torch.float32).min if min_value is None else min_value
    q_idx = torch.arange(seq_q, device=device).unsqueeze(1) + kv_offset
    k_idx = torch.arange(seq_k, device=device).unsqueeze(0)
    allowed = k_idx <= q_idx
    mask = torch.zeros((seq_q, seq_k), dtype=torch.float32, device=device)
    mask.masked_fill_(k_idx > q_idx, neg)
    return mask.to(dtype).view(1, 1, seq_q, seq_k)


def padding_mask(
    attention_mask: torch.Tensor,
    seq_q: int,
    seq_k: int,
    dtype: torch.dtype,
    kv_offset: int = 0,
    min_value: float | None = None,
) -> torch.Tensor:
    """Additive padding mask built from a ``(B, S)`` 0/1 mask.

    Shape ``(B, 1, seq_q, seq_k)``. ``attention_mask`` spans the *keys*, i.e. the cached
    prefix plus the current chunk, which is why the width is ``seq_k`` rather than
    ``seq_q``.
    """
    neg = torch.finfo(torch.float32).min if min_value is None else min_value
    pad = attention_mask.to(torch.float32).to(dtype)  # (B, S)
    add = (1.0 - pad) * neg                       # 0 for kept keys, neg for padded keys
    return add[:, None, None, :].to(dtype)


__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "KVCache",
    "causal_mask",
    "padding_mask",
    "rope_tables_for_positions",
]
