"""The VeltronLM decoder-only Transformer.

Shape convention throughout: activations are ``(B, T, D)`` for the residual stream and
``(B, H, T, Dhd)`` inside attention. Grouped-query attention (GQA, Ainslie et al. 2023)
shares one key/value head across ``n_heads // n_kv_heads`` query heads, which cuts the
KV cache by 3x relative to multi-head attention at equal query-head count while keeping
most of the quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import (
    KVCache,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    causal_mask,
    padding_mask,
    rope_tables_for_positions,
)


@dataclass
class Output:
    """Result of a forward pass."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class Attention(nn.Module):
    """Grouped-query self-attention with optional QK normalization.

    QK-norm (per-head RMSNorm on queries and keys, as in ViT-22B and Gemma) removes the
    logit growth that otherwise forces the output projection's learning rate to be very
    small; without it, fp16 training diverges within a few hundred steps at the learning
    rates this project uses.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.group_size = cfg.gqa_group_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.use_qk_norm else nn.Identity()

        self.attn_dropout = cfg.attn_dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        kv_offset: int = 0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self._apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            if kv_offset > 0:
                k_prev, v_prev = kv_cache.read(self.layer_idx, kv_offset)
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)
            kv_cache.append(self.layer_idx, k[:, :, -T:], v[:, :, -T:], kv_offset)

        # (B, Hkv, T, Dhd) -> (B, Hkv, G, T, Dhd) -> (B, H, T, Dhd)
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # ``mask`` is additive (0 allowed / large-negative masked); see layers.causal_mask.
        if mask is not None:
            scores = scores + mask

        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        if self.attn_dropout > 0 and self.training:
            probs = F.dropout(probs, p=self.attn_dropout)

        ctx = torch.matmul(probs, v)  # (B, H, T, Dhd)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(ctx)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q * cos + self._rot(q) * sin
        k = k * cos + self._rot(k) * sin
        return q, k

    @staticmethod
    def _rot(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((-x2, x1), dim=-1)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = Attention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff, cfg.ffn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        kv_offset: int = 0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kv_cache, kv_offset, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class VeltronLM(nn.Module):
    """Decoder-only Transformer language model.

    Implements the standard causal next-token objective with an optional
    next-token-prediction loss so a single forward pass serves pretraining,
    instruction tuning, preference optimisation and evaluation.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rotary = RotaryEmbedding(
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len,
            base=cfg.rope_theta,
            scaling_factor=cfg.rope_scaling_factor,
            beta_fast=cfg.rope_scaling_beta_fast,
            beta_slow=cfg.rope_scaling_beta_slow,
            original_max_seq_len=cfg.rope_scaling_original_max_seq_len,
        )
        self.layers = nn.ModuleList(TransformerBlock(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)
        # Scaled init on residual projections: keeps the variance of the residual stream
        # from growing with depth, which is what makes deep pre-norm stacks trainable.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.initializer_range / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------ helpers
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def num_parameters_breakdown(self) -> dict[str, int]:
        """Group live parameter objects by subsystem, from real module types.

        This is deliberately independent of :meth:`ModelConfig.param_counts` so the two
        can be cross-checked in tests.
        """
        out = {
            "embedding": 0,
            "attention": 0,
            "mlp": 0,
            "norm": 0,
            "output_head": 0,
        }
        for name, p in self.named_parameters():
            module = self.get_submodule(name.rsplit(".", 1)[0])
            n = p.numel()
            if isinstance(module, nn.Embedding):
                out["embedding"] += n
            elif isinstance(module, nn.Linear):
                if ".self_attn." in name or name.startswith("layers"):
                    if ".self_attn." in name:
                        out["attention"] += n
                    else:
                        out["mlp"] += n
                else:
                    out["output_head"] += n
            else:
                out["norm"] += n
        return out

    def _rope_tables(self, T: int, device: torch.device, dtype: torch.dtype, offset: int):
        cos, sin = self.rotary._cos_sin(offset + T, device, dtype)
        return cos[offset : offset + T].unsqueeze(0).unsqueeze(0), sin[offset : offset + T].unsqueeze(0).unsqueeze(0)

    # ------------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        kv_offset: int = 0,
        attention_mask: torch.Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        num_logits_to_keep: int = 0,
        positions: torch.Tensor | None = None,
        loss_chunk_tokens: int = 0,
    ) -> Output:
        """Run the model.

        Args:
            input_ids: ``(B, T)`` token ids.
            labels: optional ``(B, T)`` targets; loss is computed internally.
            kv_cache: when supplied, keys/values are appended and the model becomes
                incremental for the last ``T`` tokens.
            kv_offset: absolute position of ``input_ids[0, 0]`` in the full sequence.
            attention_mask: ``(B, kv_len)`` of 1/0 for padding, where ``kv_len`` is
                ``kv_offset + T`` when a cache is in use.
            num_logits_to_keep: during decode, only project the final ``k`` positions to
                vocabulary space. Saves the vocab-sized matmul on every step.
            positions: optional ``(B, T)`` explicit RoPE positions. Required for
                left-padded batches, where each row's first real token sits at a
                different absolute index than ``kv_offset`` implies.
        """
        B, T = input_ids.shape
        h = self.embed_tokens(input_ids)

        # With a KV cache, keys span [0, kv_offset + T) while queries are the trailing
        # T positions, so the mask must be built against the full key length.
        kv_len = kv_offset + T

        if positions is None:
            cos, sin = self._rope_tables(T, h.device, h.dtype, kv_offset)
        else:
            cos, sin = rope_tables_for_positions(
                positions, self.rotary.inv_freq, h.dtype
            )

        mask: torch.Tensor
        if attention_mask is None:
            mask = causal_mask(T, kv_len, h.device, h.dtype, kv_offset=kv_offset)
        else:
            # Two independent additive masks, summed rather than combined with a boolean
            # OR: DirectML cannot materialise a computed 4-D boolean tensor, and addition
            # is both supported and cheaper.
            pad_kv_len = attention_mask.shape[1]
            mask = causal_mask(T, pad_kv_len, h.device, h.dtype, kv_offset=kv_offset)
            mask = mask + padding_mask(attention_mask, T, pad_kv_len, h.dtype)

        for layer in self.layers:
            h = layer(h, cos, sin, kv_cache, kv_offset, mask)

        h = self.norm(h)

        if num_logits_to_keep > 0:
            h = h[:, -num_logits_to_keep:, :]
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = self.compute_loss(
                logits, labels, ignore_index=ignore_index,
                reduction=reduction, chunk_tokens=loss_chunk_tokens,
            )

        return Output(logits=logits, loss=loss, hidden_states=h)

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
        reduction: str = "mean",
        chunk_tokens: int = 0,
    ) -> torch.Tensor:
        """Causal cross-entropy, optionally computed in token chunks.

        ``logits[:, :-1]`` predicts ``labels[:, 1:]``; no external shift is required so
        callers cannot get the classic off-by-one bug.

        Args:
            chunk_tokens: when > 0, the flattened tokens are processed in slices of this
                size and the losses are summed. This is what keeps memory bounded: the
                fp32 log-softmax buffer is ``chunk_tokens x vocab`` instead of
                ``batch*seq x vocab``. For a 4B model at 8k context the undivided version
                needs ~2 GiB per micro-batch for the logit tensor alone, plus an equal
                amount for the fp32 softmax and again for its backward -- which is the
                difference between fitting in 12 GB and not.
        """
        T = logits.shape[1]
        if labels.shape[1] != T:
            raise ValueError(f"labels seq len {labels.shape[1]} != logits seq len {T}")
        lg = logits[:, :-1, :]
        tg = labels[:, 1:]

        if chunk_tokens and chunk_tokens > 0:
            flat_logits = lg.reshape(-1, lg.shape[-1])
            flat_labels = tg.reshape(-1)
            total = flat_logits.new_zeros((), dtype=torch.float32)
            n_kept = 0
            for start in range(0, flat_logits.shape[0], chunk_tokens):
                end = min(flat_logits.shape[0], start + chunk_tokens)
                sl = flat_logits[start:end].float()
                st = flat_labels[start:end]
                loss_sum = F.cross_entropy(sl, st, ignore_index=ignore_index, reduction="sum")
                total = total + loss_sum
                n_kept += int((st != ignore_index).sum())
            if reduction == "sum":
                return total
            if n_kept == 0:
                return total * 0.0
            return total / n_kept

        flat_logits = lg.contiguous().view(-1, lg.shape[-1])
        flat_labels = tg.contiguous().view(-1)
        return F.cross_entropy(flat_logits.float(), flat_labels,
                               ignore_index=ignore_index, reduction=reduction)

    # ----------------------------------------------------------------- KV cache
    def init_kv_cache(
        self,
        batch: int = 1,
        max_seq_len: int | None = None,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> KVCache:
        cfg = self.cfg
        return KVCache(
            n_layers=cfg.n_layers,
            batch=batch,
            max_seq_len=max_seq_len or cfg.max_seq_len,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            dtype=dtype,
            device=device,
        )

    # ----------------------------------------------------------------- gradient
    def enable_input_require_grads(self) -> None:
        """Required when gradient checkpointing is combined with frozen embeddings."""
        self.embed_tokens.weight.requires_grad_(True)

    def gradient_checkpointing_enable(self) -> None:
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self._gradient_checkpointing = False


def build_model_on_meta(cfg: ModelConfig) -> VeltronLM:
    """Instantiate the architecture on ``meta`` -- exact shapes, zero bytes of memory.

    This is how the 4B configuration is verified without allocating 16GB of fp32
    weights or requiring the hardware that would be needed to train it.
    """
    with torch.device("meta"):
        return VeltronLM(cfg)


def count_parameters_via_meta(cfg: ModelConfig) -> int:
    """Ground-truth parameter count read off real ``Parameter`` objects."""
    with torch.device("meta"):
        model = VeltronLM(cfg)
        total = sum(p.numel() for p in model.parameters())
        del model
    return total


__all__ = [
    "VeltronLM",
    "Attention",
    "TransformerBlock",
    "Output",
    "build_model_on_meta",
    "count_parameters_via_meta",
]
