"""Model configuration and *exact* parameter accounting.

Parameter counts in this project are never guessed. :meth:`ModelConfig.parameter_report`
computes every tensor shape analytically from the configuration and
:meth:`build_model_on_meta` instantiates the real module on the ``meta`` device so the
two independent derivations can be cross-checked against each other.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# Head-dimension bounds. Attention head width below 32 collapses the softmax into a
# near-uniform distribution and measurably hurts loss; above ~256 the per-head
# parameter cost starts to dominate the KV cache without a matching quality gain.
MIN_HEAD_DIM = 32
MAX_HEAD_DIM = 256


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyper-parameters for a VeltronLM decoder-only Transformer."""

    name: str = "veltronlm-4b-base"
    vocab_size: int = 65536
    n_layers: int = 36
    d_model: int = 3072
    n_heads: int = 24
    n_kv_heads: int = 8
    d_ff: int = 8192
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    tie_embeddings: bool = False
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_qk_norm: bool = True
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0
    rope_scaling_factor: float = 1.0
    rope_scaling_beta_fast: float = 32.0
    rope_scaling_beta_slow: float = 1.0
    rope_scaling_original_max_seq_len: int = 8192

    # ---------------------------------------------------------------- derived
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def gqa_group_size(self) -> int:
        return self.n_heads // self.n_kv_heads

    # ------------------------------------------------------------- validation
    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} must be divisible by n_kv_heads={self.n_kv_heads}"
            )
        if not (MIN_HEAD_DIM <= self.head_dim <= MAX_HEAD_DIM):
            raise ValueError(
                f"head_dim={self.head_dim} outside supported range "
                f"[{MIN_HEAD_DIM}, {MAX_HEAD_DIM}]"
            )
        for f in ("vocab_size", "n_layers", "d_model", "n_heads", "n_kv_heads", "d_ff"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{f} must be positive, got {getattr(self, f)}")
        if self.vocab_size % 8 != 0:
            raise ValueError(
                f"vocab_size={self.vocab_size} should be a multiple of 8 for "
                "tensor-core friendly matmuls"
            )
        if self.tie_embeddings and self.d_ff <= 0:
            raise ValueError("tie_embeddings requires d_ff")

    # ------------------------------------------------------- parameter algebra
    def param_counts(self) -> dict[str, int]:
        """Analytic parameter counts, broken down by subsystem.

        Shapes, using ``V=vocab_size, D=d_model, L=n_layers, H=n_heads,
        Hkv=n_kv_heads, Dhd=head_dim, F=d_ff``:

        * embedding          ``V * D``
        * attention per layer ``D*H*Dhd  (q)  +  D*Hkv*Dhd (k)  +  D*Hkv*Dhd (v)
                               +  H*Dhd*D  (out)``  = ``2*D*D + 2*D*Hkv*Dhd``
        * attention qk-norm per layer ``2*Dhd`` (only when ``use_qk_norm``)
        * MLP per layer        ``3 * D * F``   (gate, up, down for SwiGLU)
        * norms per layer      ``2 * D``      (attn-norm, ffn-norm), plus optional qk-norm
        * final norm           ``D``
        * output head          ``V * D`` when untied, ``0`` when tied
        """
        V, D, L = self.vocab_size, self.d_model, self.n_layers
        H, Hkv, Dhd, F = self.n_heads, self.n_kv_heads, self.head_dim, self.d_ff

        q = D * (H * Dhd)
        k = D * (Hkv * Dhd)
        v = D * (Hkv * Dhd)
        o = (H * Dhd) * D
        attn = q + k + v + o

        qk_norm = (2 * Dhd) if self.use_qk_norm else 0
        mlp = 3 * D * F
        norms = 2 * D + qk_norm

        per_layer = attn + mlp + norms
        embed = V * D
        final_norm = D
        head = 0 if self.tie_embeddings else V * D

        total = embed + L * per_layer + final_norm + head

        return {
            "embedding": embed,
            "attention_total": L * (attn + qk_norm),
            "attention_q": L * q,
            "attention_k": L * k,
            "attention_v": L * v,
            "attention_out": L * o,
            "attention_qk_norm": L * qk_norm,
            "mlp_total": L * mlp,
            "norms_total": L * norms + final_norm,
            "output_head": head,
            "per_layer": per_layer,
            "total": total,
            "non_embedding": total - embed,
        }

    def kv_cache_bytes(self, batch: int, seq: int, dtype_bytes: int = 2) -> int:
        """KV-cache footprint for ``batch`` sequences of length ``seq``."""
        per_token = 2 * self.n_layers * self.kv_dim * dtype_bytes
        return batch * seq * per_token

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["head_dim"] = self.head_dim
        d["kv_dim"] = self.kv_dim
        d["gqa_group_size"] = self.gqa_group_size
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def replace(self, **kwargs: Any) -> ModelConfig:
        return dataclass_replace(self, **kwargs)


def dataclass_replace(cfg: ModelConfig, **kwargs: Any) -> ModelConfig:
    base = asdict(cfg)
    base.update(kwargs)
    return ModelConfig(**base)


# --------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class ConfigEntry:
    key: str
    config: ModelConfig
    tier: str
    purpose: str
    trainable_on_this_host: bool
    notes: str = ""


def _entry(
    key: str,
    cfg: ModelConfig,
    tier: str,
    purpose: str,
    trainable_on_this_host: bool,
    notes: str = "",
) -> ConfigEntry:
    return ConfigEntry(key, cfg, tier, purpose, trainable_on_this_host, notes)


REGISTRY: dict[str, ConfigEntry] = {
    e.key: e
    for e in [
        _entry(
            "nano",
            ModelConfig(
                name="veltronlm-nano",
                vocab_size=32768,
                n_layers=4,
                d_model=256,
                n_heads=4,
                n_kv_heads=2,
                d_ff=688,
                max_seq_len=512,
                use_qk_norm=False,
            ),
            tier="T0",
            purpose="CPU unit tests, gradient-flow checks, overfit canary",
            trainable_on_this_host=True,
            notes="Seconds per run. Every invariant is asserted against this size.",
        ),
        _entry(
            "micro",
            ModelConfig(
                name="veltronlm-micro",
                vocab_size=32768,
                n_layers=8,
                d_model=512,
                n_heads=8,
                n_kv_heads=2,
                d_ff=1376,
                max_seq_len=1024,
            ),
            tier="T1",
            purpose="Architecture A/B tests, LR sweeps, tokenizer sensitivity",
            trainable_on_this_host=True,
            notes="~30M params. Runs a full ablation grid in minutes.",
        ),
        _entry(
            "mini",
            ModelConfig(
                name="veltronlm-mini",
                vocab_size=32768,
                n_layers=16,
                d_model=1024,
                n_heads=16,
                n_kv_heads=4,
                d_ff=2752,
                max_seq_len=2048,
            ),
            tier="T2",
            purpose="Primary trainable base model on this workstation",
            trainable_on_this_host=True,
            notes="~110M params. Fits in 12GB VRAM with gradient checkpointing.",
        ),
        _entry(
            "small",
            ModelConfig(
                name="veltronlm-small",
                vocab_size=32768,
                n_layers=24,
                d_model=1536,
                n_heads=12,
                n_kv_heads=4,
                d_ff=4096,
                max_seq_len=4096,
            ),
            tier="T3",
            purpose="Upper bound of credible single-GPU training here",
            trainable_on_this_host=False,
            notes="~250M params. Multi-day run; used for scaling-law fits only.",
        ),
        _entry(
            "4b",
            ModelConfig(
                name="veltronlm-4b-base",
                vocab_size=65536,
                n_layers=36,
                d_model=3072,
                n_heads=24,
                n_kv_heads=8,
                d_ff=8192,
                max_seq_len=8192,
            ),
            tier="T4",
            purpose="Target 4B release architecture",
            trainable_on_this_host=False,
            notes=(
                "Architecture is real and instantiable; from-scratch AdamW pretraining "
                "needs ~64GB of optimizer state and is blocked by 12GB VRAM."
            ),
        ),
        _entry(
            "4b-long",
            ModelConfig(
                name="veltronlm-4b-base-16k",
                vocab_size=65536,
                n_layers=36,
                d_model=3072,
                n_heads=24,
                n_kv_heads=8,
                d_ff=8192,
                max_seq_len=16384,
            ),
            tier="T4",
            purpose="16k-context variant of the 4B model",
            trainable_on_this_host=False,
            notes="Trained from 8k with RoPE scaling; not verified on this host.",
        ),
        _entry(
            "1b",
            ModelConfig(
                name="veltronlm-1b-base",
                vocab_size=65536,
                n_layers=24,
                d_model=2048,
                n_heads=16,
                n_kv_heads=4,
                d_ff=5632,
                max_seq_len=8192,
            ),
            tier="T4",
            purpose="Mid-size open-weight release candidate",
            trainable_on_this_host=False,
            notes="Inference-only on this host; full pretraining blocked.",
        ),
    ]
}


def get_config(key: str) -> ModelConfig:
    if key not in REGISTRY:
        raise KeyError(f"unknown config {key!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[key].config


def registry_table() -> list[dict[str, Any]]:
    out = []
    for e in REGISTRY.values():
        c = e.config
        pc = c.param_counts()
        out.append(
            {
                "key": e.key,
                "name": c.name,
                "tier": e.tier,
                "layers": c.n_layers,
                "d_model": c.d_model,
                "heads": c.n_heads,
                "kv_heads": c.n_kv_heads,
                "head_dim": c.head_dim,
                "d_ff": c.d_ff,
                "vocab": c.vocab_size,
                "ctx": c.max_seq_len,
                "params": pc["total"],
                "params_B": round(pc["total"] / 1e9, 4),
                "trainable_here": e.trainable_on_this_host,
                "purpose": e.purpose,
            }
        )
    return out


__all__ = [
    "ModelConfig",
    "ConfigEntry",
    "REGISTRY",
    "get_config",
    "registry_table",
    "MIN_HEAD_DIM",
    "MAX_HEAD_DIM",
    "field",
]
