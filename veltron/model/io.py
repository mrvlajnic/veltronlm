"""Weight init / export helpers: safetensors, HuggingFace-compatible layout, quantization."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import ModelConfig
from .transformer import VeltronLM

log = logging.getLogger(__name__)

SAFE_INDEX = "model.safetensors.index.json"
WEIGHTS_NAME = "model.safetensors"


@dataclass
class ExportResult:
    path: Path
    total_bytes: int
    n_tensors: int
    dtype: str


def _iter_state_dict_tensors(model: nn.Module):
    for name, tensor in model.state_dict().items():
        if tensor.numel() > 0:
            yield name, tensor


def count_parameters_exact(model: nn.Module) -> int:
    """Exact count over distinct parameter storages (handles weight tying)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        key = p.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        total += p.numel()
    return total


def module_dtype(model: nn.Module) -> torch.dtype:
    """The model's compute dtype, taken from its first parameter.

    ``nn.Module`` has no ``.dtype`` attribute, so callers must not assume one.
    """
    try:
        return next(model.parameters()).dtype
    except StopIteration:  # pragma: no cover - parameterless module
        return torch.get_default_dtype()


def save_safetensors(model: nn.Module, out_dir: str | Path, dtype: torch.dtype | None = None) -> ExportResult:
    """Write weights in safetensors, sharded at 2GB, with an index file."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dtype is not None:
        model = model.to(dtype)
    eff_dtype = dtype or module_dtype(model)
    state: dict[str, torch.Tensor] = {}
    for name, tensor in _iter_state_dict_tensors(model):
        state[name] = tensor.detach().contiguous().cpu()

    shard_bytes = 2 * 1024**3
    shards: list[dict[str, torch.Tensor]] = [{}]
    sizes = [0]
    for name, tensor in state.items():
        nb = tensor.numel() * tensor.element_size()
        if sizes[-1] + nb > shard_bytes and shards[-1]:
            shards.append({})
            sizes.append(0)
        shards[-1][name] = tensor
        sizes[-1] += nb

    total_bytes = 0
    n_tensors = 0
    weight_map: dict[str, str] = {}
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors" if len(shards) > 1 else WEIGHTS_NAME
        save_file(shard, str(out_dir / fname))
        for name, tensor in shard.items():
            weight_map[name] = fname
            total_bytes += tensor.numel() * tensor.element_size()
            n_tensors += 1

    if len(shards) > 1:
        (out_dir / SAFE_INDEX).write_text(
            json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2),
            encoding="utf-8",
        )

    log.info("saved %d tensors (%.2f GB) to %s", n_tensors, total_bytes / 1024**3, out_dir)
    return ExportResult(out_dir, total_bytes, n_tensors, str(eff_dtype).replace("torch.", ""))


def save_checkpoint(model: nn.Module, out_dir: str | Path, dtype: torch.dtype | None = None) -> ExportResult:
    out_dir = Path(out_dir)
    res = save_safetensors(model, out_dir, dtype=dtype)
    cfg = model.cfg
    cfg.save(out_dir / "config.json")
    return res


def load_safetensors(model: VeltronLM, path: str | Path) -> VeltronLM:
    """Load weights from a directory or a single safetensors file into ``model``."""
    from safetensors.torch import load_file

    path = Path(path)
    files: list[Path]
    if path.is_file():
        files = [path]
    else:
        idx = path / SAFE_INDEX
        if idx.exists():
            files = sorted({p.name for p in path.glob("*.safetensors")})
            files = [path / f for f in files]
        else:
            files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors found under {path}")

    state: dict[str, torch.Tensor] = {}
    for f in files:
        state.update(load_file(str(f)))

    # Tolerate a HuggingFace-style "model." prefix on every key.
    if state and all(k.startswith("model.") for k in state):
        state = {k[len("model.") :]: v for k, v in state.items()}

    own = model.state_dict()
    loaded, skipped = 0, []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(own, strict=True)
    if skipped:
        log.warning("skipped %d unmatched tensors (first 5): %s", len(skipped), skipped[:5])
    log.info("loaded %d/%d tensors into %s", loaded, len(state), model.cfg.name)
    return model


def write_hf_card(
    out_dir: str | Path,
    cfg: ModelConfig,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write config.json in the HF layout plus a minimal generation_config.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf = {
        "architectures": ["VeltronLMForCausalLM"],
        "model_type": "veltronlm",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.d_ff,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.max_seq_len,
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": cfg.tie_embeddings,
        "torch_dtype": "bfloat16",
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    if cfg.rope_scaling_factor > 1.0:
        hf["rope_scaling"] = {
            "type": "yarn",
            "factor": cfg.rope_scaling_factor,
            "beta_fast": cfg.rope_scaling_beta_fast,
            "beta_slow": cfg.rope_scaling_beta_slow,
            "original_max_position_embeddings": cfg.rope_scaling_original_max_seq_len,
        }
    (out_dir / "config.json").write_text(json.dumps(hf, indent=2), encoding="utf-8")

    gen = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.05,
        "do_sample": True,
    }
    (out_dir / "generation_config.json").write_text(json.dumps(gen, indent=2), encoding="utf-8")

    cfg.save(out_dir / "veltron_config.json")
    if extra:
        (out_dir / "veltron_export.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    return out_dir


# ------------------------------------------------------------------- quantization
@dataclass
class QuantStats:
    bits: int
    group_size: int
    n_tensors: int
    original_bytes: int
    quantized_bytes: int
    max_abs_err: float
    mean_abs_err: float
    rmse: float

    @property
    def compression(self) -> float:
        return self.original_bytes / max(self.quantized_bytes, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "n_tensors": self.n_tensors,
            "original_MB": round(self.original_bytes / 1024**2, 2),
            "quantized_MB": round(self.quantized_bytes / 1024**2, 2),
            "compression": round(self.compression, 3),
            "max_abs_err": self.max_abs_err,
            "mean_abs_err": self.mean_abs_err,
            "rmse": self.rmse,
        }


def quantize_tensor_groupwise(t: torch.Tensor, bits: int, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-wise symmetric affine quantization with fp16 scale/zero per group."""
    if bits not in (4, 8):
        raise ValueError("only 4-bit and 8-bit quantization are supported")
    orig_dtype = t.dtype
    flat = t.detach().float().reshape(-1)
    n_groups = (flat.numel() + group_size - 1) // group_size
    pad = n_groups * group_size - flat.numel()
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    g = flat.view(n_groups, group_size)

    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    amax = g.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    zero = torch.full_like(scale, qmin)
    q = torch.clamp(torch.round(g / scale), qmin, qmax).to(torch.int8)
    deq = (q.float() * scale).reshape(-1)[: t.numel()].view(t.shape).to(orig_dtype)
    meta = torch.cat([scale, zero], dim=1).to(torch.float16)  # (n_groups, 2)
    return deq, meta


def dequantize_tensor_groupwise(q: torch.Tensor, meta: torch.Tensor, group_size: int) -> torch.Tensor:
    n_groups = meta.shape[0]
    scale = meta[:, :1].float()
    zero = meta[:, 1:2].float()
    out = (q.float() * scale + zero).reshape(-1)[: q.numel()].view(q.shape)
    return out.to(q.dtype)


def quantize_model_inplace(
    model: nn.Module, bits: int = 8, group_size: int = 128, keep_layers: Iterable[str] = ()
) -> QuantStats:
    """Simulated (fake) weight quantization.

    Real deployment should ship integer kernels; this measures the *fidelity cost* of
    low-bit weights with the dequantized values left in place so the rest of the stack is
    untouched. That is what makes a before/after perplexity comparison meaningful.
    """
    keep = tuple(keep_layers)
    original = 0
    qbytes = 0
    errs: list[torch.Tensor] = []
    n = 0
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        if any(k in name for k in keep):
            continue
        w = module.weight.data
        original += w.numel() * w.element_size()
        deq, _meta = quantize_tensor_groupwise(w, bits, group_size)
        errs.append((deq.float() - w.float()).reshape(-1))
        module.weight.data = deq
        qbytes += w.numel() * (bits / 8) + (w.numel() / group_size) * 4
        n += 1
    allerr = torch.cat(errs) if errs else torch.zeros(1)
    return QuantStats(
        bits=bits,
        group_size=group_size,
        n_tensors=n,
        original_bytes=original,
        quantized_bytes=int(qbytes),
        max_abs_err=float(allerr.abs().max()),
        mean_abs_err=float(allerr.abs().mean()),
        rmse=float(allerr.pow(2).mean().sqrt()),
    )


def copy_model(src: VeltronLM) -> VeltronLM:
    clone = VeltronLM(src.cfg)
    clone.load_state_dict(src.state_dict())
    return clone


def dir_size(path: str | Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def sha256_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: str | Path, dst: str | Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)


__all__ = [
    "ExportResult",
    "QuantStats",
    "module_dtype",
    "save_safetensors",
    "save_checkpoint",
    "load_safetensors",
    "write_hf_card",
    "count_parameters_exact",
    "quantize_tensor_groupwise",
    "dequantize_tensor_groupwise",
    "quantize_model_inplace",
    "copy_model",
    "dir_size",
    "sha256_file",
    "copy_tree",
]
