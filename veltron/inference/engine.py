"""Model loading for inference and serving.

One entry point, :func:`load_engine`, that resolves a checkpoint directory into a live
model + tokenizer + generator. Every consumer (CLI, API, chatbot, evaluation) goes through
it, so a checkpoint is loaded exactly one way and a mismatch between config, weights and
tokenizer is impossible to introduce by accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..model.config import ModelConfig
from ..model.io import load_safetensors
from ..model.transformer import VeltronLM
from ..tokenizer.trainer import VeltronTokenizer
from ..utils.device import get_device, probe
from ..utils.logging_utils import get_logger
from .generator import GenerationConfig, Generator

log = get_logger(__name__)


@dataclass
class EngineInfo:
    model_name: str
    checkpoint: str
    tokenizer: str
    device: str
    dtype: str
    parameters: int
    parameters_billions: float
    context_length: int
    vocab_size: int
    step: int | None = None
    dataset_version: str = ""
    tokenizer_sha256: str = ""
    quantized_bits: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "checkpoint": self.checkpoint,
            "tokenizer": self.tokenizer,
            "device": self.device,
            "dtype": self.dtype,
            "parameters": self.parameters,
            "parameters_billions": round(self.parameters_billions, 4),
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "step": self.step,
            "dataset_version": self.dataset_version,
            "tokenizer_sha256": self.tokenizer_sha256,
            "quantized_bits": self.quantized_bits,
        }


@dataclass
class Engine:
    """A ready-to-serve model."""

    model: VeltronLM
    tokenizer: VeltronTokenizer
    generator: Generator
    info: EngineInfo
    device: torch.device

    def generate(self, prompt: str, cfg: GenerationConfig | None = None) -> Any:
        return self.generator.generate(prompt, cfg)

    def generate_chat(self, messages: list[dict[str, str]], cfg: GenerationConfig | None = None) -> Any:
        return self.generator.generate_chat(messages, cfg)

    def stream(self, prompt: str, cfg: GenerationConfig | None = None) -> Any:
        return self.generator.stream(prompt, cfg)


def _read_config(checkpoint: Path) -> ModelConfig:
    for name in ("veltron_config.json", "config.json"):
        p = checkpoint / name
        if not p.exists():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        known = set(ModelConfig.__dataclass_fields__)
        if "model_type" in raw and raw["model_type"] == "veltronlm":
            # HuggingFace-style layout: remap the HF names onto our own fields.
            raw = {
                "name": raw.get("name", "veltronlm"),
                "vocab_size": raw["vocab_size"],
                "n_layers": raw["num_hidden_layers"],
                "d_model": raw["hidden_size"],
                "n_heads": raw["num_attention_heads"],
                "n_kv_heads": raw["num_key_value_heads"],
                "d_ff": raw["intermediate_size"],
                "max_seq_len": raw["max_position_embeddings"],
                "rope_theta": raw.get("rope_theta", 10000.0),
                "tie_embeddings": raw.get("tie_word_embeddings", False),
                "norm_eps": raw.get("rms_norm_eps", 1e-5),
            }
        return ModelConfig.from_dict({k: v for k, v in raw.items() if k in known})
    raise FileNotFoundError(
        f"no model config in {checkpoint}; expected veltron_config.json or config.json"
    )


def find_checkpoint(path: str | Path) -> Path | None:
    """Resolve a run directory, best checkpoint, or explicit step directory to weights."""
    from ..training.checkpoint import CheckpointManager

    p = Path(path)
    if p.is_dir() and any(p.glob("*.safetensors")):
        return p
    if p.is_dir():
        mgr = CheckpointManager(p)
        best = mgr.best_valid() or mgr.latest_valid()
        if best:
            return best
        # Fall back to the highest-numbered directory that has weights at all.
        for cand in sorted(p.glob("step-*"), reverse=True):
            if any(cand.glob("*.safetensors")):
                return cand
    return None


def load_engine(
    checkpoint: str | Path,
    tokenizer_dir: str | Path | None = None,
    device: str = "auto",
    dtype: str = "auto",
    quantize_bits: int | None = None,
    quantize_group_size: int = 128,
) -> Engine:
    """Load a checkpoint into a runnable :class:`Engine`.

    Args:
        checkpoint: checkpoint directory, run directory, or best-checkpoint parent.
        tokenizer_dir: overrides the tokenizer recorded in the run config.
        dtype: ``auto`` | ``fp32`` | ``fp16`` | ``bf16``. ``auto`` uses fp16 on GPU when
            the backend supports it, and fp32 on CPU.
        quantize_bits: simulate 4- or 8-bit group-wise weight quantization in place.
    """
    ckpt = find_checkpoint(checkpoint)
    if ckpt is None:
        raise FileNotFoundError(f"no checkpoint weights found under {checkpoint}")

    meta: dict[str, Any] = {}
    meta_path = ckpt / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_meta = meta.get("model_config") or {}

    if model_meta:
        cfg = ModelConfig.from_dict(
            {k: v for k, v in model_meta.items() if k in ModelConfig.__dataclass_fields__}
        )
    else:
        cfg = _read_config(ckpt)

    tok_path = tokenizer_dir or _tokenizer_from_metadata(meta) or "models/tok-mini-32k"
    tokenizer = VeltronTokenizer.load(tok_path)

    if tokenizer.get_vocab_size() > cfg.vocab_size:
        raise ValueError(
            f"tokenizer has {tokenizer.get_vocab_size()} tokens but the model was built with "
            f"vocab_size={cfg.vocab_size}; the checkpoint and tokenizer do not match"
        )

    info = probe(device)
    dev = get_device(device)
    if dtype == "auto":
        if dev.type == "cpu":
            torch_dtype = torch.float32
        elif info.supports_bf16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    elif dtype in ("fp16", "float16"):
        torch_dtype = torch.float16
    elif dtype in ("bf16", "bfloat16"):
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    log.info("loading %s from %s on %s as %s", cfg.name, ckpt.name, dev, torch_dtype)
    model = VeltronLM(cfg)
    load_safetensors(model, ckpt)
    model.to(dev)
    model.to(torch_dtype)
    model.eval()

    if quantize_bits:
        from ..model.io import quantize_model_inplace

        stats = quantize_model_inplace(model, bits=quantize_bits, group_size=quantize_group_size)
        log.info("weight quantization: %s", json.dumps(stats.as_dict()))
        model.to(dev)
        model.eval()

    generator = Generator(model, tokenizer, device=dev, dtype=torch_dtype)
    n = sum(p.numel() for p in model.parameters())

    engine_info = EngineInfo(
        model_name=cfg.name,
        checkpoint=str(ckpt),
        tokenizer=str(tok_path),
        device=info.kind,
        dtype=str(torch_dtype).replace("torch.", ""),
        parameters=n,
        parameters_billions=n / 1e9,
        context_length=cfg.max_seq_len,
        vocab_size=cfg.vocab_size,
        step=meta.get("step"),
        dataset_version=meta.get("dataset_version", ""),
        tokenizer_sha256=tokenizer.sha256,
        quantized_bits=quantize_bits,
    )
    return Engine(model=model, tokenizer=tokenizer, generator=generator,
                  info=engine_info, device=dev)


def _tokenizer_from_metadata(meta: dict[str, Any]) -> str | None:
    """Recover the tokenizer path recorded when the checkpoint was written."""
    tcfg = (meta.get("trainer_config") or {}).get("tokenizer_dir")
    if tcfg and Path(tcfg, "tokenizer.json").exists():
        return tcfg
    sha = meta.get("tokenizer_sha256")
    if sha:
        for candidate in sorted(Path("models").glob("*/tokenizer_meta.json")):
            try:
                m = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if m.get("sha256") == sha:
                return str(candidate.parent)
    return None


__all__ = ["Engine", "EngineInfo", "load_engine", "find_checkpoint"]
