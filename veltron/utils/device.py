"""Compute-device abstraction.

VeltronLM targets three execution backends in order of preference:

1. :class:`DirectMLDevice`  -- AMD/NVIDIA/Intel GPU through ``torch-directml``.
2. :class:`CudaDevice`       -- NVIDIA GPU through CUDA.
3. :class:`CpuDevice`       -- always available fallback.

The abstraction exists because ``torch-directml`` does not expose a device object with
the full ``torch.cuda`` API surface (``synchronize``, ``mem_get_info``, ``empty_cache``,
``current_stream``). Rather than sprinkling ``if device.type == ...`` through the
codebase, every backend is wrapped so the rest of VeltronLM can be written against a
single small interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Description of a usable compute device."""

    kind: str
    name: str
    total_memory_bytes: int
    supports_bf16: bool
    supports_fp16: bool
    supports_fp8: bool
    #: The string ``torch.device`` accepts. Differs from :attr:`kind` for DirectML, which
    #: torch exposes as the generic ``privateuseone`` type with index 0.
    torch_device_type: str = "cpu"

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_bytes / 1024**3

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "torch_device_type": self.torch_device_type,
            "name": self.name,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
            "supports_fp8": self.supports_fp8,
        }


def _pynvml_vram(kind: str) -> int:
    """Best-effort VRAM query. Falls back to 0 when the vendor library is missing."""
    try:
        if kind == "cuda":
            free, total = torch.cuda.mem_get_info()
            del free
            return int(total)
        if kind == "directml":
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            best = 0
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(h)
                if info.total > best:
                    best = info.total
            pynvml.nvmlShutdown()
            return int(best)
    except Exception as exc:  # pragma: no cover - vendor library dependent
        log.debug("VRAM query via %s failed: %s", kind, exc)
    return 0


def _directml_info() -> DeviceInfo | None:
    try:
        import torch_directml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        log.debug("torch_directml unavailable: %s", exc)
        return None
    try:
        dev = torch_directml.device()
        torch.zeros(1, device=dev).cpu()
    except Exception as exc:  # pragma: no cover
        log.warning("DirectML present but unusable: %s", exc)
        return None
    name = "DirectML device"
    vram = _pynvml_vram("directml")
    if vram == 0:
        vram = 8 * 1024**3
    return DeviceInfo(
        kind="directml",
        torch_device_type="privateuseone",
        name=name,
        total_memory_bytes=vram,
        # DirectML executes fp16 kernels through shader model 6.x packing. bf16 is
        # emulated and unsupported for matmul, so it is advertised as unavailable.
        supports_bf16=False,
        supports_fp16=True,
        supports_fp8=False,
    )


def _cuda_info() -> DeviceInfo | None:
    if not torch.cuda.is_available():
        return None
    major, _minor = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    del free
    return DeviceInfo(
        kind="cuda",
        torch_device_type="cuda",
        name=props.name,
        total_memory_bytes=int(total),
        supports_bf16=major >= 8,
        supports_fp16=True,
        supports_fp8=major >= 9,
    )


def _cpu_info() -> DeviceInfo:
    import os

    try:
        import psutil  # type: ignore[import-not-found]

        ram = int(psutil.virtual_memory().total)
    except Exception:  # pragma: no cover
        ram = 8 * 1024**3
    del os
    return DeviceInfo(
        kind="cpu",
        name=f"CPU ({torch.get_num_threads()} threads)",
        total_memory_bytes=ram,
        supports_bf16=False,
        supports_fp16=False,
        supports_fp8=False,
    )


def probe_all() -> list[DeviceInfo]:
    """Return every usable backend, most capable first."""
    found: list[DeviceInfo] = []
    cuda = _cuda_info()
    if cuda is not None:
        found.append(cuda)
    dml = _directml_info()
    if dml is not None:
        found.append(dml)
    found.append(_cpu_info())
    return found


def probe(requested: str = "auto") -> DeviceInfo:
    """Select a backend. ``requested`` may be ``auto``/``cuda``/``directml``/``cpu``."""
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return _cpu_info()
    if requested == "cuda":
        info = _cuda_info()
        if info is None:
            raise RuntimeError("CUDA requested but no CUDA device is available")
        return info
    if requested == "directml":
        info = _directml_info()
        if info is None:
            raise RuntimeError("DirectML requested but torch-directml is unavailable")
        return info
    if requested not in ("auto", ""):
        raise ValueError(f"unknown device request: {requested!r}")
    for candidate in probe_all():
        if candidate.kind != "cpu":
            return candidate
    return _cpu_info()


def get_device(requested: str = "auto") -> torch.device:
    """Return the ``torch.device`` for the selected backend.

    DirectML is reported by torch as ``privateuseone:0``; ``torch_directml.device()`` is
    consulted so the device index matches the one torch-directml actually bound.
    """
    info = probe(requested)
    if info.kind == "directml":
        try:
            import torch_directml  # type: ignore[import-not-found]

            return torch_directml.device()
        except Exception:
            return torch.device("privateuseone:0")
    return torch.device(info.torch_device_type)


def synchronize(device: torch.device) -> None:
    """Block until queued work on ``device`` has completed.

    ``torch-directml`` exposes no synchronize entry point, so a blocking device->host
    copy of a 1-element tensor is used. The copy is enqueued on the same queue and can
    only complete once prior queued work has completed, which makes it a valid fence.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type in ("privateuseone", "directml"):
        torch.zeros(1, device=device).cpu()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    """Release cached device memory. No-op for backends without an allocator cache."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device.type == "cpu":
        import gc

        gc.collect()


def autocast_dtype(info: DeviceInfo) -> torch.dtype | None:
    """Pick the autocast dtype to use on this backend, or ``None`` for full precision."""
    if info.kind == "cpu":
        return None
    if info.supports_bf16:
        return torch.bfloat16
    if info.supports_fp16:
        return torch.float16
    return None


def summary() -> dict[str, Any]:
    """Machine-readable snapshot of every backend, for docs and reports."""
    return {"backends": [b.as_dict() for b in probe_all()]}
