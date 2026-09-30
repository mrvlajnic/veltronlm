"""Global determinism control.

Reproducibility is not just "seed everything": torch's own kernels are non-deterministic
by default on CUDA, and thread-count changes alter float reduction order. These helpers
make all of it explicit so a run can be replayed exactly.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int = 1234, deterministic: bool = False, num_threads: int | None = None) -> dict[str, Any]:
    """Seed Python, NumPy and torch; optionally force deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    info: dict[str, Any] = {"seed": seed, "deterministic": deterministic}

    if num_threads is not None:
        torch.set_num_threads(num_threads)
    info["num_threads"] = torch.get_num_threads()

    if deterministic:
        # cuBLAS requires this env var before the handle is created for deterministic
        # GEMM; warn rather than silently produce irreproducible numbers.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            info["torch_deterministic"] = True
        except Exception as exc:  # pragma: no cover
            info["torch_deterministic"] = False
            info["torch_deterministic_error"] = str(exc)
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
    else:
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    return info


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


__all__ = ["seed_everything", "make_generator"]
