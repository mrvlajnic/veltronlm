"""Report VeltronLM training state as JSON.

One source of truth for the Windows automation scripts, so PowerShell never re-implements
checkpoint discovery. It reuses the repository's own
:class:`~veltron.training.checkpoint.CheckpointManager`, which is what the trainer itself
uses to decide what is resumable.

Deliberately does **not** check whether the training process is running: process detection
is a Windows concern and is done in PowerShell with CIM. This reports only what is on disk.

    python scripts/veltron_state.py --run checkpoints/mini-pretrain

Output keys:
    run, exists, latest_valid, latest_valid_step, best, all_checkpoints,
    invalid, tokens_seen, train_loss, val_loss, val_perplexity, model,
    parameters, dataset_version, tokenizer_version, disk_gib, disk_free_gib,
    log_path, stop_file, summary
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def collect(run_dir: Path) -> dict[str, Any]:
    from veltron.training.checkpoint import CheckpointManager

    state: dict[str, Any] = {
        "run": str(run_dir),
        "exists": run_dir.is_dir(),
        "latest_valid": None,
        "latest_valid_step": None,
        "best": None,
        "all_checkpoints": [],
        "invalid": [],
        "tokens_seen": None,
        "train_loss": None,
        "val_loss": None,
        "val_perplexity": None,
        "parameters": None,
        "model": None,
        "dataset_version": None,
        "tokenizer_version": None,
        "disk_gib": 0.0,
        "disk_free_gib": None,
        "log_path": str(run_dir / "train_log.jsonl") if run_dir.is_dir() else None,
        "summary_path": str(run_dir / "summary.json") if (run_dir / "summary.json").exists() else None,
    }

    if not state["exists"]:
        return state

    usage = shutil.disk_usage(str(run_dir))
    state["disk_gib"] = round(usage.used / 1024**3, 2)
    state["disk_free_gib"] = round(usage.free / 1024**3, 2)

    mgr = CheckpointManager(run_dir)

    for path in mgr.list_checkpoints():
        meta = mgr.read_metadata(path) or {}
        valid = mgr.is_valid(path)
        row = {
            "name": path.name,
            "step": meta.get("step"),
            "tokens_seen": meta.get("tokens_seen"),
            "train_loss": meta.get("train_loss"),
            "val_loss": meta.get("val_loss"),
            "val_perplexity": meta.get("val_perplexity"),
            "valid": valid,
            "bytes": sum(f.stat().st_size for f in path.glob("*") if f.is_file()),
            "mtime": path.stat().st_mtime,
        }
        state["all_checkpoints"].append(row)
        if not valid:
            state["invalid"].append(path.name)

    latest = mgr.latest_valid()
    if latest is not None:
        meta = mgr.read_metadata(latest) or {}
        state["latest_valid"] = str(latest)
        state["latest_valid_step"] = meta.get("step")
        state["tokens_seen"] = meta.get("tokens_seen")
        state["train_loss"] = meta.get("train_loss")
        state["val_loss"] = meta.get("val_loss")
        state["val_perplexity"] = meta.get("val_perplexity")
        state["model"] = (meta.get("model_config") or {}).get("name")
        state["parameters"] = (meta.get("model_config") or {}).get("_params")
        state["dataset_version"] = meta.get("dataset_version")
        state["tokenizer_version"] = meta.get("tokenizer_version")

    best = mgr.best_valid()
    if best is not None:
        state["best"] = str(best)

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            state["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state["summary"] = {"error": "summary.json is not valid JSON"}

    if state["parameters"] is None and state["model"]:
        from veltron.model.config import REGISTRY

        for entry in REGISTRY.values():
            if entry.config.name == state["model"]:
                state["parameters"] = entry.config.param_counts()["total"]
                break

    return state


def main() -> int:
    ap = argparse.ArgumentParser(description="VeltronLM training state (JSON)")
    ap.add_argument("--run", default="checkpoints/mini-pretrain")
    ap.add_argument("--compact", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = (Path.cwd() / run).resolve()

    state = collect(run)
    print(json.dumps(state, indent=None if args.compact else 2, default=str))
    # Exit 0 whenever state is readable, even if nothing is resumable yet: "no checkpoint"
    # is a valid answer, not an error. Callers inspect `latest_valid`.
    return 0


if __name__ == "__main__":
    sys.exit(main())