"""Train VeltronLM: full CLI for pretraining, fine-tuning, alignment and evaluation.

Examples::

    python -m veltron.train --config configs/pretrain_micro.yaml
    python -m veltron.train --model mini --seq-len 1024 --max-steps 2000
    python -m veltron.train --list-configs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from veltron.model.config import REGISTRY, registry_table
from veltron.training.trainer import TrainConfig, Trainer
from veltron.utils.config_utils import load_config
from veltron.utils.device import probe_all, summary
from veltron.utils.logging_utils import setup_logging


def print_configs() -> None:
    rows = registry_table()
    print(f"{'key':8}{'tier':6}{'name':26}{'layers':>7}{'d_model':>8}{'ctx':>7}"
          f"{'vocab':>8}{'params':>16}{'B':>9}  trainable_here")
    print("-" * 108)
    for r in rows:
        print(f"{r['key']:8}{r['tier']:6}{r['name']:26}{r['layers']:>7}{r['d_model']:>8}"
              f"{r['ctx']:>7}{r['vocab']:>8}{r['params']:>16,}{r['params_B']:>9}  {r['trainable_here']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m veltron.train")
    ap.add_argument("--config", default=None, help="YAML/JSON config file")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="config override as key.path=value (repeatable)")
    ap.add_argument("--list-configs", action="store_true")
    ap.add_argument("--devices", action="store_true", help="list compute backends and exit")
    ap.add_argument("--model", default=None, help="registry key, e.g. nano/micro/mini/4b")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--dataset-version", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--schedule", default=None)
    ap.add_argument("--precision", default=None, choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None, help="auto | none | path")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--eval-interval", type=int, default=None)
    ap.add_argument("--save-interval", type=int, default=None)
    ap.add_argument("--log-interval", type=int, default=None)
    ap.add_argument("--no-canary", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)

    if args.list_configs:
        print_configs()
        return 0
    if args.devices:
        print(json.dumps(summary(), indent=2))
        for b in probe_all():
            print(f"  {b.kind:10} {b.name:30} {b.total_memory_gb:6.2f} GB  "
                  f"fp16={b.supports_fp16} bf16={b.supports_bf16}")
        return 0

    setup_logging(level="INFO")

    raw: dict = {}
    if args.config:
        cfg_obj = load_config(args.config, args.overrides)
        raw = dict(cfg_obj)
        train_section = cfg_obj.get("train", {}) or {}
        raw.update(train_section)
    else:
        raw = {}
        if args.overrides:
            tmp = load_config(None)
            tmp.apply_overrides(args.overrides)
            raw = dict(tmp)

    cli_map = {
        "model_config": "model", "dataset_root": "dataset", "dataset_version": "dataset_version",
        "tokenizer_dir": "tokenizer", "output_dir": "out", "run_name": "run_name",
        "seq_len": "seq_len", "micro_batch_size": "batch_size",
        "grad_accum_steps": "grad_accum", "max_steps": "max_steps", "lr": "lr",
        "warmup_steps": "warmup", "schedule": "schedule", "precision": "precision",
        "device": "device", "seed": "seed", "resume": "resume", "max_hours": "max_hours",
        "eval_interval": "eval_interval", "save_interval": "save_interval",
        "log_interval": "log_interval",
    }
    for field, argname in cli_map.items():
        val = getattr(args, argname, None)
        if val is not None:
            raw[field] = val
    if args.no_canary:
        raw["run_canary_first"] = False
    if args.deterministic:
        raw["deterministic"] = True
    if args.threads is not None:
        raw["num_threads"] = args.threads

    known = set(TrainConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        print(f"warning: ignoring unknown config keys: {unknown}", file=sys.stderr)
    tcfg = TrainConfig(**{k: v for k, v in raw.items() if k in known})

    print("=" * 78)
    print("VELTRONLM TRAINING")
    print("=" * 78)
    print(json.dumps(tcfg.as_dict(), indent=2, default=str))
    print()

    if tcfg.model_config not in REGISTRY:
        print(f"error: unknown model key {tcfg.model_config!r}; "
              f"available: {sorted(REGISTRY)}", file=sys.stderr)
        return 2
    if not Path(tcfg.dataset_root).exists():
        print(f"error: dataset {tcfg.dataset_root} not found; "
              f"run scripts/build_dataset.py first", file=sys.stderr)
        return 2
    if not Path(tcfg.tokenizer_dir, "tokenizer.json").exists():
        print(f"error: tokenizer not found at {tcfg.tokenizer_dir}", file=sys.stderr)
        return 2

    trainer = Trainer(tcfg)
    summary_out = trainer.train()
    print("\n" + json.dumps(summary_out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
