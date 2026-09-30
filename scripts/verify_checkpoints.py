"""Audit every checkpoint in a run directory.

Reports validity, contents and problems per checkpoint so a damaged run can be diagnosed
without loading weights into memory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from veltron.training.checkpoint import CheckpointManager, verify_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit VeltronLM checkpoints")
    ap.add_argument("run", help="checkpoint run directory")
    ap.add_argument("--json", default=None, help="write a JSON report here")
    ap.add_argument("--skip-valid", action="store_true", help="only report problems")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_dir():
        print(f"error: {run} is not a directory", file=sys.stderr)
        return 2

    mgr = CheckpointManager(run)
    checkpoints = mgr.list_checkpoints()
    if not checkpoints:
        print(f"no step-* directories under {run}")
        return 1

    print("=" * 100)
    print(f"CHECKPOINT AUDIT: {run}")
    print("=" * 100)
    print(f"{'checkpoint':>20} {'valid':>6} {'MiB':>9} {'step':>8} {'tokens':>13} "
          f"{'train_loss':>11} {'val_loss':>10} {'hash':>18}")
    print("-" * 100)

    report = []
    bad = 0
    for path in checkpoints:
        info = verify_checkpoint(path)
        meta = info.get("meta", {})
        mib = info["bytes"] / 1024**2
        if not info["valid"]:
            bad += 1
        if args.skip_valid and info["valid"]:
            continue
        print(f"{path.name:>20} {str(info['valid']):>6} {mib:>9.1f} "
              f"{str(meta.get('step', '-')):>8} {str(meta.get('tokens_seen', '-')):>13} "
              f"{_fmt(meta.get('train_loss')):>11} {_fmt(meta.get('val_loss')):>10} "
              f"{str(meta.get('checkpoint_hash', '-'))[:16]:>18}")
        for err in info.get("errors", []):
            print(f"{'':>20}   problem: {err}")
        report.append(info)

    print("-" * 100)
    latest = mgr.latest_valid()
    best = mgr.best_valid()
    print(f"  checkpoints: {len(checkpoints)}   invalid: {bad}")
    print(f"  disk usage:  {mgr.disk_usage_bytes() / 1024**3:.2f} GiB")
    print(f"  latest valid: {latest.name if latest else 'NONE'}")
    print(f"  best valid:   {best.name if best else 'NONE'}")
    if latest is None:
        print("\n  No valid checkpoint: this run cannot be resumed.")
        return 1
    if bad:
        print(f"\n  {bad} checkpoint(s) are damaged. Delete them to clean up, or resume "
              f"from {latest.name}.")
    return 0


def _fmt(v) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
        return "-" if f != f else f"{f:.4f}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    sys.exit(main())
