"""Repair shard index files whose recorded dtype disagrees with the data on disk.

The index is authoritative for the loader, so a wrong dtype silently reinterprets every
token. The true element width is recoverable exactly from geometry: a shard of N tokens
occupying 2N bytes is uint16, 4N bytes is uint32.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def repair(split_dir: Path, prefix: str) -> dict:
    idx_path = split_dir / f"{prefix}-index.json"
    payload = json.loads(idx_path.read_text(encoding="utf-8"))
    claimed = payload.get("dtype")
    changed = []
    for shard in payload["shards"]:
        p = split_dir / shard["path"]
        n = int(shard["tokens"])
        if n == 0:
            continue
        width = p.stat().st_size / n
        true_dtype = "uint16" if width < 3 else "uint32"
        shard["bytes_on_disk"] = p.stat().st_size
        shard["dtype"] = true_dtype
        changed.append((shard["path"], true_dtype))
    widths = {d for _, d in changed}
    true_dtype = widths.pop() if len(widths) == 1 else "MIXED"
    payload["dtype"] = true_dtype
    payload["dtype_repaired"] = claimed != true_dtype
    payload["dtype_previously_claimed"] = claimed
    idx_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "index": str(idx_path),
        "was": claimed,
        "now": true_dtype,
        "repaired": claimed != true_dtype,
        "shards": len(changed),
    }


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/dataset-v1")
    for split in ("train", "val", "test"):
        d = root / split
        if not (d / f"{split}-index.json").exists():
            continue
        info = repair(d, split)
        print(f"{split:6} was={info['was']:8} now={info['now']:8} repaired={info['repaired']} "
              f"shards={info['shards']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
