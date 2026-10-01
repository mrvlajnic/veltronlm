"""Measure one training configuration in a FRESH process.

DirectML's allocator grows its heap and does not shrink it within a process. Measuring
several configurations in one interpreter therefore contaminates every result after the
first: memory allocated by trial 1 stays reserved for trial 2, which then reports an OOM
that has nothing to do with its own size.

That produced two wrong conclusions earlier in this project:
  * "b=1 seq=2048 OOMs for the 117M tier"  -- harness residue, not a hardware limit
  * "the usable VRAM budget is 3.62 GiB"    -- that is the single-allocation cap, not
                                              the total; 10 GiB is reachable in smaller
                                              tensors

So: one config per process. Invoked by scripts/measure_one.py.

    python scripts/measure_one.py --tier 117m --batch 2 --seq 1024 --steps 3
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from veltron.model.config import get_config
from veltron.model.transformer import VeltronLM
from veltron.utils.device import get_device, probe, synchronize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--accum", type=int, default=1)
    args = ap.parse_args()

    device = get_device(args.device)
    cfg = get_config(args.tier)
    n_params = cfg.param_counts()["total"]

    model = VeltronLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)
    labels = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)

    times: list[float] = []
    oom = ""
    for i in range(args.steps + args.warmup):
        t0 = time.perf_counter()
        try:
            for _ in range(args.accum):
                out = model(ids, labels=labels, loss_chunk_tokens=4096)
                opt.zero_grad(set_to_none=True)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        except RuntimeError as exc:
            oom = str(exc)[:220]
            break
        synchronize(device)
        dt = time.perf_counter() - t0
        if i >= args.warmup:
            times.append(dt)

    result = {
        "tier": args.tier,
        "parameters": n_params,
        "batch": args.batch,
        "seq": args.seq,
        "accum": args.accum,
        "device": probe(args.device).kind,
    }
    if oom or not times:
        result["status"] = "OOM"
        result["reason"] = oom or "no successful steps"
    else:
        med = statistics.median(times)
        tps = args.batch * args.seq * args.accum / med
        result.update({
            "status": "OK",
            "seconds_per_step": round(med, 3),
            "tokens_per_second": round(tps, 1),
            "tokens_per_step": args.batch * args.seq * args.accum,
            "effective_tflops": round(6 * n_params * tps / 1e12, 4),
            "gpu_engine_utilisation": round(6 * n_params * tps / 2.8e12, 4),
        })

    print(json.dumps(result))
    del model, opt, ids, labels
    gc.collect()
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())