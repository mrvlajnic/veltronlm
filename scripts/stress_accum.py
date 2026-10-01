"""Stress-test gradient-accumulation settings for memory survival.

Symptom: `mini` OOMs after the canary passes, inside `cross_entropy`, during training.
The canary runs at 128 tokens with batch 7, which is a much smaller memory profile than
the real configuration, so it cannot catch this.

DirectML's allocator grows its heap and never shrinks it within a process, so the
question is not "does one micro-step fit" but "does micro-step N still fit". This runs a
realistic micro-step repeatedly and reports how many survive.

    python scripts/stress_accum.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

from veltron.model.config import get_config
from veltron.model.transformer import VeltronLM
from veltron.utils.device import get_device

OUTPUT = Path("reports/stress_accum.json")


def trial(tier: str, batch: int, seq: int, accum: int, chunk: int, micro_steps: int,
          device) -> dict:
    cfg = get_config(tier)
    model = VeltronLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    labels = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)

    completed = 0
    error = ""
    t0 = time.perf_counter()
    try:
        for _ in range(micro_steps):
            out = model(ids, labels=labels, loss_chunk_tokens=chunk)
            opt.zero_grad(set_to_none=True)
            (out.loss / accum).backward()
            completed += 1
    except RuntimeError as exc:
        error = str(exc)[:150]

    # One real optimiser step, which is where the CPU-fallback lerp cost lands.
    step_seconds = 0.0
    try:
        t1 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type != "cpu":
            torch.zeros(1, device=device).cpu()
        step_seconds = time.perf_counter() - t1
    except RuntimeError as exc:
        error = error or str(exc)[:150]

    elapsed = time.perf_counter() - t0
    del model, opt, ids, labels, out
    gc.collect()
    from veltron.utils.device import empty_cache

    empty_cache(device)

    return {
        "tier": tier, "batch": batch, "seq": seq, "accum": accum, "chunk": chunk,
        "micro_steps_requested": micro_steps,
        "micro_steps_completed": completed,
        "survived": completed >= micro_steps and not error,
        "error": error,
        "seconds": round(elapsed, 2),
        "micro_step_seconds": round(elapsed / max(completed, 1), 3),
        "optimizer_step_seconds": round(step_seconds, 3),
        "tokens_per_second": round(batch * seq * completed / elapsed, 1) if completed else 0.0,
        "tokens_per_optimizer_step": batch * seq * accum,
        "projected_optimizer_step_seconds": round(
            elapsed / max(completed, 1) * accum + step_seconds, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="mini")
    ap.add_argument("--micro-steps", type=int, default=40)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(OUTPUT))
    args = ap.parse_args()
    device = get_device(args.device)

    combos = [
        (2, 512, 32, 1024),   # what failed
        (2, 512, 32, 256),    # same, tiny chunks
        (2, 512, 8, 1024),    # less accumulation
        (2, 512, 8, 256),
        (1, 512, 16, 512),
        (2, 256, 16, 512),    # shorter sequences
    ]

    print("=" * 100)
    print(f"ACCUMULATION STRESS TEST  ({args.tier}, {args.micro_steps} micro-steps each)")
    print("=" * 100)
    print(f"{'b':>3}{'seq':>6}{'accum':>7}{'chunk':>7}{'survived':>10}"
          f"{'micro_s':>9}{'opt_s':>8}{'proj step':>11}{'tok/s':>10}  note")
    print("-" * 100)

    results = []
    for batch, seq, accum, chunk in combos:
        r = trial(args.tier, batch, seq, accum, chunk, args.micro_steps, device)
        results.append(r)
        note = r["error"][:60] if r["error"] else ""
        status = "OK" if r["survived"] else "FAIL@%d" % r["micro_steps_completed"]
        print(f"{batch:>3}{seq:>6}{accum:>7}{chunk:>7}{status:>10}"
              f"{r['micro_step_seconds']:>9.2f}{r['optimizer_step_seconds']:>8.2f}"
              f"{r['projected_optimizer_step_seconds']:>10.1f}s"
              f"{r['tokens_per_second']:>10,.0f}  {note}")

    print("\n--- recommended ---")
    ok = [r for r in results if r["survived"]]
    if not ok:
        print("  no combination survived; the model itself is too large for this card.")
    else:
        best = max(ok, key=lambda r: r["tokens_per_second"])
        print(f"  b={best['batch']} seq={best['seq']} accum={best['accum']} "
              f"chunk={best['chunk']}")
        print(f"  {best['tokens_per_second']:,.0f} tok/s, "
              f"{best['tokens_per_optimizer_step']:,} tokens per optimiser step, "
              f"~{best['projected_optimizer_step_seconds']:.0f} s/step")
        corpus = 19_485_297
        print(f"  one epoch of dataset-v1: {corpus / best['tokens_per_second'] / 3600:.1f} h")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())