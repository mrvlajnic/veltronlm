"""Reproducible compute benchmarks for this host.

Every number printed here is measured at run time. The matmul benchmark forces
synchronisation with a blocking device->host read on every iteration; an unsynchronised
variant reported 53 TFLOP/s on the same GPU where the true figure is 2.80, which is the
reason the synchronised path is the only one kept.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from veltron.model.config import REGISTRY
from veltron.utils.device import get_device, probe_all, synchronize
from veltron.utils.logging_utils import setup_logging

OUTPUT = Path("reports/compute_benchmark.json")


def bench_matmul(device, n: int, iters: int, dtype: torch.dtype) -> dict:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        (a @ b).cpu()          # blocking read == forced completion
    t0 = time.perf_counter()
    for _ in range(iters):
        (a @ b).cpu()
    dt = time.perf_counter() - t0
    return {
        "n": n,
        "dtype": str(dtype).replace("torch.", ""),
        "seconds": round(dt, 4),
        "tflops": round((2 * n**3 * iters) / dt / 1e12, 4),
    }


def train_step_throughput(model, device, seq_len: int, batch: int, steps: int,
                          dtype: torch.dtype) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, model.cfg.vocab_size, (batch, seq_len), device=device)
    for _ in range(2):
        out = model(ids, labels=ids)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        out = model(ids, labels=ids)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
    synchronize(device)
    dt = time.perf_counter() - t0
    tokens = batch * seq_len * steps
    return {
        "model": model.cfg.name,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seq_len": seq_len,
        "batch": batch,
        "steps": steps,
        "seconds": round(dt, 2),
        "tokens": tokens,
        "tokens_per_second": round(tokens / dt, 1),
        "seconds_per_step": round(dt / steps, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(OUTPUT))
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    setup_logging(level="WARNING")

    backends = [b.as_dict() for b in probe_all()]
    print("=" * 84)
    print("VELTRONLM COMPUTE BENCHMARK (all values measured on this host)")
    print("=" * 84)
    for b in backends:
        print(f"  backend {b['kind']:10} {b['name']:22} {b['total_memory_gb']:7.2f} GB  "
              f"fp16={b['supports_fp16']} bf16={b['supports_bf16']}")

    device = get_device(args.device)
    # torch reports DirectML as "privateuseone:0", so match on the torch device type
    # rather than on str(device) containing the human-readable backend name.
    probe = next(b for b in probe_all() if b.torch_device_type == device.type)
    info = probe.as_dict()

    print("\n--- matmul (synchronised) ---")
    print(f"{'device':10}{'dtype':9}{'n':>7}{'iters':>7}{'TFLOP/s':>11}{'seconds':>10}")
    matmuls = []
    specs = []
    for dt_name, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
        for n, iters in ((512, 20), (1024, 15), (2048, 8)):
            specs.append((n, iters, dt))
    cpu_device = torch.device("cpu")
    for n, iters, dt in specs:
        row = bench_matmul(device, n, iters, dt)
        matmuls.append({"device": info["kind"], **row})
        print(f"{info['kind']:10}{row['dtype']:9}{n:>7}{iters:>7}{row['tflops']:>11.4f}{row['seconds']:>10.4f}")
    for n, iters in ((1024, 10), (2048, 5)):
        row = bench_matmul(cpu_device, n, iters, torch.float32)
        matmuls.append({"device": "cpu", **row})
        print(f"{'cpu':10}{'fp32':9}{n:>7}{iters:>7}{row['tflops']:>11.4f}{row['seconds']:>10.4f}")

    peak = max((m["tflops"] for m in matmuls if m["device"] != "cpu"), default=0.0)
    cpu_peak = max((m["tflops"] for m in matmuls if m["device"] == "cpu"), default=0.0)
    print(f"\n  accelerator peak: {peak:.3f} TFLOP/s")
    print(f"  cpu peak:         {cpu_peak:.3f} TFLOP/s")
    print(f"  speedup:          {peak / max(cpu_peak, 1e-9):.1f}x")

    train_rows = []
    if not args.skip_train:
        print("\n--- end-to-end training step ---")
        from veltron.model.transformer import VeltronLM

        for key, seq_len, batch in (("nano", 512, 4), ("micro", 1024, 2)):
            if key not in REGISTRY:
                continue
            cfg = REGISTRY[key].config
            if cfg.vocab_size > 32768:
                continue
            model = VeltronLM(cfg).to(device)
            if device.type != "cpu":
                model = model.half()
            row = train_step_throughput(model, device, seq_len, batch, 5,
                                        torch.float16 if device.type != "cpu" else torch.float32)
            train_rows.append(row)
            print(f"  {row['model']:22} {row['parameters']:>13,} params  "
                  f"seq={seq_len} bs={batch}  {row['tokens_per_second']:>9.1f} tok/s  "
                  f"{row['seconds_per_step']:>7.2f} s/step")
            del model

    payload = {
        "backends": backends,
        "matmul": matmuls,
        "train_step": train_rows,
        "accelerator_peak_tflops": peak,
        "cpu_peak_tflops": cpu_peak,
        "speedup_vs_cpu": round(peak / max(cpu_peak, 1e-9), 2),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
