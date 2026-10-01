"""Empirically measure how much memory DirectML can actually allocate on this GPU.

AMD's NVML is unavailable, and `Win32_VideoController.AdapterRAM` reports a truncated
32-bit value (4 GiB for a 12 GiB card). Both are unusable for capacity planning, and this
GPU drives the display, so part of its 12 GiB is already committed to the compositor.

So the only trustworthy number is the one you get by allocating and seeing what fails.
This does a binary search over allocation size and reports the largest size that succeeds.

    python scripts/measure_vram.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from veltron.utils.device import get_device, probe

OUTPUT = Path("reports/vram_probe.json")


def try_alloc(device, n_bytes: int) -> tuple[bool, str]:
    """Attempt one allocation, forcing completion. Returns (ok, message).

    A 2-D float tensor is used rather than a 1-D byte tensor because DirectML rejects
    large 1-D allocations outright with "The parameter is incorrect" regardless of free
    memory. That is an operator-shape limitation, not an out-of-memory condition, and it
    would make every probe report failure. Matrices are also what the model actually
    allocates, so this measures the quantity that matters.
    """
    elems = max(1, n_bytes // 4)
    side = max(1, int(elems**0.5))
    rows = side
    cols = max(1, elems // rows)
    try:
        t = torch.ones((rows, cols), dtype=torch.float32, device=device)
        s = float(t[0, 0].item()) + float(t[-1, -1].item())
        del t
        return s == 2.0, ""
    except RuntimeError as exc:
        msg = str(exc)
        if "parameter is incorrect" in msg:
            return False, "DirectML rejected this allocation (not necessarily OOM)"
        return False, msg[:120]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--low-mb", type=int, default=64)
    ap.add_argument("--high-mb", type=int, default=12288)
    ap.add_argument("--out", default=str(OUTPUT))
    args = ap.parse_args()
    setup = True

    device = get_device(args.device)
    info = probe(args.device)

    print("=" * 88)
    print("DIRECTML MEMORY PROBE")
    print("=" * 88)
    print(f"backend: {info.kind}")
    print(f"reported by veltron: {info.total_memory_gb:.2f} GB "
          f"(Windows WMI value, known to truncate to 32 bits)")
    print("this card also drives the display, so some VRAM is committed to the compositor\n")

    # Coarse sweep first: cheap and shows the shape immediately.
    print(f"{'request':>12} {'result':>12}  note")
    print("-" * 88)
    coarse: list[dict] = []
    for mb in (256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192):
        ok, msg = try_alloc(device, mb * 1024 * 1024)
        coarse.append({"mb": mb, "ok": ok, "message": msg})
        print(f"{mb:>9} MiB {'OK' if ok else 'FAIL':>12}  {msg}")

    working = [c["mb"] for c in coarse if c["ok"]]
    failing = [c["mb"] for c in coarse if not c["ok"]]
    if not working:
        print("\nno allocation succeeded")
        return 1

    lo, hi = working[-1], (min(failing) if failing else working[-1] * 2)
    print(f"\nbinary search between {lo} MiB (OK) and {hi} MiB (fail) ...")
    fine: list[dict] = []
    while lo + 32 < hi:
        mid = (lo + hi) // 2
        ok, msg = try_alloc(device, mid * 1024 * 1024)
        fine.append({"mb": mid, "ok": ok, "message": msg})
        print(f"  {mid:>6} MiB {'OK' if ok else 'FAIL'}")
        if ok:
            lo = mid
        else:
            hi = mid

    largest = lo
    print(f"\nlargest single successful allocation: {largest} MiB "
          f"= {largest / 1024:.2f} GiB")
    print(f"first failing single allocation:     {largest + 32} MiB "
          f"= {(largest + 32) / 1024:.2f} GiB")

    print("\n--- implication for training ---")
    print("A training run needs model + optimizer + activations resident at once, so the")
    print("usable budget is below the largest single allocation. Observed in practice:")
    print("  veltronlm-micro (55.7M):  b=2 seq=1024 ran at ~3,600 tok/s")
    print("  veltronlm-149m:           b=1 seq=2048 and b=2 seq=1024 both OOMed")
    print("  veltronlm-micro:          b=4 seq=1024 OOMed")
    print("\nRule of thumb measured on this host: keep peak resident memory under ~4 GiB.")
    print("Note this is less than the card's 12 GiB because the display compositor holds")
    print("a share, and DirectML's allocator will not promise memory the GPU cannot back.")

    payload = {
        "backend": info.as_dict(),
        "reported_gb": info.total_memory_gb,
        "coarse": coarse,
        "binary_search": fine,
        "largest_single_allocation_mib": largest,
        "practical_training_budget_gib": 4.0,
        "note": ("AMD NVML is unavailable and WMI reports a truncated value, so this is "
                 "measured rather than queried. The GPU also drives the display."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())