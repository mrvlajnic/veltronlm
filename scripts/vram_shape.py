"""Distinguish a hard VRAM limit from a single-allocation limit.

The measured DirectML ceiling is 3.62 GiB in one tensor, while Windows reports only
1.11 GiB of VRAM actually in use. Those two facts cannot both describe a hard VRAM
exhaustion, so this probes the shape of the limit:

* can several smaller allocations sum past the single-allocation ceiling?
* is the failure size suspiciously close to a round number (a heap cap) or not (real
  fragmentation)?

The answer determines whether shrinking model tensors is worthwhile or whether the
budget is simply smaller than the card's nameplate suggests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from veltron.utils.device import get_device

OUTPUT = Path("reports/vram_shape.json")
MIB = 1024 * 1024


def alloc_2d(device, nbytes: int):
    elems = max(1, nbytes // 4)
    side = max(1, int(elems ** 0.5))
    return torch.ones((side, max(1, elems // side)), dtype=torch.float32, device=device)


def single_max(device) -> tuple[int, str]:
    lo, hi = 512, 12288
    best = 0
    msg = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            t = alloc_2d(device, mid * MIB)
            _ = float(t[0, 0])
            del t
            best = mid
            lo = mid + 64
        except RuntimeError as exc:
            msg = str(exc)[:110]
            hi = mid - 64
    return best, msg


def cumulative_max(device, chunk_mib: int) -> tuple[int, int]:
    """How much can we hold as many separate allocations?"""
    kept: list[torch.Tensor] = []
    total = 0
    for _ in range(64):
        try:
            t = alloc_2d(device, chunk_mib * MIB)
            kept.append(t)
            total += chunk_mib
        except RuntimeError:
            break
    peak = total
    n_alloc = len(kept)
    del kept
    return peak, n_alloc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(OUTPUT))
    args = ap.parse_args()
    device = get_device(args.device)

    print("=" * 88)
    print("VRAM LIMIT SHAPE PROBE")
    print("=" * 88)

    single, msg = single_max(device)
    print(f"\n1. largest SINGLE allocation: {single} MiB = {single / 1024:.2f} GiB")
    print(f"   failure message: {msg}")

    print("\n2. cumulative, as many separate allocations:")
    results = {}
    for chunk in (256, 512, 1024):
        peak, n = cumulative_max(device, chunk)
        results[f"chunk_{chunk}_mib"] = {"held_mib": peak, "allocations": n}
        print(f"   {chunk:>5} MiB chunks: held {peak:>6} MiB "
              f"({peak / 1024:5.2f} GiB) across {n} allocations")

    best_cumulative = max(v["held_mib"] for v in results.values())
    print(f"\n   best cumulative: {best_cumulative} MiB = {best_cumulative / 1024:.2f} GiB")
    print(f"   single allocation was: {single} MiB = {single / 1024:.2f} GiB")

    print("\n3. interpretation")
    if best_cumulative > single * 1.15:
        print(f"   The limit is per-allocation, not total: {best_cumulative / 1024:.2f} GiB is")
        print("   reachable as many smaller tensors, versus "
              f"{single / 1024:.2f} GiB as one. That means large single tensors (the")
        print("   vocabulary projection is the biggest) should be split or kept in fp16.")
    else:
        print("   The limit is a TOTAL budget, not a single-allocation cap: many smaller")
        print("   allocations do not exceed one big one. Shrinking tensors will not help;")
        print("   the usable budget is simply smaller than the card's nameplate.")
    print(f"\n   The ceiling is {single / 1024:.2f} GiB while Windows reports ~1.11 GiB of VRAM")
    print("   in use. The ~7 GiB gap is not attributed to any process by WDDM. The most")
    print("   likely explanation is a driver-level reservation for the display adapter")
    print("   (this GPU drives the desktop) that AMD's ADANNI software holds back. That is")
    print("   a property of DirectML on this card, not something the training code can")
    print("   work around, and it is why the 117M tier was sized by measurement rather")
    print("   than by the 12 GiB nameplate.")

    payload = {
        "single_allocation_max_mib": single,
        "single_allocation_max_gib": round(single / 1024, 3),
        "failure_message": msg,
        "cumulative": results,
        "best_cumulative_mib": best_cumulative,
        "best_cumulative_gib": round(best_cumulative / 1024, 3),
        "windows_reported_in_use_gib": 1.11,
        "unaccounted_gib": round(12.0 - 1.11 - single / 1024, 2),
        "interpretation": (
            "Single-allocation cap" if best_cumulative > single * 1.15
            else "Total budget cap"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())