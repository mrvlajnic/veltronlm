"""Measure per-process GPU memory on Windows using the GPU performance counters.

The 3.62 GiB DirectML allocation ceiling was measured, but the *cause* was assumed
rather than verified. Windows exposes dedicated-memory usage per process through the
``GPU Adapter Memory`` and ``GPU Process Memory`` counters, so the actual consumer can be
identified instead of guessed at.

    python scripts/gpu_memory_report.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

OUTPUT = Path("reports/gpu_memory.json")
GIB = 1024 ** 3


def query_counters() -> list[str]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "(Get-Counter -ListSet 'GPU Process Memory').PathsWithSpaces"],
        capture_output=True, text=True, timeout=60,
    )
    return [p.strip() for p in out.stdout.split("\n") if p.strip()]


def sample_dedicated_usage() -> tuple[list[dict], list[dict]]:
    """One sample of the GPU memory counters, grouped per adapter and per process."""
    script = r"""
$ErrorActionPreference = 'SilentlyContinue'
$adapter = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage'
$procs   = Get-Counter '\GPU Process Memory(*)\Dedicated Usage'
[pscustomobject]@{
  adapter = @($adapter.CounterSamples | ForEach-Object {
      [pscustomobject]@{ instance = $_.InstanceName; bytes = $_.CookedValue } })
  process = @($procs.CounterSamples | ForEach-Object {
      [pscustomobject]@{ instance = $_.InstanceName; bytes = $_.CookedValue } })
} | ConvertTo-Json -Depth 5
"""
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                         capture_output=True, text=True, timeout=120)
    txt = out.stdout.strip()
    if not txt:
        return [], []
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        return [], []
    if isinstance(data, str):
        data = json.loads(data)
    return (data.get("adapter") or []), (data.get("process") or [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUT))
    args = ap.parse_args()

    print("=" * 92)
    print("WINDOWS GPU MEMORY REPORT")
    print("=" * 92)

    paths = query_counters()
    print(f"\nGPU counter sets available: {len(paths)}")
    if not paths:
        print("GPU counters unavailable (not Windows, or counters disabled).")
        return 1

    adapters, processes = sample_dedicated_usage()
    if not adapters and not processes:
        print("No samples returned. GPU counters may need an elevated shell.")
        return 1

    print("\n--- per adapter ---")
    print(f"{'adapter':52}{'used':>12}")
    print("-" * 92)
    total = 0.0
    for a in sorted(adapters, key=lambda x: -x.get("bytes", 0)):
        gib = a.get("bytes", 0) / GIB
        total = max(total, gib)
        print(f"{a.get('instance', '?')[:52]:52}{gib:>10.2f} GiB")

    print("\n--- per process (top 20) ---")
    print(f"{'instance':62}{'used':>12}")
    print("-" * 92)
    top = sorted(processes, key=lambda x: -x.get("bytes", 0))[:20]
    for p in top:
        if p.get("bytes", 0) <= 0:
            continue
        print(f"{p.get('instance', '?')[:62]:62}{p.get('bytes', 0) / GIB:>10.2f} GiB")

    nonzero = [p for p in processes if p.get("bytes", 0) > 0]
    print(f"\nprocesses holding dedicated GPU memory: {len(nonzero)}")
    print(f"highest per-adapter usage observed:      {total:.2f} GiB")

    print("\n--- what this means for training ---")
    print("DirectML could allocate 3.62 GiB in a single tensor (measured in")
    print("scripts/measure_vram.py). Whatever is NOT held by other processes should be")
    print("available to training. If per-process usage above is only a fraction of the")
    print("card's capacity, then the ceiling is set by something else -- most likely")
    print("DirectML's own allocator reserving a conservative share, or the display")
    print("adapter's fixed overhead that is not attributed to any process.")

    payload = {
        "per_adapter_gib": {a.get("instance", "?"): round(a.get("bytes", 0) / GIB, 3) for a in adapters},
        "per_process_gib": {p.get("instance", "?"): round(p.get("bytes", 0) / GIB, 3)
                            for p in processes if p.get("bytes", 0) > 0},
        "highest_adapter_gib": round(total, 3),
        "measured_directml_ceiling_gib": 3.62,
        "note": ("Dedicated usage is sampled once; WDDM does not attribute the display "
                 "adapter's fixed framebuffer and compositing overhead to any process, so "
                 "the per-process total can be lower than the memory a compute process "
                 "actually fails to obtain."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())