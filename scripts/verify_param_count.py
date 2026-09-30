"""Programmatic parameter accounting for every registered config.

Cross-checks two independent derivations:
  A) analytic shapes from ModelConfig.param_counts()
  B) real nn.Parameter objects built on the ``meta`` device

and prints the full subsystem breakdown used in docs/model-architecture.md.
"""

from __future__ import annotations

import sys

from veltron.model.config import REGISTRY, ModelConfig
from veltron.model.transformer import count_parameters_via_meta


def table() -> list[dict]:
    rows = []
    for e in REGISTRY.values():
        c: ModelConfig = e.config
        p = c.param_counts()
        meta = count_parameters_via_meta(c)
        rows.append(
            {
                "key": e.key,
                "tier": e.tier,
                "name": c.name,
                "L": c.n_layers,
                "D": c.d_model,
                "H": c.n_heads,
                "KV": c.n_kv_heads,
                "hd": c.head_dim,
                "F": c.d_ff,
                "V": c.vocab_size,
                "ctx": c.max_seq_len,
                "total": p["total"],
                "B": round(p["total"] / 1e9, 4),
                "analytic_equals_meta": p["total"] == meta,
                "meta_total": meta,
            }
        )
    return rows


def main() -> int:
    rows = table()
    bad = [r for r in rows if not r["analytic_equals_meta"]]
    print("=" * 108)
    print("VELTRONLM MODEL REGISTRY - parameter counts computed programmatically")
    print("=" * 108)
    hdr = f"{'key':7}{'tier':5}{'L':>4}{'D':>6}{'H':>4}{'KV':>4}{'hd':>4}{'F':>7}{'V':>7}{'ctx':>7}{'params':>15}{'B':>9}  xcheck"
    print(hdr)
    print("-" * 108)
    for r in rows:
        print(
            f"{r['key']:7}{r['tier']:5}{r['L']:>4}{r['D']:>6}{r['H']:>4}{r['KV']:>4}"
            f"{r['hd']:>4}{r['F']:>7}{r['V']:>7}{r['ctx']:>7}{r['total']:>15,}{r['B']:>9}  "
            f"{'OK' if r['analytic_equals_meta'] else 'MISMATCH'}"
        )
    print("-" * 108)

    c = REGISTRY["4b"].config
    p = c.param_counts()
    print()
    print("=" * 108)
    print(f"FULL BREAKDOWN: {c.name}")
    print("=" * 108)
    items = [
        ("token embedding  (V x D)", p["embedding"]),
        ("attention q_proj  (L x D x H*Dhd)", p["attention_q"]),
        ("attention k_proj  (L x D x Hkv*Dhd)", p["attention_k"]),
        ("attention v_proj  (L x D x Hkv*Dhd)", p["attention_v"]),
        ("attention o_proj  (L x H*Dhd x D)", p["attention_out"]),
        ("attention qk_norm (L x 2*Dhd)", p["attention_qk_norm"]),
        ("mlp gate/up/down  (L x 3*D*F)", p["mlp_total"]),
        ("all norms          (L x 2*D + D)", p["norms_total"]),
        ("lm head            (V x D)", p["output_head"]),
    ]
    for label, n in items:
        pct = 100.0 * n / p["total"]
        print(f"  {label:40s} {n:>15,}  {pct:6.3f}%")
    print(f"  {'-'*40}")
    print(f"  {'per layer':40s} {p['per_layer']:>15,}  {100.0*p['per_layer']*c.n_layers/p['total']:6.3f}%")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':40s} {p['total']:>15,}")
    print(f"  {'non-embedding':40s} {p['non_embedding']:>15,}")

    print()
    print("=" * 108)
    print("4B TRAINING MEMORY REQUIREMENT (AdamW, fp32 master weights)")
    print("=" * 108)
    n = p["total"]
    items = [
        ("fp32 params", n * 4),
        ("fp32 grads", n * 4),
        ("Adam m (fp32)", n * 4),
        ("Adam v (fp32)", n * 4),
    ]
    tot = 0
    for label, b in items:
        tot += b
        print(f"  {label:26s} {b / 1024**3:8.2f} GiB")
    print(f"  {'-'*26}")
    print(f"  {'minimum to hold states':26s} {tot / 1024**3:8.2f} GiB   (+ activations, fragmentation)")

    print()
    print("4B INFERENCE FOOTPRINT")
    for dt, b in (("fp32", 4), ("bf16", 2), ("int8", 1), ("int4", 0.5)):
        w = n * b
        print(f"  weights {dt:5s} {w / 1024**3:7.2f} GiB   kv-cache @8k ctx (bs=1): "
              f"{c.kv_cache_bytes(1, 8192, 2) / 1024**3:6.3f} GiB")

    print()
    if bad:
        print("FAIL: analytic and meta counts disagree for: " + ", ".join(r["key"] for r in bad))
        return 1
    print("PASS: analytic parameter algebra matches meta-device instantiation for all configs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
