"""117M-parameter configuration: feasibility, memory, and measured throughput.

Answers "can this GPU train a 117M model?" with arithmetic and a real timing run rather
than an estimate.

    python scripts/feasibility_117m.py            # design + memory + FLOPs
    python scripts/feasibility_117m.py --measure   # add a real timing run
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from veltron.model.config import ModelConfig, get_config, registry_table
from veltron.model.transformer import VeltronLM, count_parameters_via_meta
from veltron.utils.device import get_device, probe

OUTPUT = Path("reports/feasibility_117m.json")

TARGET = 117_000_000


def design(n_params: int = TARGET, vocab: int = 32768) -> ModelConfig:
    """Search an architecture grid for the config closest to the target size.

    Searched rather than hand-tuned so the reported parameter count is whatever the
    architecture actually produces, not what a formula hoped for. The grid extends below
    d_model=768 because at a 32k vocabulary the embedding plus output head alone cost
    ``2 x 32768 x d_model`` -- 50.3M parameters at d_model=768 -- which is why a naive
    16-layer/768-wide design lands at 149M rather than 117M.
    """
    best: tuple[float, ModelConfig] | None = None
    for layers in (8, 10, 11, 12, 14, 16, 18, 20, 22, 24):
        for d_model in (512, 576, 640, 704, 768, 832, 896, 1024):
            for n_heads in (8, 12, 16):
                if d_model % n_heads or d_model // n_heads < 32:
                    continue
                for kv_div in (2, 4):
                    n_kv = max(1, n_heads // kv_div)
                    if n_heads % n_kv:
                        continue
                    for ratio in (2.67, 3.0, 3.5, 4.0):
                        cfg = ModelConfig(
                            name="veltronlm-117m",
                            vocab_size=vocab, n_layers=layers, d_model=d_model,
                            n_heads=n_heads, n_kv_heads=n_kv,
                            d_ff=int(d_model * ratio), max_seq_len=2048,
                        )
                        n = cfg.param_counts()["total"]
                        err = abs(n - n_params)
                        if best is None or err < best[0]:
                            best = (err, cfg)
    assert best is not None
    n = best[1].param_counts()["total"]
    return best[1].replace(name=f"veltronlm-{round(n / 1e6)}m")


def memory_budget(n_params: int, batch: int, seq: int, d_model: int, n_layers: int,
                  n_heads: int, vocab: int) -> dict[str, float]:
    """Estimate peak GPU memory in bytes.

    Optimizer state dominates at this scale and is exact. Activations are estimated for
    the no-gradient-checkpointing path, which is what the trainer currently uses.
    """
    params_fp32 = n_params * 4
    grads_fp32 = n_params * 4
    adam = n_params * 8
    base = params_fp32 + grads_fp32 + adam

    # Rough activation footprint: ~12 hidden-width tensors per layer plus attention
    # scores (B * H * T * T) and the fp32 log-softmax chunk.
    per_layer = batch * seq * d_model * 2 * 12
    attn = batch * n_heads * seq * seq * 4 * 2
    loss_chunk = 4096 * vocab * 4 * 2
    activations = n_layers * per_layer + attn + loss_chunk
    total = base + activations
    return {
        "params_fp32_bytes": params_fp32,
        "grads_fp32_bytes": grads_fp32,
        "adam_state_bytes": adam,
        "optimizer_subtotal_bytes": base,
        "activations_estimated_bytes": activations,
        "peak_estimated_bytes": total,
        "peak_estimated_gib": total / 1024**3,
        "optimizer_gib": base / 1024**3,
    }


def trainable(params: int, seq: int, tok_per_s: float, epochs: float,
              corpus_tokens: int = 19_485_297) -> dict[str, float]:
    """Tokens/second and wall-clock for a given compute budget."""
    flops_per_token = 6 * params
    effective_tflops = flops_per_token * tok_per_s / 1e12
    total_tokens = corpus_tokens * epochs
    seconds = total_tokens / tok_per_s
    return {
        "params": params,
        "tokens": total_tokens,
        "seconds": seconds,
        "hours": seconds / 3600,
        "effective_tflops": effective_tflops,
        "model_flops_utilisation_of_2_8_tflops": (flops_per_token * tok_per_s) / (2.8e12),
        "tokens_per_param": total_tokens / params,
    }


def measure(cfg: ModelConfig, device, batch: int, seq: int, steps: int = 3) -> dict:
    """Real timing: forward + backward + AdamW, on this GPU, with this backend."""
    model = VeltronLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    labels = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)

    times: list[float] = []
    for i in range(steps + 2):
        if device.type != "cpu":
            torch.zeros(1, device=device).cpu()
        t0 = time.perf_counter()
        try:
            out = model(ids, labels=labels, loss_chunk_tokens=4096)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "alloc" in str(exc).lower():
                return {"status": "OOM", "batch": batch, "seq": seq,
                        "reason": str(exc)[:200]}
            raise
        if device.type != "cpu":
            torch.zeros(1, device=device).cpu()
        dt = time.perf_counter() - t0
        if i >= 2:  # discard warm-up iterations
            times.append(dt)

    tps = batch * seq / statistics.median(times)
    return {
        "status": "OK",
        "batch": batch,
        "seq": seq,
        "seconds_per_step": round(statistics.median(times), 3),
        "tokens_per_second": round(tps, 1),
        "effective_tflops": round(6 * cfg.param_counts()["total"] * tps / 1e12, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true", help="run a real timing test")
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(OUTPUT))
    args = ap.parse_args()

    device = get_device(args.device)
    info = probe(args.device)
    vram_gib = info.total_memory_gb

    print("=" * 92)
    print(f"117M FEASIBILITY ANALYSIS  (device: {info.kind}, {vram_gib} GB reported)")
    print("=" * 92)

    cfg = design(args.target)
    counts = cfg.param_counts()
    n = counts["total"]
    meta = count_parameters_via_meta(cfg)

    print("\n--- architecture ---")
    print(f"  name          {cfg.name}")
    print(f"  layers        {cfg.n_layers}")
    print(f"  d_model       {cfg.d_model}")
    print(f"  heads         {cfg.n_heads} query / {cfg.n_kv_heads} KV (GQA {cfg.gqa_group_size})")
    print(f"  head_dim      {cfg.head_dim}")
    print(f"  ffn           {cfg.d_ff} (SwiGLU, ratio {cfg.d_ff / cfg.d_model:.2f})")
    print(f"  vocab         {cfg.vocab_size:,}")
    print(f"  context       {cfg.max_seq_len:,}")
    print(f"  parameters    {n:,}  ({n / 1e6:.2f}M)")
    print(f"  analytic == meta: {n == meta}")

    print("\n--- parameter breakdown ---")
    for label, key in (("embedding", "embedding"), ("attention", "attention_total"),
                       ("mlp", "mlp_total"), ("norms", "norms_total"),
                       ("lm head", "output_head")):
        v = counts[key]
        print(f"  {label:12} {v:>14,}  {100.0 * v / n:6.2f}%")
    print(f"  {'per layer':12} {counts['per_layer']:>14,}")
    print(f"  {'non-embedding':12} {counts['non_embedding']:>14,}  "
          f"({100.0 * counts['non_embedding'] / n:.1f}%)")

    print("\n--- memory requirement (AdamW, fp32 master weights) ---")
    print(f"  {'fp32 params':24} {n * 4 / 1024**3:8.2f} GiB")
    print(f"  {'fp32 grads':24} {n * 4 / 1024**3:8.2f} GiB")
    print(f"  {'Adam m + v':24} {n * 8 / 1024**3:8.2f} GiB")
    print(f"  {'optimizer subtotal':24} {n * 16 / 1024**3:8.2f} GiB")

    for batch, seq in ((1, 2048), (1, 1024), (2, 1024)):
        mem = memory_budget(n, batch, seq, cfg.d_model, cfg.n_layers, cfg.n_heads,
                            cfg.vocab_size)
        verdict = "FITS" if mem["peak_estimated_gib"] < vram_gib else "TOO TIGHT"
        print(f"  b={batch} seq={seq}: peak ~{mem['peak_estimated_gib']:.2f} GiB  "
              f"(activations {mem['activations_estimated_bytes'] / 1024**3:.2f})  {verdict}")

    print("\n--- fit verdict ---")
    optimizer_gib = n * 16 / 1024**3
    print(f"  optimizer state      {optimizer_gib:6.2f} GiB")
    print(f"  reported VRAM        {vram_gib:6.2f} GiB")
    headroom = vram_gib - optimizer_gib
    print(f"  headroom for activations {headroom:6.2f} GiB")
    print(f"  verdict: {'TRAINABLE on this GPU' if optimizer_gib < vram_gib * 0.75 else 'NOT trainable'}")
    print(f"  (4B needed 60.00 GiB against the same {vram_gib:.0f} GiB -> 5.0x shortfall)")

    results: dict = {
        "device": info.as_dict(),
        "config": cfg.to_dict(),
        "parameters": n,
        "analytic_matches_meta": n == meta,
        "breakdown": counts,
    }

    if args.measure:
        print("\n--- measured throughput (real forward+backward+AdamW) ---")
        trials = []
        for batch, seq in ((1, 2048), (1, 1024), (2, 1024), (2, 512)):
            r = measure(cfg, device, batch, seq, steps=2)
            trials.append(r)
            if r["status"] == "OK":
                print(f"  b={batch} seq={seq}: {r['tokens_per_second']:>9,.0f} tok/s  "
                      f"{r['seconds_per_step']:>7.2f} s/step  "
                      f"{r['effective_tflops']:>6.3f} TFLOP/s eff")
            else:
                print(f"  b={batch} seq={seq}: OOM  {r['reason'][:70]}")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        ok = [t for t in trials if t["status"] == "OK"]
        results["measured"] = trials
        if ok:
            best = max(ok, key=lambda t: t["tokens_per_second"])
            tps = best["tokens_per_second"]
            print(f"\n  best: b={best['batch']} seq={best['seq']} at {tps:,.0f} tok/s")
            print("\n--- training budget at that throughput ---")
            corpus = 19_485_297
            for epochs in (1, 5, 20):
                t = trainable(n, best["seq"], tps, epochs, corpus)
                print(f"  {epochs:2d} epoch(s): {t['tokens']:>13,.0f} tokens  "
                      f"{t['hours']:>7.2f} h  "
                      f"{t['tokens_per_param']:>6.2f} tok/param  "
                      f"{t['model_flops_utilisation_of_2_8_tflops'] * 100:>5.1f}% of peak")
                results.setdefault("budget", {})[f"epochs_{epochs}"] = t

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())