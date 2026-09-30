"""Inference benchmark: latency, throughput, memory, quantization cost.

Measures real numbers on real weights. The 4B tier is instantiated and (optionally)
quantized to *prove* it loads and serves here; it is clearly labelled as untrained.

    python scripts/bench_inference.py --tiers micro --prompts 8
    python scripts/bench_inference.py --tiers 4b --quantize 8,4 --report-only
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from veltron.inference.engine import find_checkpoint, load_engine
from veltron.inference.generator import GenerationConfig
from veltron.model.config import REGISTRY, get_config
from veltron.model.io import quantize_model_inplace
from veltron.model.transformer import VeltronLM
from veltron.tokenizer.trainer import VeltronTokenizer
from veltron.utils.device import empty_cache, get_device, probe, synchronize
from veltron.utils.logging_utils import setup_logging

OUTPUT = Path("reports/inference_benchmark.json")

PROMPTS = [
    "The VeltronHub X1",
    "Once upon a time",
    "def compute_loss(logits, labels):",
    "Warranty coverage",
    "What is the airspeed velocity",
    "Вештачка интелигенција је",
    "import torch\nimport torch.nn as nn\n\nclass",
    "{\"product\": \"VeltronHub X1\", \"warranty_months\":",
]


def measure_generation(engine, prompt: str, max_new_tokens: int, greedy: bool,
                       warmup: int = 1) -> dict[str, Any]:
    cfg = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0 if greedy else 0.7,
                           top_p=0.95, greedy=greedy, seed=1234)
    for _ in range(warmup):
        engine.generator.generate(prompt, cfg)
    t0 = time.perf_counter()
    res = engine.generator.generate(prompt, cfg)
    synchronize(engine.device)
    elapsed = time.perf_counter() - t0
    return {
        "prompt_chars": len(prompt),
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "seconds": round(elapsed, 4),
        "tokens_per_second": round(res.completion_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "time_to_first_token_s": round(res.time_to_first_token, 4),
        "finish_reason": res.finish_reason,
    }


def measure_batch(engine, prompts: list[str], max_new_tokens: int) -> dict[str, Any]:
    ids = [engine.tokenizer.encode(p) for p in prompts]
    ids = [i for i in ids if i] or [[1]] * len(prompts)
    cfg = GenerationConfig(max_new_tokens=max_new_tokens, greedy=True, max_batch_size=len(prompts))
    results = engine.generator.generate_ids(ids, cfg)
    total = sum(r.completion_tokens for r in results)
    return {
        "batch_size": len(prompts),
        "completion_tokens": total,
        "seconds": round(results[0].seconds, 4),
        "aggregate_tokens_per_second": round(total / max(results[0].seconds, 1e-9), 2),
    }


def measure_kv_cache(engine, prompt_tokens: int, lengths: list[int]) -> list[dict[str, Any]]:
    out = []
    ids = list(range(100, 100 + prompt_tokens))
    x = torch.tensor([ids], dtype=torch.long, device=engine.device)
    for n in lengths:
        cache = engine.model.init_kv_cache(1, n + 8, next(engine.model.parameters()).dtype,
                                           engine.device)
        cache.reset()
        # Warm-up so the first measured pass is not paying allocation cost.
        engine.model(x[:, : min(32, prompt_tokens)], kv_cache=cache, kv_offset=0,
                     num_logits_to_keep=1)
        synchronize(engine.device)
        t0 = time.perf_counter()
        for start in range(0, prompt_tokens, 128):
            engine.model(x[:, start : start + 128], kv_cache=cache, kv_offset=start,
                         num_logits_to_keep=1)
        synchronize(engine.device)
        dt = time.perf_counter() - t0
        out.append({
            "context_tokens": prompt_tokens,
            "cache_capacity": n,
            "seconds": round(dt, 4),
            "tokens_per_second": round(prompt_tokens / dt, 1),
        })
    return out


def instantiate_untrained(tier: str, device: torch.device, quantize_bits: int | None
                          ) -> tuple[Any, dict[str, Any]]:
    """Materialise a tier with random weights to measure its serving footprint.

    Labelled untrained everywhere it appears. The point is to show the architecture fits in
    memory and to measure latency, not to imply any weights exist.
    """
    from veltron.inference.engine import Engine, EngineInfo

    cfg = get_config(tier)
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    t0 = time.perf_counter()
    model = VeltronLM(cfg)
    params = sum(p.numel() for p in model.parameters())
    model = model.to(dtype)
    quant_stats = None
    if quantize_bits:
        quant_stats = quantize_model_inplace(model, bits=quantize_bits,
                                             group_size=128).as_dict()
    model = model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - t0

    vram = 0
    if device.type == "cuda":
        vram = torch.cuda.memory_allocated()
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    info = EngineInfo(
        model_name=cfg.name, checkpoint="RANDOM-INIT (untrained)", tokenizer="",
        device=probe("auto").kind, dtype=str(dtype).replace("torch.", ""),
        parameters=params, parameters_billions=params / 1e9,
        context_length=cfg.max_seq_len, vocab_size=cfg.vocab_size,
        quantized_bits=quantize_bits,
    )
    from veltron.inference.generator import Generator

    engine = Engine(model=model, tokenizer=fallback_tokenizer_for(tier),
                    generator=Generator(model, fallback_tokenizer_for(tier), device=device,
                                        dtype=dtype),
                    info=info, device=device)
    return engine, {
        "tier": tier,
        "status": "UNTRAINED — random weights, serving footprint only",
        "parameters": params,
        "weight_bytes": weight_bytes,
        "weight_gib": round(weight_bytes / 1024**3, 3),
        "load_seconds": round(load_seconds, 2),
        "quantization": quant_stats,
        "allocated_bytes": vram,
    }


_TOK: dict[str, Any] = {}


def fallback_tokenizer_for(tier: str):
    """Use the real tokenizer when its vocabulary fits the tier, else a byte tokenizer."""
    if not _TOK:
        for name in ("models/tok-mini-32k", "models/tok-4b-64k"):
            if Path(name, "tokenizer.json").exists():
                _TOK[name] = VeltronTokenizer.load(name)
    for path in ("models/tok-mini-32k", "models/tok-4b-64k"):
        tok = _TOK.get(path)
        if tok and tok.get_vocab_size() <= get_config(tier).vocab_size:
            return tok
    from veltron.tokenizer.trainer import fallback_byte_tokenizer

    return fallback_byte_tokenizer()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="*", default=["micro"])
    ap.add_argument("--checkpoint", default="", help="run a trained checkpoint instead")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--kv-lengths", nargs="*", type=int, default=[512, 2048, 8192])
    ap.add_argument("--quantize", nargs="*", type=int, default=[],
                    help="also measure these bit widths for untrained tiers")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(OUTPUT))
    ap.add_argument("--skip-kv", action="store_true")
    args = ap.parse_args()
    setup_logging(level="WARNING")

    device = get_device(args.device)
    payload: dict[str, Any] = {
        "device": probe(args.device).as_dict(),
        "results": {},
    }
    print("=" * 96)
    print("VELTRONLM INFERENCE BENCHMARK")
    print("=" * 96)
    print(f"device: {payload['device']['kind']}  dtype support: "
          f"fp16={payload['device']['supports_fp16']} bf16={payload['device']['supports_bf16']}")

    # ---- trained checkpoint
    if args.checkpoint:
        ckpt = find_checkpoint(args.checkpoint)
        print(f"\nloading checkpoint: {ckpt}")
        engine = load_engine(checkpoint=ckpt, device=args.device)
        payload["results"]["trained"] = {
            "model": engine.info.as_dict(),
            "generation": [measure_generation(engine, PROMPTS[i % len(PROMPTS)],
                                             args.max_new_tokens, True)
                           for i in range(args.prompts)],
            "batched": measure_batch(engine, PROMPTS[:4], args.max_new_tokens),
        }
        if not args.skip_kv:
            payload["results"]["trained"]["kv_cache"] = measure_kv_cache(
                engine, min(2048, engine.info.context_length), args.kv_lengths)
        g = payload["results"]["trained"]["generation"]
        tps = [x["tokens_per_second"] for x in g]
        ttft = [x["time_to_first_token_s"] for x in g]
        print(f"  generation: {len(g)} prompts  "
              f"median {statistics.median(tps):.2f} tok/s  "
              f"mean TTFT {statistics.mean(ttft)*1000:.1f} ms")
        del engine
        gc.collect()
        empty_cache(device)

    # ---- untrained tier footprints
    for tier in args.tiers:
        if tier not in REGISTRY:
            print(f"unknown tier {tier}")
            continue
        widths = [None] + list(args.quantize)
        for bits in widths:
            key = f"{tier}" + (f"-int{bits}" if bits else "")
            try:
                engine, meta = instantiate_untrained(tier, device, bits)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                print(f"\n  {key}: FAILED to instantiate on this device: "
                      f"{str(exc)[:110]}")
                payload["results"][key] = {
                    "tier": tier, "status": "FAILED",
                    "reason": str(exc)[:400],
                }
                gc.collect()
                empty_cache(device)
                continue

            print(f"\n  {key}: {meta['parameters']:,} params  "
                  f"{meta['weight_gib']:.2f} GiB weights  "
                  f"loaded in {meta['load_seconds']:.1f}s")
            if bits:
                print(f"      quantization: {json.dumps(meta['quantization'])}")
            gen = [measure_generation(engine, PROMPTS[i % len(PROMPTS)],
                                      args.max_new_tokens, True)
                   for i in range(min(args.prompts, 3))]
            tps = [x["tokens_per_second"] for x in gen]
            print(f"      generation: median {statistics.median(tps):.2f} tok/s over "
                  f"{len(gen)} prompts")
            payload["results"][key] = {
                **meta, "generation": gen,
                "median_tokens_per_second": round(statistics.median(tps), 2),
            }
            del engine
            gc.collect()
            empty_cache(device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
