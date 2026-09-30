"""End-to-end check: train nano briefly, then generate, save/load, and sample deterministically."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import torch

from veltron.inference.generator import GenerationConfig, Generator
from veltron.model.config import get_config
from veltron.model.io import load_safetensors, save_safetensors
from veltron.model.transformer import VeltronLM
from veltron.tokenizer.trainer import fallback_byte_tokenizer
from veltron.utils.device import get_device, probe
from veltron.utils.seed import seed_everything


def main() -> int:
    seed_everything(7)
    dev = get_device("auto")
    print("device:", probe("auto").as_dict())

    cfg = get_config("nano")
    model = VeltronLM(cfg).to(dev)
    tok = fallback_byte_tokenizer()
    print(f"model params {sum(p.numel() for p in model.parameters()):,}  vocab {tok.get_vocab_size()}")

    gen = Generator(model, tok, device=dev)

    # --- untrained: verify the plumbing, not the quality
    prompt = "VeltronHub X1 is offline. "
    r = gen.generate(prompt, GenerationConfig(max_new_tokens=24, temperature=0.0, greedy=True))
    print(f"\n[untrained greedy] tokens={r.completion_tokens} reason={r.finish_reason} "
          f"tok/s={r.tokens_per_second:.1f} ttft={r.time_to_first_token*1000:.0f}ms")
    print("  text repr:", repr(r.text[:160]))

    # --- overfit a single sequence so generation becomes meaningful, then re-check
    print("\noverfitting one sequence for 120 steps...")
    ids = torch.tensor([tok.encode("The VeltronHub X1 hub is offline. Restart the device.")][:1], device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    t0 = time.perf_counter()
    losses = []
    for step in range(120):
        out = model(ids, labels=ids)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(out.loss))
    print(f"  loss {losses[0]:.4f} -> {losses[-1]:.4f} in {time.perf_counter()-t0:.1f}s")
    assert losses[-1] < losses[0] * 0.2, "overfit failed"

    r = gen.generate("The VeltronHub X1 hub is offline.", GenerationConfig(max_new_tokens=30, greedy=True))
    print(f"[overfit greedy] {r.text!r}")

    # --- sampling
    r = gen.generate("The VeltronHub X1 hub is offline.",
                     GenerationConfig(max_new_tokens=20, temperature=0.9, top_k=20, top_p=0.9, seed=3))
    print(f"[sampled k=20 p=0.9 seed=3] {r.text!r}")
    r2 = gen.generate("The VeltronHub X1 hub is offline.",
                      GenerationConfig(max_new_tokens=20, temperature=0.9, top_k=20, top_p=0.9, seed=3))
    same = r.text == r2.text
    print(f"[seeded determinism] identical: {same}")
    assert same, "seeded sampling must be reproducible"

    # --- sampling strategies differ
    a = gen.generate("The VeltronHub", GenerationConfig(max_new_tokens=16, greedy=True, seed=1)).text
    b = gen.generate("The VeltronHub", GenerationConfig(max_new_tokens=16, temperature=0.0, greedy=True, seed=1)).text
    assert a == b
    print("[greedy determinism] identical: True")

    # --- stop token
    r = gen.generate("The VeltronHub", GenerationConfig(max_new_tokens=50, greedy=True,
                                                      stop_strings=[" offline"]))
    print(f"[stop_string] finish={r.finish_reason} tokens={r.completion_tokens} text={r.text!r}")

    # --- streaming
    deltas = list(gen.stream("The VeltronHub X1", GenerationConfig(max_new_tokens=12, greedy=True)))
    print(f"[stream] {len(deltas)} deltas: {''.join(deltas)!r}")

    # --- scoring (used by RAG ranking and preference eval)
    sc = gen.score("The VeltronHub X1 hub is", " offline. Restart the device.")
    sc2 = gen.score("The VeltronHub X1 hub is", " purple elephants dance nightly.")
    print(f"[score] target sum={sc['sum_logprob']:.3f} mean={sc['mean_logprob']:.3f} "
          f"ppl={sc['perplexity']:.2f}")
    print(f"[score] distractor sum={sc2['sum_logprob']:.3f} mean={sc2['mean_logprob']:.3f} "
          f"ppl={sc2['perplexity']:.2f}")
    assert sc["sum_logprob"] > sc2["sum_logprob"], "model should prefer the memorised string"

    # --- batched generation with left padding
    batch = [
        tok.encode("Short"),
        tok.encode("A considerably longer prompt that must be left-padded so batching works"),
    ]
    results = gen.generate_ids(batch, GenerationConfig(max_new_tokens=8, greedy=True, max_batch_size=4))
    print(f"[batched] {len(results)} results, tokens={[r.completion_tokens for r in results]}")
    assert len(results) == 2

    # --- save / load round trip
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ckpt"
        res = save_safetensors(model, p)
        print(f"[save] {res.n_tensors} tensors, {res.total_bytes/1024:.1f} KiB -> {res.path.name}")
        model2 = VeltronLM(cfg).to(dev)
        load_safetensors(model2, p)
        g2 = Generator(model2, tok, device=dev)
        a = gen.generate("The VeltronHub", GenerationConfig(max_new_tokens=10, greedy=True)).text
        b = g2.generate("The VeltronHub", GenerationConfig(max_new_tokens=10, greedy=True)).text
        print(f"[save/load] identical output: {a == b}")
        assert a == b, "reload changed the output"

    print("\nALL INFERENCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
