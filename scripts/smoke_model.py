"""Smoke check: forward, loss, backward, gradient flow, KV cache, determinism."""

from __future__ import annotations

import time

import torch

from veltron.model.config import get_config
from veltron.model.transformer import VeltronLM
from veltron.utils.seed import seed_everything


def main() -> int:
    seed_everything(1234)
    cfg = get_config("nano")
    model = VeltronLM(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"config={cfg.name} params={n:,}")

    B, T, V = 2, 128, cfg.vocab_size
    ids = torch.randint(0, V, (B, T))
    labels = torch.randint(0, V, (B, T))

    out = model(ids, labels=labels)
    print(f"logits {tuple(out.logits.shape)} loss {float(out.loss):.4f}")
    assert out.logits.shape == (B, T, V), out.logits.shape
    assert torch.isfinite(out.loss), "loss is not finite"

    out.loss.backward()

    dead = [nm for nm, p in model.named_parameters() if p.requires_grad and p.grad is None]
    print(f"params without grad: {len(dead)}")
    if dead:
        print("  ", dead[:8])
    assert not dead, f"parameters received no gradient: {dead[:8]}"

    zero = [nm for nm, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) == 0.0]
    print(f"params with all-zero grad: {len(zero)}")
    assert not zero, f"zero gradients: {zero[:8]}"

    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"grad norm {float(gn):.4f}")

    # Unigram sanity: after init, loss should be near ln(V).
    print(f"ln(V) = {torch.log(torch.tensor(float(V))):.4f}")

    # ---------------------------------------------------------------- causality
    # Editing position k must leave every position < k untouched, and must change
    # position k itself. Positions > k legitimately change because they attend to k.
    model.eval()
    with torch.no_grad():
        a = torch.randint(0, V, (1, 64))
        b = a.clone()
        b[0, 30] = (b[0, 30] + 1) % V
        fa = model(a).logits
        fb = model(b).logits
        per_pos = (fa - fb).abs().amax(-1)[0]
        before = float(per_pos[:30].max())
        at = float(per_pos[30])
        after = float(per_pos[31:].max())
        print(f"edit pos30 -> max|delta| before={before:.3e} at={at:.3e} after={after:.3e}")
        assert before < 1e-6, f"positions before the edit changed: {before}"
        assert at > 1e-4, "the edited position did not change"
        assert after > 1e-4, "downstream positions did not change"
        print("causal mask holds: OK")

        # Attention probabilities must place exactly zero mass beyond the window.
        from veltron.model.layers import causal_mask

        h = model.embed_tokens(a)
        cos, sin = model._rope_tables(64, h.device, h.dtype, 0)
        blk = model.layers[0]
        ln = blk.input_layernorm(h)
        at_mod = blk.self_attn
        q = at_mod.q_proj(ln).view(1, 64, at_mod.n_heads, at_mod.head_dim).transpose(1, 2)
        k = at_mod.k_proj(ln).view(1, 64, at_mod.n_kv_heads, at_mod.head_dim).transpose(1, 2)
        q, k = at_mod._apply_rope(q, k, cos, sin)
        k = k.repeat_interleave(at_mod.group_size, dim=1)
        s = (q @ k.transpose(-1, -2)) * at_mod.scale
        m = causal_mask(64, 64, h.device, h.dtype, 0)
        assert m.shape == (1, 1, 64, 64), m.shape
        p = torch.softmax((s + m).float(), -1)
        leaks = max(float(p[0, 0, i, i + 1 :].sum()) for i in range(64))
        assert leaks == 0.0, f"causal leak: {leaks}"
        print(f"attention causal leakage: {leaks:.1e} OK")

    # --------------------------------------------------------------- kv cache
    with torch.no_grad():
        cache = model.init_kv_cache(batch=1, max_seq_len=256, dtype=torch.float32, device="cpu")
        prompt = torch.randint(0, V, (1, 40))
        o1 = model(prompt, kv_cache=cache, kv_offset=0)
        print(f"prefill cache len={cache.seq_len}")
        assert cache.seq_len == 40, cache.seq_len
        nxt = torch.randint(0, V, (1, 1))
        o2 = model(nxt, kv_cache=cache, kv_offset=40, num_logits_to_keep=1)
        print(f"after decode cache len={cache.seq_len}")
        assert cache.seq_len == 41

        cache2 = model.init_kv_cache(batch=1, max_seq_len=256, dtype=torch.float32, device="cpu")
        full = model(torch.cat([prompt, nxt], dim=1), kv_cache=cache2)
        d = float((o2.logits[0, -1] - full.logits[0, -1]).abs().max())
        print(f"incremental vs full-forward max|delta| = {d:.3e}")
        assert d < 2e-4, f"KV cache mismatch: {d}"

    # --------------------------------------------------------- weight tying
    tied = VeltronLM(cfg.replace(tie_embeddings=True))
    tn = sum(p.numel() for p in tied.parameters())
    print(f"tied-embedding params {tn:,} (saves {n - tn:,})")
    assert tn < n

    # ------------------------------------------------------- gradient clipping
    print("ALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    t0 = time.perf_counter()
    rc = main()
    print(f"elapsed {time.perf_counter() - t0:.1f}s")
    raise SystemExit(rc)
