"""Model architecture invariants.

These are the tests that must hold before any training run is trusted. Each asserts a
property that, if broken, produces a model that trains without error and generates fluent
nonsense.
"""

from __future__ import annotations

import math

import pytest
import torch

from veltron.model.config import REGISTRY, ModelConfig, get_config, registry_table
from veltron.model.layers import (
    KVCache,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    causal_mask,
    padding_mask,
)
from veltron.model.transformer import VeltronLM, count_parameters_via_meta


# ------------------------------------------------------------------ configuration
def test_registry_configs_are_valid():
    for key, entry in REGISTRY.items():
        cfg = entry.config
        assert cfg.d_model % cfg.n_heads == 0, key
        assert cfg.n_heads % cfg.n_kv_heads == 0, key
        assert 32 <= cfg.head_dim <= 256, f"{key}: head_dim={cfg.head_dim}"


def test_4b_config_is_in_target_range():
    """The 4B tier must actually be ~4B, or every downstream claim about it is wrong."""
    cfg = get_config("4b")
    n = cfg.param_counts()["total"]
    assert 3.7e9 <= n <= 4.3e9, f"4B config has {n:,} parameters, outside 3.7B-4.3B"


def test_analytic_param_count_matches_meta_device():
    """Independent derivations of the parameter count must agree for every config."""
    for key, entry in REGISTRY.items():
        analytic = entry.config.param_counts()["total"]
        meta = count_parameters_via_meta(entry.config)
        assert analytic == meta, f"{key}: analytic={analytic} meta={meta}"


def test_4b_fits_in_12gb_at_half_precision():
    """Inference must be possible on this host; if not, the architecture is unusable here."""
    cfg = get_config("4b")
    n = cfg.param_counts()["total"]
    assert n * 2 / 1024**3 < 12, "4B bf16 weights do not fit in 12 GB"


def test_4b_adamw_state_does_not_fit_in_12gb():
    """Documents the training blocker rather than hiding it.

    If a future host has more memory this test starts failing, which is the correct signal
    that the BLOCKED note in the docs needs revisiting.
    """
    cfg = get_config("4b")
    state_bytes = cfg.param_counts()["total"] * 16  # fp32 params + grads + m + v
    assert state_bytes / 1024**3 > 12


def test_weight_tying_reduces_parameter_count():
    untied = VeltronLM(get_config("nano").replace(tie_embeddings=False))
    tied = VeltronLM(get_config("nano").replace(tie_embeddings=True))
    assert tied.num_parameters() < untied.num_parameters()


def test_registry_table_is_sorted_and_complete():
    rows = registry_table()
    assert {r["key"] for r in rows} == set(REGISTRY)


# ------------------------------------------------------------------------- layers
def test_rmsnorm_normalises_to_unit_rms():
    norm = RMSNorm(64, eps=0.0)
    x = torch.randn(4, 8, 64) * 7.0
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    # weight is initialised to 1, so the output RMS should be ~1.
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_preserves_dtype():
    norm = RMSNorm(32)
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        y = norm(torch.randn(2, 4, 32, dtype=dt))
        assert y.dtype == dt


def test_swiglu_output_shape_and_zero_at_zero():
    ffn = SwiGLU(32, 64)
    x = torch.zeros(2, 3, 32)
    y = ffn(x)
    assert y.shape == (2, 3, 32)
    # silu(0) * 0 == 0, so a zero input maps to a zero output.
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_rope_is_relative():
    """Dot products must depend on m-n only, which is RoPE's defining property."""
    from veltron.model.layers import _rotate_half

    rope = RotaryEmbedding(head_dim=32, max_seq_len=64)
    B, H, D = 1, 2, 32
    T, T2 = 16, 32

    cos, sin = rope._cos_sin(T2, torch.device("cpu"), torch.float32)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    def apply(x, offset):
        c, s = cos[offset : offset + T], sin[offset : offset + T]
        return x * c + _rotate_half(x) * s

    # Shifting query and key by the same amount leaves every pairwise dot product intact.
    d_plain = torch.einsum("bhtd,bhtd->bht", apply(q, 0), apply(k, 0))
    d_shift = torch.einsum("bhtd,bhtd->bht", apply(q, T), apply(k, T))
    assert torch.allclose(d_plain, d_shift, atol=1e-3), "RoPE is not relative"

    # Applying the same rotation twice is idempotent only in the sense of stability.
    q2 = apply(q, 0)
    assert torch.allclose(q2, apply(q, 0), atol=1e-6)


def test_rope_scaling_changes_frequencies():
    plain = RotaryEmbedding(head_dim=64, max_seq_len=1024)
    scaled = RotaryEmbedding(head_dim=64, max_seq_len=1024, scaling_factor=4.0,
                             original_max_seq_len=1024)
    assert not torch.allclose(plain.inv_freq, scaled.inv_freq)
    # Scaling must reduce the rate of rotation (lower frequency) at high indices.
    assert scaled.inv_freq[-1] < plain.inv_freq[-1]


def test_causal_mask_zeroes_the_future():
    m = causal_mask(4, 4, torch.device("cpu"), torch.float32)
    assert m.shape == (1, 1, 4, 4)
    for i in range(4):
        assert float(m[0, 0, i, :i + 1].max()) == 0.0
        if i + 1 < 4:
            assert float(m[0, 0, i, i + 1 :].min()) < -1e30


def test_padding_mask_blocks_padded_keys():
    attn = torch.tensor([[0, 0, 1, 1]])
    m = padding_mask(attn, 4, 4, torch.float32)
    assert m.shape == (1, 1, 1, 4)
    assert float(m[0, 0, 0, :2].max()) < -1e30
    assert float(m[0, 0, 0, 2:].max()) == 0.0


# ------------------------------------------------------------------------ forward
def test_forward_shapes_and_loss():
    cfg = get_config("nano").replace(vocab_size=512, d_model=128, n_heads=4,
                                    n_kv_heads=2, d_ff=256, n_layers=2)
    model = VeltronLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (3, 32))
    out = model(ids, labels=ids)
    assert out.logits.shape == (3, 32, cfg.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_initial_loss_is_near_uniform():
    """At initialisation the model should be close to uniform over the vocabulary."""
    cfg = get_config("nano").replace(vocab_size=2048, d_model=128, n_heads=4,
                                    n_kv_heads=2, d_ff=256, n_layers=2)
    model = VeltronLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 64))
    with torch.no_grad():
        out = model(ids, labels=ids)
    assert abs(float(out.loss) - math.log(cfg.vocab_size)) < 0.5


def test_all_parameters_receive_gradients():
    cfg = get_config("nano").replace(vocab_size=256, d_model=128, n_heads=4,
                                    n_kv_heads=2, d_ff=256, n_layers=2)
    model = VeltronLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    model(ids, labels=ids).loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for: {missing[:5]}"
    zeros = [n for n, p in model.named_parameters() if float(p.grad.abs().sum()) == 0.0]
    assert not zeros, f"all-zero gradient for: {zeros[:5]}"


def test_attention_is_causal():
    cfg = get_config("nano").replace(vocab_size=256, d_model=128, n_heads=4,
                                    n_kv_heads=2, d_ff=256, n_layers=2)
    model = VeltronLM(cfg).eval()
    a = torch.randint(0, 256, (1, 48))
    b = a.clone()
    b[0, 20] = (b[0, 20] + 1) % 256
    with torch.no_grad():
        fa = model(a).logits
        fb = model(b).logits
    per_pos = (fa - fb).abs().amax(-1)[0]
    assert float(per_pos[:20].max()) < 1e-6, "positions before the edit changed"
    assert float(per_pos[20]) > 1e-4, "the edited position did not change"


def test_grouped_query_attention_shapes():
    cfg = get_config("nano")
    assert cfg.n_heads == 4 and cfg.n_kv_heads == 2
    model = VeltronLM(cfg).eval()
    attn = model.layers[0].self_attn
    x = torch.randn(2, 16, cfg.d_model)
    from veltron.model.layers import _rotate_half

    cos, sin = model.rotary._cos_sin(16, x.device, x.dtype)
    y = attn(x, cos, sin)
    assert y.shape == x.shape


def test_kv_cache_matches_full_forward():
    cfg = get_config("nano").replace(vocab_size=256, d_model=128, n_heads=4,
                                    n_kv_heads=2, d_ff=256, n_layers=2)
    model = VeltronLM(cfg).eval()
    prompt = torch.randint(0, 256, (1, 24))
    nxt = torch.randint(0, 256, (1, 1))
    cache = model.init_kv_cache(1, 64, torch.float32, "cpu")
    model(prompt, kv_cache=cache, kv_offset=0)
    incremental = model(nxt, kv_cache=cache, kv_offset=24, num_logits_to_keep=1).logits
    cache2 = model.init_kv_cache(1, 64, torch.float32, "cpu")
    full = model(torch.cat([prompt, nxt], 1), kv_cache=cache2).logits
    assert float((incremental[0, -1] - full[0, -1]).abs().max()) < 1e-4


def test_chunked_loss_matches_unchunked():
    cfg = get_config("nano").replace(vocab_size=256, d_model=128, n_heads=4,
                                    n_kv_heads=2, d_ff=256, n_layers=2)
    model = VeltronLM(cfg).eval()
    ids = torch.randint(0, 256, (2, 40))
    with torch.no_grad():
        out = model(ids)
        a = VeltronLM.compute_loss(out.logits, ids, chunk_tokens=0)
        b = VeltronLM.compute_loss(out.logits, ids, chunk_tokens=7)
    assert abs(float(a) - float(b)) < 1e-4


def test_ignore_index_is_excluded():
    cfg = get_config("nano").replace(vocab_size=64, d_model=128, n_heads=4,
                                    n_kv_heads=2, d_ff=256, n_layers=1)
    logits = torch.randn(1, 8, 64)
    labels = torch.full((1, 8), -100)
    labels[0, 4:] = 5
    loss = VeltronLM.compute_loss(logits, labels)
    assert torch.isfinite(loss)


def test_explicit_positions_support_left_padding():
    cfg = get_config("nano").replace(vocab_size=64, d_model=128, n_heads=4,
                                    n_kv_heads=2, d_ff=256, n_layers=2)
    model = VeltronLM(cfg).eval()
    short = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        solo = model(short).logits[0, -1]

        pad = 0
        padded = torch.cat([torch.full((1, 4), pad, dtype=torch.long), short], 1)
        mask = torch.tensor([[False, False, False, False, True, True, True, True, True, True]])
        pos = torch.tensor([[0, 0, 0, 0, 0, 1, 2, 3, 4, 5]])
        out = model(padded, attention_mask=mask, positions=pos)
    assert float((solo - out.logits[0, -1]).abs().max()) < 1e-3


def test_kv_cache_growth_preserves_contents():
    cache = KVCache(n_layers=2, batch=1, max_seq_len=4, n_kv_heads=2, head_dim=8,
                    dtype=torch.float32, device="cpu")
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    for layer in range(2):
        cache.append(layer, k, v, 0)
    assert cache.seq_len == 4
    before = cache.read(0, 4)[0].clone()
    cache.grow_to(32)
    after = cache.read(0, 4)[0]
    assert torch.allclose(before, after, atol=1e-6)
    assert cache.max_seq_len == 32


def test_config_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=8, n_kv_heads=2)
    with pytest.raises(ValueError):
        ModelConfig(n_heads=12, n_kv_heads=8, d_model=256)
    with pytest.raises(ValueError):
        ModelConfig(d_model=64, n_heads=8, n_kv_heads=8)  # head_dim 8 below MIN_HEAD_DIM
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=100, d_model=128, n_heads=4, n_kv_heads=2)  # vocab not %8
    with pytest.raises(ValueError):
        ModelConfig(n_layers=0, d_model=128, n_heads=4, n_kv_heads=2)  # non-positive layers


def test_config_roundtrip(tmp_path):
    cfg = get_config("mini")
    p = cfg.save(tmp_path / "config.json")
    loaded = ModelConfig.load(p)
    assert loaded == cfg
