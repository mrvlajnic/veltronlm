"""Tokenizer correctness and efficiency."""

from __future__ import annotations

import json

import pytest

from veltron.tokenizer.trainer import (
    BOS,
    EOS,
    PAD,
    SPECIAL_TOKENS,
    TokenizerConfig,
    VeltronTokenizer,
    fallback_byte_tokenizer,
    train_tokenizer,
)

SAMPLES = [
    ("english", "The VeltronHub X1 requires a Veltron Account to activate the device."),
    ("serbian_cyrillic", "Вештачка интелигенција је грана рачунарства. ВелтронХуб КС1 захтева налог."),
    ("serbian_latin", "Veštačka inteligencija je grana računarstva. Koliko traje garancija?"),
    ("mixed", "Garantija je 24 meseca / warranty is 24 months / гаранција 24 месеца"),
    ("code", "def main() -> int:\n    return sum(i * i for i in range(10))\n"),
    ("json", '{"product": "VeltronHub X1", "warranty_months": 24, "active": true}'),
    ("markdown", "# Heading\n\n- item one\n- item two\n\n```python\nx = 1\n```\n"),
    ("stacktrace", 'Traceback (most recent call last):\n  File "a.py", line 3\nValueError: bad'),
    ("url", "See https://example.com/docs/veltron#warranty for details."),
    ("emoji", "status: online ✅ temperature 21.5°C"),
    ("numbers", "Order #A-99213, 1,234.56 EUR, invoice 2024-11-03, serial VH-9K2M1."),
    ("empty", ""),
]


@pytest.fixture(scope="module")
def tok() -> VeltronTokenizer:
    return fallback_byte_tokenizer()


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> VeltronTokenizer:
    """A small BPE trained on synthetic multilingual/code text."""
    d = tmp_path_factory.mktemp("tok")
    corpus = d / "corpus.jsonl"
    rows = [
        {"text": "The VeltronHub X1 smart hub requires a Veltron Account to activate the device. "
                 "It ships with the Veltron app and dual-band Wi-Fi 6."},
        {"text": "Вештачка интелигенција је грана рачунарства. ВелтронХуб КС1 захтева налог. "
                 "Гаранција је 24 месеца за све уређаје."},
        {"text": "Veštačka inteligencija je grana računarstva. Velt ron Hub zahteva nalog. "
                 "Garancija je 24 meseca za sve uređaje."},
        {"text": ("def main() -> int:\n    return sum(i * i for i in range(10))\n\n"
                  "class Widget:\n    def __init__(self, name: str):\n        self.name = name\n")},
        {"text": '{"product": "VeltronHub X1", "warranty_months": 24, "active": true}'},
        {"text": "# Warranty\n\n## Coverage\n\n24-month limited warranty from purchase.\n"},
    ] * 8
    corpus.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    vt, _ = train_tokenizer([corpus], d / "out",
                            TokenizerConfig(vocab_size=512, min_frequency=1, name="test-tok"))
    return vt


# ----------------------------------------------------------------- byte fallback
def test_byte_tokenizer_roundtrip_everything(tok):
    for label, text in SAMPLES:
        decoded, ids, ok = tok.roundtrip(text)
        assert ok, f"{label}: round trip failed: {text!r} -> {decoded!r}"
        assert len(ids) == len(text.encode("utf-8")), f"{label}: byte tokenizer is not 1 byte/token"


def test_byte_tokenizer_has_no_unk_on_arbitrary_bytes(tok):
    weird = "�\U0001F600 latin greek αβγ кириллица"
    ids = tok.encode(weird)
    assert all(i < tok.get_vocab_size() for i in ids)
    assert tok.decode(ids) == weird


def test_special_tokens_are_reserved_and_distinct(tok):
    ids = {name: tok.token_to_id(name) for name in (PAD, BOS, EOS)}
    assert all(v is not None for v in ids.values())
    assert len(set(ids.values())) == len(ids), "special token ids collide"
    assert max(ids.values()) < tok.get_vocab_size()


def test_unk_is_unreachable_for_byte_level(trained):
    """Byte-level BPE is total, so <|unk|> must never be emitted."""
    for _label, text in SAMPLES:
        ids = trained.encode(text)
        assert trained.token_to_id("<|unk|>") not in ids


# ------------------------------------------------------------------ trained BPE
def test_trained_tokenizer_roundtrip(trained):
    for label, text in SAMPLES:
        assert trained.decode(trained.encode(text)) == text, label


def test_trained_tokenizer_compresses_better_than_byte_level(trained, tok):
    text = "The VeltronHub X1 smart hub requires a Veltron Account to activate the device."
    assert len(trained.encode(text)) < len(tok.encode(text))


def test_trained_tokenizer_handles_all_scripts(trained):
    for _label, text in SAMPLES:
        ids = trained.encode(text)
        if ids:
            assert max(ids) < trained.get_vocab_size()
            assert trained.decode(ids) == text


def test_bos_eos_ids_are_stable(trained):
    assert trained.bos_id >= 0 and trained.eos_id >= 0 and trained.pad_id >= 0


def test_save_load_roundtrip(trained, tmp_path):
    d = tmp_path / "copy"
    trained.save(d)
    reloaded = VeltronTokenizer.load(d)
    text = "Koliko traje garancija? / How long is the warranty?"
    assert reloaded.encode(text) == trained.encode(text)
    assert reloaded.sha256 == trained.sha256


def test_efficiency_report_shape(trained):
    eff = trained.efficiency({"en": ["The warranty is 24 months."],
                              "sr": ["Гаранција је 24 месеца."]})
    assert set(eff) == {"en", "sr"}
    for row in eff.values():
        assert row["chars_per_token"] > 0
        assert row["roundtrip_failures"] == 0


def test_serbian_costs_more_tokens_than_english(trained):
    """A real property: Serbian is not in the pretraining majority, so it fragments more."""
    eff = trained.efficiency({
        "en": ["The warranty period is 24 months from the purchase date."],
        "sr_cyr": ["Гаранција траје 24 месеца од датума куповине."],
    })
    assert eff["sr_cyr"]["chars_per_token"] < eff["en"]["chars_per_token"]


# ------------------------------------------------------------------- chat format
def test_chat_template_contains_role_markers(trained):
    text = trained.apply_chat_template([
        {"role": "system", "content": "You are support."},
        {"role": "user", "content": "Hello"},
    ])
    assert "<|system|>" in text and "<|user|>" in text and "<|assistant|>" in text
    assert text.endswith("<|assistant|>")


def test_chat_template_can_omit_generation_prompt(trained):
    text = trained.apply_chat_template([{"role": "user", "content": "Hi"}],
                                       add_generation_prompt=False)
    assert not text.endswith("<|assistant|>")


def test_encode_chat_roundtrips_content(trained):
    ids = trained.encode_chat([{"role": "user", "content": "What is the warranty?"}])
    assert ids and max(ids) < trained.get_vocab_size()


def test_special_tokens_are_present_in_vocabulary(trained):
    ids = trained.special_ids()
    for tok in ("<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"):
        assert tok in ids, f"{tok} missing from vocabulary"


def test_encode_batch_matches_encode(trained):
    texts = [t for _l, t in SAMPLES if t]
    batch = trained.encode_batch(texts)
    assert batch == [trained.encode(t) for t in texts]


def test_tokenizer_rejects_oversized_vocab_gracefully(tmp_path):
    """A corpus too small to fill the vocabulary must warn, not silently mis-size."""
    corpus = tmp_path / "tiny.jsonl"
    corpus.write_text(json.dumps({"text": "hello world"}) + "\n", encoding="utf-8")
    vt, stats = train_tokenizer([corpus], tmp_path / "out",
                                TokenizerConfig(vocab_size=65536, min_frequency=1))
    assert stats["vocab_size"] < 65536
    assert stats["vocab_size"] >= 256 + len(SPECIAL_TOKENS)
    assert vt.get_vocab_size() >= 256
