"""Data pipeline: normalisation, language ID, PII, quality, dedup, packing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from veltron.data.filters import (
    Deduper,
    FilterStats,
    detect_format,
    detect_language,
    detect_pii,
    exact_hash,
    minhash,
    normalize,
    pii_verdict,
    process_document,
    quality_report,
    redact_pii,
)
from veltron.data.loader import PackedDataset, TokenShardLoader, read_index
from veltron.data.pipeline import (
    DEFAULT_MIXTURE,
    MixtureComponent,
    mix_documents,
    normalize_mixture,
    pack_documents_list,
    split_documents,
    write_shards,
)
from veltron.finetuning.data import is_boilerplate_heavy


# ---------------------------------------------------------------- normalisation
def test_normalize_strips_control_characters():
    assert "\x00" not in normalize("a\x00b")
    assert "\x07" not in normalize("bell\x07")


def test_normalize_collapses_whitespace_but_keeps_code_indent():
    text = "def f():\n        return 1   \n\n\n\n\nreturn 2"
    out = normalize(text)
    assert out.count("\n\n\n\n") == 0
    assert "        return 1" in out


def test_normalize_is_idempotent():
    once = normalize("Some  text\r\nwith\tcrlf\r")
    assert normalize(once) == once


# ---------------------------------------------------------------- language ID
@pytest.mark.parametrize(
    "text,expected",
    [
        ("The quick brown fox jumps over the lazy dog and it is not a problem", "en"),
        ("Вештачка интелигенција је грана рачунарства која се бави", "sr"),
        ("Ovo je rečenica na srpskom jeziku koja ima ćčžšđ", "sr"),
        ("这是一段中文文本内容用于测试语言检测", "zho"),
        ("", "und"),
        ("x", "und"),
    ],
)
def test_detect_language(text, expected):
    lang, _conf = detect_language(text)
    assert lang == expected, f"{text!r} -> {lang}, expected {expected}"


def test_detect_language_returns_confidence_in_range():
    for _l, t in SAMPLES:
        _lang, conf = detect_language(t)
        assert 0.0 <= conf <= 1.0


SAMPLES = [
    ("en", "The VeltronHub X1 requires an account and it is not difficult to set up."),
    ("sr", "ВелтронХуб КС1 захтева налог и није тешко га подесити."),
]


# ---------------------------------------------------------------------- PII
PII_CASES = [
    ("email", "Contact me at anna.kowalski@example.com for details."),
    ("aws_key", "AKIAIOSFODNN7EXAMPLE is the key."),
    ("github_token", "Use ghp_1234567890abcdefghijklmnopqrstuvwxyzAB to clone."),
    ("jwt", "Token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghij"),
    ("us_ssn", "SSN 123-45-6789 on file."),
    ("iban", "IBAN DE89370400440532013000 please."),
    ("private_key", "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----"),
    ("phone_sr", "Call me on +381641234567 tomorrow."),
]


#: Patterns that identify a specific account or machine and are never legitimate in
#: pretraining text. Documents containing one of these are dropped outright.
DROP_ALWAYS = ("aws_key", "github_token", "jwt", "us_ssn", "iban", "private_key")
#: Contact details a support log may legitimately contain, so they are redacted instead.
REDACTABLE = ("email", "phone_sr")


@pytest.mark.parametrize("name,text", PII_CASES)
def test_detect_pii_finds_each_pattern(name, text):
    counts = detect_pii(text)
    assert name in counts, f"{name} not detected in {text!r}; found {counts}"


@pytest.mark.parametrize("name,text", [(n, t) for n, t in PII_CASES if n in DROP_ALWAYS])
def test_pii_verdict_drops_on_credentials(name, text):
    verdict, counts = pii_verdict(text)
    assert verdict == "drop", f"{name} should drop, got {verdict}"
    assert counts


@pytest.mark.parametrize("name,text", [(n, t) for n, t in PII_CASES if n in REDACTABLE])
def test_pii_verdict_redacts_ordinary_contact_details(name, text):
    """One email in a support thread is normal and is redacted, not dropped."""
    verdict, _ = pii_verdict(text)
    assert verdict == "redact", f"{name} should redact, got {verdict}"


def test_pii_verdict_drops_bulk_identifier_dump():
    """Three or more contacts is not a support log, it is a leaked export."""
    text = "Emails: a@x.com, b@y.com, c@z.com, d@w.com, e@v.com"
    verdict, _ = pii_verdict(text)
    assert verdict == "drop"
    assert "a@x.com" not in redact_pii(text)


def test_redaction_removes_secret_material():
    out = redact_pii("Key AKIAIOSFODNN7EXAMPLE and mail a@b.com")
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "a@b.com" not in out


def test_clean_text_has_no_pii():
    text = "Contact anna.kowalski@example.com or call +381641234567"
    verdict, _ = pii_verdict(text)
    assert verdict in ("redact", "drop")
    assert not detect_pii(redact_pii(text))


# ------------------------------------------------------------------- quality
def test_quality_report_rejects_short_text():
    assert quality_report("too short")["ok"] is False


def test_quality_scores_real_prose_above_boilerplate():
    prose = ("The VeltronHub X1 connects to your home network over Gigabit Ethernet and "
             "supports dual-band Wi-Fi 6. It requires a Veltron Account to activate, which "
             "you can create from the companion mobile application. Activation takes about "
             "two minutes once the device is powered on. ") * 4
    spam = "BUY NOW CHEAP BEST OFFER CLICK HERE " * 60
    assert quality_report(prose)["score"] > quality_report(spam)["score"]


def test_quality_flags_link_stuffing():
    text = ("word " * 200) + ("https://spam.example " * 60)
    rep = quality_report(text)
    assert "link_stuffing" in rep["issues"]


# --------------------------------------------------------------------- dedup
def test_exact_hash_detects_identical_text():
    assert exact_hash("hello world") == exact_hash("  hello world  ")


def test_minhash_is_deterministic_across_calls():
    text = "The VeltronHub X1 requires an account to activate the device safely."
    a = minhash(text)
    b = minhash(text)
    assert np.array_equal(a, b)


def test_minhash_of_similar_texts_is_close():
    base = ("The VeltronHub X1 requires a Veltron Account to activate the device. "
            "It ships with the Veltron app for iOS and Android. ") * 3
    near = base + "The warranty period is 24 months from purchase."
    far = "Serbian language is a South Slavic language spoken in Serbia and neighbouring countries."
    s_base, s_near, s_far = minhash(base), minhash(near), minhash(far)
    est_base_near = float((s_base == s_near).mean())
    est_base_far = float((s_base == s_far).mean())
    assert est_base_near > est_base_far


def test_deduper_flags_exact_duplicates():
    d = Deduper()
    text = "A support paragraph about warranty coverage that is long enough to shingle." * 3
    assert d.check_and_add(text) == (False, "unique")
    assert d.check_and_add(text)[0] is True


def test_deduper_flags_near_duplicates():
    d = Deduper()
    base = ("The VeltronHub X1 requires a Veltron Account to activate the device. "
            "It ships with the Veltron app for iOS and Android. ") * 4
    d.check_and_add(base)
    dup, reason = d.check_and_add(base + " Additional trailing sentence.")
    assert dup is True and "near" in reason


def test_deduper_keeps_distinct_documents():
    d = Deduper()
    a = d.check_and_add("Warranty coverage lasts 24 months from the date of purchase receipt.")
    b = d.check_and_add("Dispatch to Serbia happens within one business day of the order.")
    assert a[0] is False and b[0] is False
    assert d.stats()["kept"] == 2


def test_deduper_stats_are_self_consistent():
    d = Deduper()
    for i in range(5):
        d.check_and_add(f"Distinct support sentence number {i} about topic {i * 7}.")
    d.check_and_add("Distinct support sentence number 0 about topic 0.")
    s = d.stats()
    assert s["kept"] + s["duplicate_exact"] + s["duplicate_near"] == s["documents_seen"]


# --------------------------------------------------------------------- process
def test_process_document_accepts_clean_text():
    text = ("The VeltronHub X1 is a smart home controller. It supports Wi-Fi 6, Gigabit "
            "Ethernet, Matter, Zigbee and Z-Wave. The device measures 118 by 78 by 26 "
            "millimetres and weighs 240 grams. ") * 3
    stats = FilterStats()
    out = process_document({"text": text, "source": "t", "license": "x"}, stats, None)
    assert out is not None
    assert out["format"] == "prose"
    # chars reflects the *normalized* text, which is what actually reaches the tokenizer.
    assert out["chars"] == len(normalize(text))
    assert out["text"] == normalize(text)
    assert stats.accepted == 1


def test_process_document_rejects_too_short():
    stats = FilterStats()
    assert process_document({"text": "hi"}, stats, None) is None
    assert stats.stages.get("too_short") == 1


def test_process_document_drops_pii_documents():
    text = ("Our support policy states that we will help every customer. "
            "AKIAIOSFODNN7EXAMPLE is the internal key we use. ") * 3
    stats = FilterStats()
    assert process_document({"text": text, "source": "t"}, stats, None) is None
    assert "pii_drop" in stats.stages


def test_filter_stats_report():
    stats = FilterStats()
    stats.seen = 10
    stats.accepted = 6
    stats.reject("too_short")
    stats.reject("pii_drop")
    d = stats.as_dict()
    assert d["seen"] == 10 and d["accepted"] == 6
    assert d["accept_rate"] == 0.6
    assert d["rejected_by_stage"]["too_short"] == 1


def test_detect_format():
    assert detect_format("def f():\n    return 1\n") == "code"
    assert detect_format('{"a": 1}') == "json"
    assert detect_format("# Heading\n\ntext here") == "markdown"
    assert detect_format("A normal sentence of ordinary prose goes here.") == "prose"


def test_is_boilerplate_heavy():
    clean = "The device requires a Veltron Account to activate. " * 20
    dirty = ("Credit for this e-text: Project Gutenberg " + "[Illustration] ") * 30
    assert not is_boilerplate_heavy(clean)
    assert is_boilerplate_heavy(dirty)


# ------------------------------------------------------------------- mixture
def test_normalize_mixture_sums_to_one():
    mix = normalize_mixture([
        MixtureComponent("a", 2.0, "x", ""),
        MixtureComponent("b", 2.0, "y", ""),
    ])
    assert abs(sum(c.weight for c in mix) - 1.0) < 1e-9


def test_default_mixture_is_normalized():
    assert abs(sum(c.weight for c in normalize_mixture(DEFAULT_MIXTURE)) - 1.0) < 1e-9


def test_mix_documents_hits_target_weights():
    docs = []
    for cat, n in (("general", 200), ("code", 50), ("support", 10)):
        for i in range(n):
            docs.append({"text": f"{cat} doc {i} " * 40, "category": cat,
                         "source": "s", "language": "en"})
    mix = [MixtureComponent("general", 0.5, "g", ""),
           MixtureComponent("code", 0.3, "c", ""),
           MixtureComponent("support", 0.2, "s", "")]
    mixed, report = mix_documents(docs, mix, seed=1)
    assert len(mixed) == len(docs)
    got = report["achieved_final"]
    assert abs(got["general"] - 0.5) < 0.02
    assert abs(got["support"] - 0.2) < 0.02


def test_mix_documents_reports_upsampling():
    docs = [{"text": "support doc " * 60, "category": "support", "source": "s",
             "language": "en"} for _ in range(2)]
    docs += [{"text": "general doc " * 60, "category": "general", "source": "s",
              "language": "en"} for _ in range(100)]
    mix = [MixtureComponent("general", 0.5, "g", ""),
           MixtureComponent("support", 0.5, "s", "")]
    _mixed, report = mix_documents(docs, mix, seed=1)
    assert report["upsampled"]["support"] > 0


def test_split_keeps_documents_intact():
    docs = [{"text": f"doc {i} " * 30, "category": "general", "source": "s",
             "language": "en"} for i in range(2000)]
    train, val, test = split_documents(docs, seed=7)
    assert len(train) + len(val) + len(test) == len(docs)
    ids = {id(d) for d in train} | {id(d) for d in val} | {id(d) for d in test}
    assert len(ids) == len(docs), "a document appears in more than one split"
    assert val and test


# -------------------------------------------------------------------- packing
def test_pack_documents_concatenates_with_eos():
    class FakeTok:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 100 for c in text]

    docs = [{"text": "abcdefgh"}, {"text": "ijklmnop"}]
    shards = pack_documents_list(docs, FakeTok(), eos_id=2, max_tokens=100)
    assert len(shards) == 1
    assert shards[0].dtype == np.uint32
    flat = list(shards[0])
    assert 2 in flat, "EOS separator missing"


def test_pack_documents_respects_max_tokens():
    class FakeTok:
        def encode(self, text, add_special_tokens=False):
            return [1] * 100

    docs = [{"text": "x" * 100} for _ in range(10)]
    shards = pack_documents_list(docs, FakeTok(), eos_id=2, max_tokens=256)
    assert all(s.size <= 256 for s in shards)
    assert sum(s.size for s in shards) == 10 * 101


def test_write_shards_records_actual_dtype(tmp_path):
    arr = np.arange(100, dtype=np.uint16)
    info = write_shards([arr], tmp_path / "train", "train")
    assert info["dtype"] == "uint16"
    index = json.loads((tmp_path / "train" / "train-index.json").read_text())
    assert index["dtype"] == "uint16"
    assert index["total_tokens"] == 100


def test_write_shards_rejects_mixed_dtypes(tmp_path):
    with pytest.raises(ValueError):
        write_shards([np.arange(4, dtype=np.uint16), np.arange(4, dtype=np.uint32)],
                     tmp_path / "t", "t")


# --------------------------------------------------------------------- loader
def test_packed_dataset_reads_windows(tmp_path):
    arr = np.arange(5000, dtype=np.uint32)
    write_shards([arr], tmp_path / "train", "train")
    ds = PackedDataset(tmp_path / "train", seq_len=128, prefix="train")
    assert len(ds) == 5000 - 128 + 1
    item = ds[0]
    assert item["input_ids"].shape == (128,)
    assert item["labels"][0].item() == item["input_ids"][1].item()


def test_loader_rejects_dtype_mismatch_with_data(tmp_path):
    """A wrong index dtype must fail loudly, not reinterpret every token."""
    arr = np.arange(5000, dtype=np.uint16)
    write_shards([arr], tmp_path / "train", "train")
    idx = tmp_path / "train" / "train-index.json"
    payload = json.loads(idx.read_text())
    payload["dtype"] = "uint32"          # deliberately wrong
    idx.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="bytes/token"):
        PackedDataset(tmp_path / "train", seq_len=128, prefix="train")


def test_token_shard_loader_yields_batches(tmp_path):
    arr = np.arange(5000, dtype=np.uint32)
    write_shards([arr], tmp_path / "train", "train")
    ds = PackedDataset(tmp_path / "train", seq_len=64, prefix="train")
    loader = TokenShardLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (4, 64)
    assert batch["labels"].shape == (4, 64)
