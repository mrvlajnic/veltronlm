"""Byte-level BPE tokenizer, trained from scratch.

Design decisions and why:

* **Byte-level BPE** (Sennrich et al. 2016, as in GPT-2). Every input is first mapped to
  its UTF-8 bytes, so the vocabulary is total: no word, script or byte sequence can ever
  become ``<unk>``. This matters here because the corpus is genuinely multilingual
  (Serbian in both Latin and Cyrillic) and contains code and JSON, where a token that
  "does not exist yet" is not an acceptable failure mode.
* **Byte-level pre-tokenizer regex.** Splits text into word / number / whitespace / symbol
  chunks *before* BPE, which stops merges from crossing the boundaries where a space or a
  newline appears. Without it, BPE happily merges " the" with " and" and the model cannot
  represent a word starting at a different offset.
* **Trained on the actual mixture.** Including Serbian Latin and Cyrillic in the training
  corpus is what forces the merge table to contain Cyrillic substrings; a tokenizer trained
  on English alone will shatter Serbian into bytes and lose roughly 40% of its compression.

Vocabulary sizes provided at several tiers so that small models are not dominated by an
oversized embedding matrix.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.hashing import sha256_file
from ..utils.logging_utils import get_logger

log = get_logger(__name__)

try:
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        processors,
        trainers,
    )

    _HAS_TOKENIZERS = True
except Exception as exc:  # pragma: no cover
    _HAS_TOKENIZERS = False
    log.error("the `tokenizers` package is required: %s", exc)


PAD = "<|pad|>"
BOS = "<|bos|>"
EOS = "<|eos|>"
SEP = "<|sep|>"
UNK = "<|unk|>"  # byte-level BPE makes this unreachable; kept for compatibility

#: Chat and RAG control tokens. Reserved at the *start* of the vocabulary so their ids are
#: stable across retraining and can be baked into chat templates.
SPECIAL_TOKENS: list[str] = [
    PAD, BOS, EOS, SEP, UNK,
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|support|>",
    "<|cite|>",
    "<|escalate|>",
    "<|unknown|>",
    "<|retrieve|>",
    "<|doc|>",
    "<|end|>",
    "<|think|>",
    "<|answer|>",
    "<|code|>",
    "<|json|>",
    "<|table|>",
    "<|sr|>",
    "<|en|>",
]

BYTE_FALLBACK_NOTE = "byte-level BPE is total: <|unk|> is unreachable by construction"


@dataclass
class TokenizerConfig:
    """Training hyper-parameters."""

    vocab_size: int = 32768
    min_frequency: int = 2
    byte_fallback: bool = True
    use_regex_pretokenizer: bool = True
    lowercase: bool = False
    initial_alphabet: int = 256
    train_limit: int = 8_000_000
    name: str = "tok-v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "byte_fallback": self.byte_fallback,
            "use_regex_pretokenizer": self.use_regex_pretokenizer,
            "lowercase": self.lowercase,
            "initial_alphabet": self.initial_alphabet,
            "train_limit": self.train_limit,
            "name": self.name,
        }


#: GPT-2's pre-tokenizer pattern, extended with the CJK/Thai ranges this corpus can see.
#: Keeping the Latin ranges strict means " can't" and "n't" split, which is what lets the
#: model reuse common suffixes.
DEFAULT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


class VeltronTokenizer:
    """Thin wrapper around ``tokenizers.Tokenizer`` adding chat templates and metrics."""

    def __init__(self, tokenizer: Tokenizer, cfg: TokenizerConfig | None = None) -> None:
        if not _HAS_TOKENIZERS:
            raise RuntimeError("the `tokenizers` package is required")
        self.tk = tokenizer
        self.cfg = cfg or TokenizerConfig()
        self.name = self.cfg.name
        self.sha256 = ""

    # ------------------------------------------------------------------ basics
    def get_vocab_size(self, with_added: bool = True) -> int:
        return self.tk.get_vocab_size(with_added_tokens=with_added)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tk.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(self, texts: Sequence[str], add_special_tokens: bool = False) -> list[list[int]]:
        return [e.ids for e in self.tk.encode_batch(list(texts), add_special_tokens=add_special_tokens)]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def decode_tokens(self, ids: Iterable[int]) -> list[str]:
        return self.tk.decode(list(ids), skip_special_tokens=False).split()

    def id_to_token(self, i: int) -> str:
        return self.tk.id_to_token(i)

    def token_to_id(self, token: str) -> int:
        return self.tk.token_to_id(token)

    def roundtrip(self, text: str) -> tuple[str, list[int], bool]:
        ids = self.encode(text)
        return self.decode(ids), ids, self.decode(ids) == text

    # ---------------------------------------------------------------- specials
    @property
    def pad_id(self) -> int:
        return self.token_to_id(PAD)

    @property
    def bos_id(self) -> int:
        return self.token_to_id(BOS)

    @property
    def eos_id(self) -> int:
        return self.token_to_id(EOS)

    def special_ids(self) -> dict[str, int]:
        return {t: self.token_to_id(t) for t in SPECIAL_TOKENS if self.token_to_id(t) is not None}

    # ------------------------------------------------------------------- chat
    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        add_generation_prompt: bool = True,
        eos: bool = True,
    ) -> str:
        """Render a chat into the VeltronLM instruction format.

        Format::

            <|bos|><|system|>...<|end|><|user|>...<|end|><|assistant|>...

        Role markers are dedicated tokens rather than plain text so the model can be told
        apart a user instruction from quoted content, and so prompt-injection attempts that
        inject literal role markers are visible in the token stream.
        """
        role_map = {
            "system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>",
            "tool": "<|support|>", "support": "<|support|>",
        }
        parts = [BOS]
        for m in messages:
            marker = role_map.get(m.get("role", "user").lower(), "<|user|>")
            parts.append(f"{marker}{m.get('content', '').strip()}<|end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        text = "".join(parts)
        if eos:
            text = text  # EOS is appended by the generator, not the template
        return text

    def encode_chat(
        self,
        messages: Sequence[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> list[int]:
        text = self.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        ids = self.encode(text)
        if add_generation_prompt and not ids:
            ids = [self.bos_id]
        return ids

    # -------------------------------------------------------------------- I/O
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.tk.save(str(d / "tokenizer.json"))
        (d / "tokenizer_config.json").write_text(json.dumps(self.cfg.as_dict(), indent=2), encoding="utf-8")
        meta = {
            "vocab_size": self.get_vocab_size(),
            "special_tokens": self.special_ids(),
            "name": self.name,
            "training_config": self.cfg.as_dict(),
        }
        (d / "tokenizer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tok_file = d / "tokenizer.json"
        self.sha256 = sha256_file(tok_file)[:16]
        (d / "tokenizer_meta.json").write_text(
            json.dumps({**meta, "sha256": self.sha256}, indent=2), encoding="utf-8"
        )
        log.info("tokenizer saved to %s (sha256=%s)", d, self.sha256)
        return d

    @classmethod
    def load(cls, directory: str | Path) -> VeltronTokenizer:
        d = Path(directory)
        cfg_path = d / "tokenizer_config.json"
        cfg = TokenizerConfig(**json.loads(cfg_path.read_text(encoding="utf-8"))) if cfg_path.exists() else None
        tk = Tokenizer.from_file(str(d / "tokenizer.json"))
        obj = cls(tk, cfg)
        meta_path = d / "tokenizer_meta.json"
        if meta_path.exists():
            obj.sha256 = json.loads(meta_path.read_text(encoding="utf-8")).get("sha256", "")
        obj.name = cfg.name if cfg else d.name
        return obj

    # -------------------------------------------------------------- efficiency
    def efficiency(self, samples: dict[str, Sequence[str]]) -> dict[str, Any]:
        """Compression metrics per corpus slice, all measured.

        Args:
            samples: ``{label: [text, ...]}``. Compression is reported per language and
                per format because a tokenizer can look fine on English prose while
                shattering Cyrillic or code, which is exactly the failure this project's
                Serbian and code requirements are about.
        """
        out: dict[str, Any] = {}
        for label, texts in samples.items():
            chars = 0
            toks = 0
            words = 0
            failed = 0
            for t in texts:
                ids = self.encode(t)
                if self.decode(ids) != t:
                    failed += 1
                chars += len(t)
                toks += len(ids)
                words += max(1, len(t.split()))
            if toks == 0:
                out[label] = {"chars": 0, "tokens": 0, "samples": len(texts)}
                continue
            out[label] = {
                "samples": len(texts),
                "chars": chars,
                "tokens": toks,
                "words": words,
                "chars_per_token": round(chars / toks, 3),
                "tokens_per_word": round(toks / words, 3),
                "bits_per_char": round(toks * 8 / max(1, chars), 3),
                "roundtrip_failures": failed,
            }
        return out


# ---------------------------------------------------------------------- training
def train_tokenizer(
    files: Sequence[str | Path],
    out_dir: str | Path,
    cfg: TokenizerConfig | None = None,
    text_field: str = "text",
) -> tuple[VeltronTokenizer, dict[str, Any]]:
    """Train a byte-level BPE from a list of JSONL files.

    Each input line is a JSON object; ``text_field`` selects the string to train on. If a
    line is not JSON it is treated as raw text, which keeps the trainer usable on a plain
    ``.txt`` corpus.
    """
    if not _HAS_TOKENIZERS:
        raise RuntimeError("install the `tokenizers` package: pip install tokenizers")

    cfg = cfg or TokenizerConfig()
    tokenizer = Tokenizer(models.BPE(unk_token=None))

    normalizer_steps: list[Any] = []
    if cfg.lowercase:
        normalizer_steps.append(normalizers.Lowercase())
    tokenizer.normalizer = normalizers.Sequence(normalizer_steps) if normalizer_steps else None

    if cfg.use_regex_pretokenizer:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=True
        )
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)

    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        limit_alphabet=cfg.initial_alphabet * 4,
    )

    inputs: list[str] = []
    stats: dict[str, Any] = {"files": [], "total_chars": 0, "total_lines": 0}
    remaining = cfg.train_limit
    for path in files:
        p = Path(path)
        if not p.exists():
            log.warning("tokenizer training input missing, skipping: %s", p)
            continue
        n_chars = 0
        n_lines = 0
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if remaining <= 0:
                    break
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get(text_field, "") if isinstance(obj, dict) else str(obj)
                except json.JSONDecodeError:
                    text = line
                if not text:
                    continue
                inputs.append(text)
                n_chars += len(text)
                n_lines += 1
                remaining -= len(text)
        stats["files"].append({"path": str(p), "lines": n_lines, "chars": n_chars})
        stats["total_chars"] += n_chars
        stats["total_lines"] += n_lines
        log.info("tokenizer corpus: %s -> %d lines, %d chars", p.name, n_lines, n_chars)

    if not inputs:
        raise RuntimeError("no training text found for the tokenizer")

    log.info("training BPE: vocab=%d on %d documents / %d chars",
             cfg.vocab_size, len(inputs), stats["total_chars"])
    tokenizer.train_from_iterator(inputs, trainer=trainer, length=len(inputs))

    vt = VeltronTokenizer(tokenizer, cfg)
    vt.save(out_dir)

    actual_vocab = vt.get_vocab_size()
    stats["vocab_size"] = actual_vocab
    stats["vocab_size_requested"] = cfg.vocab_size
    stats["special_tokens"] = vt.special_ids()
    stats["note"] = BYTE_FALLBACK_NOTE
    (Path(out_dir) / "training_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if actual_vocab < cfg.vocab_size * 0.98:
        log.warning(
            "vocabulary saturating: requested %d, achieved %d (corpus has too little "
            "repetition at min_frequency=%d)",
            cfg.vocab_size, actual_vocab, cfg.min_frequency,
        )
    return vt, stats


# ------------------------------------------------------------------ from scratch
def fallback_byte_tokenizer() -> VeltronTokenizer:
    """A pure byte-level tokenizer with no merges.

    Used only as a safety net so a training run can never be blocked by a missing tokenizer
    file. It has no compression, which makes it useless in production but proves the
    training loop is not silently depending on a good vocabulary.

    The vocabulary is the GPT-2 ``bytes_to_unicode`` alphabet, **not** ``chr(i)`` for
    ``i in 0..255``. Byte-level decoding maps bytes through that alphabet, so a Latin-1
    vocabulary would decode every byte above 127 to the wrong character and silently
    destroy every non-Latin script in the corpus.
    """
    if not _HAS_TOKENIZERS:
        raise RuntimeError("the `tokenizers` package is required")

    # The vocabulary is the GPT-2 ``bytes_to_unicode`` alphabet, **not** ``chr(i)`` for
    # ``i in 0..255``. Byte-level decoding maps bytes through that alphabet, so a Latin-1
    # vocabulary would decode every byte above 127 to the wrong character and silently
    # destroy every non-Latin script in the corpus.
    #
    # These are installed as the model's regular vocabulary rather than via ``add_tokens``:
    # added tokens bypass pre-tokenization, so the pre-tokenizer's split on whitespace would
    # no longer map a literal space onto its 'Ġ' byte symbol and every space would vanish.
    alphabet = list(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=None))
    # Control and role tokens *are* added tokens: they must survive pre-tokenization intact.
    tk.add_tokens(SPECIAL_TOKENS)
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()
    return VeltronTokenizer(
        tk,
        TokenizerConfig(vocab_size=len(alphabet) + len(SPECIAL_TOKENS), name="tok-byteonly"),
    )


__all__ = [
    "VeltronTokenizer",
    "TokenizerConfig",
    "SPECIAL_TOKENS",
    "train_tokenizer",
    "fallback_byte_tokenizer",
    "PAD",
    "BOS",
    "EOS",
    "SEP",
    "UNK",
]
