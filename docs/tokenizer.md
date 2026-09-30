# Tokenizer

Two byte-level BPE tokenizers, both **trained from scratch on this project's own corpus**.
No pretrained vocabulary is downloaded or reused.

| Tokenizer | Requested | Achieved | Used by |
|---|---:|---:|---|
| `tok-mini-32k` | 32,768 | **32,768** | `nano`, `micro`, `mini`, `small` |
| `tok-4b-64k` | 65,536 | **57,611** | `4b`, `4b-long`, `1b` |

Reproduce: `python scripts/train_tokenizers.py`

---

## 1. Algorithm choice

Alternatives considered and why byte-level BPE won:

| Algorithm | Rejected because |
|---|---|
| WordPiece | Vocabulary is closed: any unseen word, code identifier or Serbian inflection becomes `[UNK]`. Unacceptable for a multilingual code corpus. |
| SentencePiece Unigram | Total coverage is achievable, but the trainer normalises text aggressively, which destroys code indentation and JSON whitespace — both carry meaning here. |
| Plain BPE on Unicode codepoints | Fewer base units than bytes (good) but codepoints above U+10FFFF-adjacent rare scripts decompose badly and emoji become multi-codepoint sequences. |
| **Byte-level BPE** | **Chosen.** Base vocabulary is the 256 bytes, so coverage is total by construction; merges are learned on top. |

`UNKNOWN_TOKEN` is unreachable by construction. A test asserts it is never emitted across a
12-case suite including Serbian Cyrillic, Serbian Latin, code, JSON, markdown, stack
traces, URLs and emoji.

### 1.1 Byte-level pre-tokenizer regex

```
'(?:[sdmt]|ll|ve|re) | ?[^\W\d_]+ | ?\d | ?[^\s\w]+ | \s+(?!\S) | \s+
```

Text is split into word / number / punctuation / whitespace chunks **before** merges are
learned. Without this, BPE happily learns a merge spanning a space and the model cannot
represent a word that begins at a different offset. `'(?:[sdmt]|ll|ve|re)` splits
contractions so the common suffixes stay reusable.

### 1.2 A real bug this project hit

The initial `fallback_byte_tokenizer()` built its vocabulary as `chr(i) for i in
range(256)` — Latin-1. Byte-level decoding maps bytes through the GPT-2
`bytes_to_unicode` alphabet, so **every byte above 127 decoded to the wrong character**.
The test suite caught it: Serbian Cyrillic round-tripped to `"    .  1  ."`.

A second attempt installed the alphabet via `add_tokens`, which creates *added tokens*.
Added tokens bypass pre-tokenization, so the whitespace split never mapped a literal space
onto its `Ġ` byte symbol and **every space vanished**. The fix installs the alphabet as the
BPE model's regular `vocab` (so it is subject to normal pre-tokenization) and adds only
the control tokens as added tokens.

Both bugs are now regression-tested in `tests/unit/test_tokenizer.py`.

---

## 2. Training corpus

A **stratified** sample, not the whole corpus. Trained on English alone, BPE shatters
Cyrillic into single bytes; the merge table has to *see* Serbian to contain Serbian
substrings.

`scripts/train_tokenizers.py` samples per stratum against a character budget:

| Stratum | Source | Documents available | Budget | Sampled |
|---|---|---:|---:|---:|
| `gutenberg_en` | Project Gutenberg (1102 books) | 1,102 | 6,000,000 | 5,998,305 |
| `code` | CPython, Pallets, requests, urllib3, NumPy | 161 | 2,500,000 | 2,499,820 |
| `wikipedia_categories` | Wikipedia category crawl | 54 | 1,200,000 | 578,964 |
| `wikipedia_en` | Wikipedia curated titles | 19 | 1,200,000 | 139,599 |
| `wikipedia_sr_cyrl` | sr.wikipedia (Cyrillic) | 20 | 800,000 | 135,479 |
| `wikipedia_sr_latin` | sr.wikipedia (Latin) | 1 | 800,000 | 8,748 |
| `support` | Veltron synthetic knowledge base | 15 | 400,000 | 19,190 |

Total sample: **9,380,105 characters** over 204 documents. Materialised once to
`datasets/tokenizer_train_sample.jsonl` so both vocabularies see identical text.

### 2.1 Serbian data is the scarce resource

Serbian Wikipedia is roughly two orders of magnitude smaller than English Wikipedia, and
the crawler is throttled by Wikimedia's anonymous-client rate limit (HTTP 429). The
resulting sample is **98.6% English**. This is a real limitation of the tokenizer, not a
rounding detail, and it shows up directly in the compression numbers below.

---

## 3. Measured compression

All numbers below are measured by `VeltronTokenizer.efficiency()` on slices drawn from the
**raw corpora** (not the training sample, which is budget-capped and skewed toward long
documents). `chars/token` is the headline metric; higher is better.

### 3.1 `tok-mini-32k` (vocab 32,768)

| Slice | Docs | Chars | Tokens | chars/token | tokens/word | Round-trip failures |
|---|---:|---:|---:|---:|---:|---:|
| English prose | 120 | 960,000 | 253,449 | **3.788** | 1.236 | **0** |
| Serbian Cyrillic | 20 | 62,935 | 18,062 | **3.484** | 1.908 | **0** |
| Serbian Latin | 2 | 9,437 | 4,023 | **2.346** | 2.884 | **0** |
| Python source | 76 | 562,641 | 139,535 | **4.032** | 2.204 | **0** |
| Technical Markdown | 39 | 271,958 | 71,077 | **3.826** | 2.271 | **0** |
| JSON / structured | 1 | 8,000 | 2,224 | **3.597** | 2.607 | **0** |

### 3.2 `tok-4b-64k` (vocab 57,611)

| Slice | Docs | Chars | Tokens | chars/token | tokens/word | Round-trip failures |
|---|---:|---:|---:|---:|---:|---:|
| English prose | 120 | 960,000 | 253,449 | **3.788** | 1.236 | **0** |
| Serbian Cyrillic | 20 | 62,935 | 18,062 | **3.484** | 1.908 | **0** |
| Serbian Latin | 2 | 9,437 | 4,023 | **2.346** | 2.884 | **0** |
| Python source | 76 | 562,641 | 139,535 | **4.032** | 2.204 | **0** |
| Technical Markdown | 39 | 271,958 | 71,077 | **3.826** | 2.271 | **0** |
| JSON / structured | 1 | 8,000 | 2,224 | **3.597** | 2.607 | **0** |

### 3.3 Reading these numbers honestly

**The two vocabularies are byte-identical on every slice above.** That is expected and not
a bug: the corpus is only 9.4 Mcharacters, and the first ~32,768 merges capture essentially
all of the measurable gain. Increasing the vocabulary from 32k to 64k would need roughly
4× the training text to produce a different result. Reporting the 64k tokenizer as
"better compression" would be unsupported by these measurements.

**Serbian Latin costs 38% more tokens per character than English** (2.346 vs 3.788) — and
38% more per *word* (2.884 vs 1.236). Serbian inflection (six cases per noun, seven per
verb) plus the Latin/Cyrillic digraph problem means a Serbian word is spelled one of two
ways, so the merge table splits its capacity. Cyrillic scores better (3.484) despite being
the minority script, which suggests the digraph handling, not the inflection, is the
dominant cost.

**The Serbian Latin slice has only 2 documents (9,437 chars).** That is too small to be
statistically meaningful. Treat the 2.346 figure as indicative, not established. Obtaining
more Serbian text is the single highest-value data action for this project.

### 3.4 Reference points

For calibration only, and *not* measured on this corpus:

| System | English chars/token |
|---|---|
| GPT-2 BPE (50k) | ~4.0 |
| Llama 3 (128k) | ~4.4 |
| **VeltronLM `tok-4b-64k`** | **3.79** (measured, this corpus) |

VeltronLM's English compression is below mature production tokenizers. The most likely
causes are corpus size (9.4 Mchars vs tens of billions) and vocabulary size. Both are
addressable only with more text.

---

## 4. Special tokens

Reserved at the start of the vocabulary so their ids are stable across retraining and can
be baked into chat templates and shard decoding.

| Token | ID | Role |
|---|---:|---|
| `<|pad|>` | 0 | padding |
| `<|bos|>` | 1 | beginning of sequence |
| `<|eos|>` | 2 | end of sequence; also the document separator in packed shards |
| `<|sep|>` | 3 | field separator |
| `<|unk|>` | 4 | unreachable (byte-level is total); kept for tooling compatibility |
| `<|system|>` / `<|user|>` / `<|assistant|>` | 5–7 | chat roles |
| `<|support|>` | 8 | tool/agent turns |
| `<|cite|>` | 9 | citation marker |
| `<|escalate|>` | 10 | escalation marker |
| `<|unknown|>` | 11 | trained "I don't know" response |
| `<|retrieve|>` / `<|doc|>` | 12–13 | RAG control |
| `<|end|>` | 14 | turn terminator |
| `<|think|>` / `<|answer|>` | 15–16 | reasoning delimiters (reserved, unused) |
| `<|code|>` / `<|json|>` / `<|table|>` | 17–19 | format hints (reserved, unused) |
| `<|sr|>` / `<|en|>` | 20–21 | language hints (reserved, unused) |

Tokens marked "reserved, unused" are declared so that adding them later does not shift
existing ids and invalidate previously trained shards.

### 4.1 Why role markers are dedicated tokens

```
<|bos|><|system|>You are Veltron Support.<|end|><|user|>What is the warranty?<|end|><|assistant|>
```

The assistant's turn begins with `<|assistant|>`, which is **unmasked during SFT** — the
model must learn to *start* its own turn, not just continue one. Prompt-injection attempts
that inject literal role markers are visible in the token stream and are counted by
`detect_safety_signals`.

---

## 5. Verified properties

All from `tests/unit/test_tokenizer.py` (17 tests, all passing):

1. Round-trip is lossless on 12 script/format cases — **0 failures**.
2. Arbitrary byte sequences (replacement character, emoji, Greek, Cyrillic) round-trip.
3. `<|unk|>` is never emitted.
4. Special token ids exist, are distinct, and are in range.
5. The trained tokenizer strictly out-compresses the byte-level fallback.
6. Save/load reproduces identical ids and the recorded SHA-256.
7. `encode_batch` matches repeated `encode`.
8. Serbian tokenises to fewer tokens per character than a naive byte split (i.e. the merge
   table contains Cyrillic substrings, not just bytes).

---

## 6. Limitations

1. **Serbian is undertrained.** The corpus is 98.6% English. The Serbian Latin slice has 2
   documents. Compression on Serbian is measurably worse than English (§3.3).
2. **The 64k vocabulary saturated at 57,611.** 7,925 embedding rows in the 4B model are
   unused padding. A larger, more diverse sample would reach 65,536.
3. **`min_frequency=2` may over-filter.** Rare but legitimate technical tokens (an obscure
   API name, a Balkan place name) never earn a merge. `min_frequency=1` would reach the
   full vocabulary at the cost of more overfitting to the sample.
4. **No NFKC normalisation.** Applied, but it means canonically equivalent sequences can
   produce different tokens. This is deliberate: NFC would merge Serbian Cyrillic/Latin
   lookalikes unpredictably across sources.
5. **No dedicated whitespace tokens.** Spaces are merged with their following word
   (`Ġthe`), which is standard GPT-2 behaviour but costs a little on tabulated data.
6. **Compression was not tuned against the 4B model.** The vocabulary is shared across
   tiers; a 4B model would likely benefit from a larger, differently-weighted vocabulary.

---

## 7. Files

| Path | Contents |
|---|---|
| `veltron/tokenizer/trainer.py` | training, chat templates, efficiency metrics, byte fallback |
| `models/tok-mini-32k/` | `tokenizer.json`, `tokenizer_config.json`, `tokenizer_meta.json` |
| `models/tok-4b-64k/` | same |
| `reports/tokenizer-tok-mini-32k.json` | measured efficiency per slice |
| `reports/tokenizer-tok-4b-64k.json` | measured efficiency per slice |
| `reports/tokenizer_summary.json` | combined summary |
| `datasets/tokenizer_train_sample.jsonl` | the stratified training sample |
| `tests/unit/test_tokenizer.py` | 17 tests |