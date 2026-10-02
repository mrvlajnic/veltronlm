# Datasets

Data is versioned, hashed, and traceable back to a licensed source for every document. The
build is a pure function of (raw corpus, tokenizer, mixture, seed), so the same inputs
reproduce the same manifest hash.

Current version: **`dataset-v1`** (manifest hash `272e134cf74c7367`).

Reproduce: `python scripts/build_dataset.py --seq-len 1024`

---

## 1. Source register

Licences are checked **before** download. A corpus that is not in
`veltron/data/sources.py` cannot be fetched by the acquisition pipeline.

| Source | Licence | Commercial use | Redistribution | Category |
|---|---|---|---|---|
| `wikipedia-en` | CC-BY-SA-4.0 | permitted | share-alike | general |
| `wikipedia-sr` | CC-BY-SA-4.0 | permitted | share-alike | multilingual |
| `wikipedia-random` | CC-BY-SA-4.0 | permitted | share-alike | general |
| `wikipedia-cat-{en,sr,sh}` | CC-BY-SA-4.0 | permitted | share-alike | general / multilingual |
| `wikisource` | CC-BY-SA-4.0 + public-domain source texts | permitted | share-alike | general |
| `gutenberg` | Public Domain | permitted | permitted | general |
| `cpython-stdlib` | PSF-2.0 | permitted | permitted | code |
| `permissive-repos` | MIT / Apache-2.0 / BSD-3-Clause per repo | permitted | permitted | code |
| `pydoc` | PSF-2.0 | permitted | permitted | technical |
| `veltron-synthetic-kb` | CC0-1.0 (project-authored) | permitted | permitted | support |

`DATA_SOURCES.md` is the citation-facing version of this table.

### 1.1 GPL exclusion

`git/git`'s `Documentation/technical/api-simple.txt` was originally listed and is **skipped
at fetch time** with an explicit log line. Mixing GPL-2.0 text into the corpus would
impose copyleft on the model weights as a derivative work in most jurisdictions. The
exclusion is in code, not just in a comment:

```python
if rf.spdx == "GPL-2.0-only":
    log.info("skipping GPL-2.0-only file %s/%s", rf.repo, path)
    continue
```

### 1.2 Code corpus, per-file provenance

194 files across CPython stdlib (PSF-2.0), Pallets Flask/Click/Werkzeug (BSD-3-Clause),
`requests` (Apache-2.0), `urllib3` (MIT), NumPy (BSD-3-Clause), plus technical READMEs
from Rust, Node.js, Go, Black and Werkzeug. **Every individual file carries its own SPDX
identifier** in `RepoFile`, so the licence of every downloaded byte is known, not inferred
from the repository.

---

## 2. Acquisition

| Script | Purpose |
|---|---|
| `scripts/acquire_data.py` | curated Wikipedia titles, 194 code files, Gutenberg IDs, support KB |
| `scripts/fetch_gutenberg_bulk.py` | derives book IDs from the real `pg_catalog.csv` (79,474 entries) |
| `scripts/crawl_wikipedia_categories.py` | category traversal for topical balance |

Raw corpus as acquired:

| File | Documents | Size |
|---|---:|---:|
| `gutenberg.jsonl` | 1,102 | 44.9 MB |
| `code_technical.jsonl` | 161 | 6.08 MB |
| `wikipedia.jsonl` | 40 | 0.39 MB |
| `wikipedia_categories.jsonl` | 17+ | 0.22 MB |
| `support_kb.jsonl` | 15 | 0.02 MB |
| **Total** | **~1,335** | **~510 MB** |

### 2.1 Why the catalogue is queried rather than hard-coded

`fetch_gutenberg_bulk.py` reads `pg_catalog.csv`, filters by subject (preferring fiction,
history, science, philosophy over one-page pamphlets and sermons), sorts by descending text
length and takes a deterministic slice. This is reproducible for a given `--limit` and gets
10× more usable text than a hand-maintained ID list.

---

## 3. Quality pipeline

Every stage is a pure function, and every stage records how many documents it removed and
why. The funnel is a real measurement, not a design intention.

```
raw JSONL
  ↓ normalize (NFKC, control chars stripped, code indentation preserved)
  ↓ boilerplate strip (cookie banners, "прихвате све колачиће", captions)
  ↓ PII verdict: keep / redact / drop
  ↓ quality report (type-token ratio, sentence length, link stuffing, char runs)
  ↓ exact dedup (SHA-256 of first 4000 chars)
  ↓ near dedup (MinHash 64-perm, LSH 16 bands x 4 rows)
  ↓ language tagging (script ratios + stopword discrimination)
  ↓ format classification (prose / code / json / markdown / markup)
  ↓ mixture resampling
  ↓ document-level train/val/test split
  ↓ tokenization and packing
  ↓ uint16/uint32 shards + index
```

### 3.1 Measured funnel for `dataset-v1`

```
documents seen                              1,380
documents kept                              1,312
characters kept                        383,615,628

rejected by stage
  pii_drop                                       56
  too_short                                       1
  bad_json_line                                   1
  dedup_exact                                     1
  dedup_near_lsh_band0                            2
  dedup_near_lsh_band1                            2
  dedup_near_lsh_band4                            2
  dedup_near_lsh_band6                            1
  dedup_near_lsh_band8                            1
  dedup_near_lsh_band12                           1
  dedup_near_lsh_band13                           1
  ----------------------------------------------
  total rejected                                  68
  accept rate                                95.1%
```

`pii_drop` at 4.1% is higher than a hand-written corpus would produce and is dominated by
credit-card-shaped digit runs in 19th-century financial texts.

### 3.2 PII policy

Implemented in `pii_verdict`. The split between *drop* and *redact* is deliberate:

| Category | Threshold | Action | Rationale |
|---|---|---|---|
| Private keys, AWS keys, GitHub/Slack tokens, JWTs | ≥ 1 | **drop** | A leaked credential is never legitimate training data. |
| Credit cards, IBANs, US SSNs | ≥ 1 | **drop** | Unique financial identifiers. |
| Email addresses | ≥ 3 | **drop** | Three or more is a leaked export, not a support log. |
| Email addresses | 1–2 | **redact** | A support thread legitimately contains a customer's address. |
| Phone numbers | ≥ 4 | **drop** | Same reasoning. |
| IPv4 addresses | ≥ 1 | **redact** | Appears legitimately in troubleshooting docs. |

Twelve patterns are detected: email, Serbian and international phone, IBAN, credit card,
US SSN, IPv4, JWT, AWS key, GitHub token, Slack token, private key. `redact_pii` replaces
each with `[REDACTED]`; `detect_pii` counts matches per category.

A real bug found by the test suite: `normalize` originally collapsed *leading* whitespace
runs, turning an 8-space Python indent into 2 spaces and silently corrupting every code
file. Trailing whitespace is now stripped; leading whitespace is never touched.

### 3.3 Deduplication

Two complementary mechanisms:

1. **Exact**: SHA-256 over the first 4,000 characters.
2. **Near-duplicate**: MinHash over word 5-grams, 64 permutations, LSH with 16 bands × 4
   rows (≈0.8 Jaccard operating point).

Two reproducibility hazards were fixed:

* Python's `hash()` is **randomised per interpreter run** for `str`. Using it made the
  MinHash signature different on every run, so deduplication was irreproducible. Now uses
  `blake2b` with a fixed key.
* `minhash` originally shingled entire documents. On 380 Mcharacters that is ~60M
  Python-level hash operations and takes hours. Near-dup detection is now bounded to a
  **4,000-character prefix** per document: templated boilerplate repeats from the first
  line, so the prefix detects the cases that matter while making cost linear in document
  count rather than corpus size.

Measured: 1 exact and 10 near-duplicates removed from 1,323 documents.

### 3.4 Language identification

Script ratios separate Cyrillic/Greek/Arabic/CJK; a small stopword table discriminates
English from Serbian within the Latin family. Serbian is additionally flagged by its
diacritics (`čćžšđ`).

**This is a heuristic, not a trained classifier.** It is reliable on long monolingual
documents and unreliable on short mixed-script text, which is why it only *overrides* an
upstream-declared language when the two disagree, and never when the upstream tag is
explicit.

---

## 4. Mixture

Target weights and measured outcome for `dataset-v1`:

| Category | Target | Achieved | Upsampled |
|---|---:|---:|---:|
| general (en.wikipedia) | 0.340 | 0.340 | 377 |
| code | 0.220 | 0.220 | 165 |
| technical | 0.120 | 0.120 | 138 |
| Serbian (both scripts) | 0.120 | 0.120 | 138 |
| public-domain literature | 0.100 | 0.100 | 0 |
| customer support | 0.100 | 0.100 | 116 |

Mixture weights are a **hypothesis under test**, not an optimum.
`scripts/build_dataset.py --mixture {no-sr,code-heavy,literature-heavy,<json>}` re-runs the
build under any alternative; no sweep result is claimed here because no sweep has been run.

### 4.1 The corpus is 98.7% Gutenberg by character, and that matters

`gutenberg.jsonl` contributes **378.29 of 383.62 Mcharacters**. Without the resampling
step above, the model would be a literary model. The mixture mechanism downsamples it by
roughly 40×.

But resampling works on **document counts**, and Gutenberg documents are ~360 KB each
while a Wikipedia article is ~7 KB. So the achieved *token* fractions differ from the
document fractions, and 116 repeats of a 15-document support knowledge base is a very high
repetition rate for a domain the system is supposed to be specialised on. This is the
single most important dataset limitation in the project. See §8.

---

## 5. Splits

| Split | Fraction | Documents | Tokens |
|---|---:|---:|---:|
| train | 99.3% | 1,303 | **19,485,297** |
| val | 0.5% | 6 | 51,287 |
| test | 0.2% | 2 | 15,085 |

Splits are by **document**, never by line, so no document straddles two splits.
`test_split_keeps_documents_intact` asserts the three splits are disjoint by identity.

Aggregate: **19,536,584 tokens**, **3.593 characters per token**.

---

## 6. Sharding

Documents are concatenated into contiguous token buffers with `<|eos|>` as the separator,
then written as fixed-length windows. Packing rather than padding means every sequence the
GPU sees is full of real tokens.

A real bug: `write_shards` hardcoded `dtype: "uint32"` in the index while writing `uint16`
arrays (chosen because the 32k vocabulary fits in 16 bits). The loader then reinterpreted
every token id — surfacing as `IndexError: index 2172472 is out of bounds for dimension 0
with size 32768`, which reads like a model bug rather than a data bug.

Fixed twice over:

1. `write_shards` derives the dtype from the arrays and raises on mixed dtypes.
2. `PackedDataset` cross-checks the recorded dtype against **file geometry**
   (bytes ÷ tokens must equal 2 for `uint16`, 4 for `uint32`) and raises a message naming
   `scripts/repair_shard_index.py`.
3. `scripts/repair_shard_index.py` repairs an already-built dataset without a 12-minute rebuild.

---

## 7. Manifest

`datasets/dataset-v1/manifest.json` records:

* dataset version, build timestamp, seed, sequence length
* tokenizer name, vocabulary size and SHA-256
* document/token/character totals and chars-per-token
* the full filter funnel and dedup statistics
* mixture targets, achieved fractions and **upsampled document counts**
* per-source retained document counts and characters
* provenance records: source, licence, licence URL, category, retrieval date, example URL
* acquisition statistics per source
* shard list with per-shard token counts and content hashes
* a `manifest_hash` computed over everything except the timestamp and build duration

`manifest_hash` is stored in every checkpoint's `metadata.json`, so "which corpus produced
these weights" is answerable without archaeology.

---

## 8. Known limitations

1. **19.5M training tokens is a small corpus.** Chinchilla-optimal training for the 55.7M
   `micro` model is ~1.1B tokens. The live run sees roughly 20M–50M, so it is heavily
   undertrained by construction. The model demonstrates that the pipeline works, not that
   it is well-trained.
2. **Serbian is ~1% of the corpus** (0.12 mixture weight against a pool of 21 Serbian
   documents). Wikipedia rate limiting (HTTP 429) prevented a larger crawl. This directly
   caps Serbian quality and is the highest-value data gap in the project.
3. **Enormous up-sampling.** 377 repeats of the general category, 116 of support. The
   support knowledge base (15 documents) is memorised rather than generalised. Retrieval,
   not model weights, carries most of the domain behaviour — which is the intended
   architecture, but it means the fine-tuned model's support knowledge is mostly recall.
4. **Gutenberg dominates by character** (98.7%). Literary register is over-represented
   relative to the intended mixture even after resampling.
5. **No deduplication against evaluation benchmarks.** Not an issue for the internally
   defined suites, but it would matter if public benchmarks were added.
6. **Wikipedia coverage is thin** (~100 articles total). The category crawler ran for
   minutes before being superseded.
7. **Toxicity filtering is not implemented.** Refusal/harassment detection exists in the
   *inference* path (`HARASSMENT_MARKERS`), but the pretraining corpus has no toxicity
   classifier applied. For a public release this is a blocker; see `docs/roadmap.md`.
8. **The `test` split has 2 documents.** It exists and is disjoint, but it is far too small
   to support a significance test.

---

## 9. Reproducing

```powershell
python scripts/acquire_data.py --force --wikipedia-random 30
python scripts/fetch_gutenberg_bulk.py --limit 1200
python scripts/crawl_wikipedia_categories.py --max-docs-per-lang 1500
python scripts/train_tokenizers.py
python scripts/build_dataset.py --seq-len 1024
```

Acquisition is **not** fully deterministic: Wikipedia's random-article generator returns
different articles each run. The category crawler and the Gutenberg ID selection are
deterministic for a given `--limit`. To get a bit-identical dataset, reuse the committed
`datasets/raw/*.jsonl` and re-run only the tokenize/mix/split stages.