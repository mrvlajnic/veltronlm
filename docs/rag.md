# Retrieval-Augmented Generation

```
question
   ↓  classify_ticket        deterministic intent taxonomy + confidence
   ↓  extract_entities       product, serial, urgency, language, version, amounts
   ↓  detect_safety_signals  prompt injection, credential requests, false premises
   ↓  rewrite_query          strip conversational filler (never the subject)
   ↓  expand_query           Serbian → English domain vocabulary
   ↓  BM25 + hashed-ngram    hybrid retrieval, Reciprocal Rank Fusion
   ↓  absolute confidence    lexical + z-scored embedding, weighted 0.70/0.30
   ↓  diversity rerank       cap chunks per document
   ↓  decide_escalation      refuse or hand off, with a named reason
   ↓  build_context          numbered blocks + citation list
   ↓  VeltronLM               grounded generation
   ↓  verify_citations       reject fabricated citation numbers
```

Reproduce: `python scripts/verify_rag.py`

---

## 1. Knowledge base

15 synthetic documents, 45 heading-aware chunks, 43 English + 2 Serbian.

| Category | Chunks | Contents |
|---|---:|---|
| `products` | 9 | VeltronHub X1, VeltronSense, Veltron Cloud plans |
| `policies` | 20 | Warranty/RMA, refunds, privacy/data, account security, device transfer |
| `shipping` | 4 | Dispatch times, carriers, tracking, delays and loss |
| `billing` | 4 | Subscription charges, invoices, VAT |
| `troubleshooting` | 4 | Hub offline, sensor dropout, app pairing, ownership transfer |
| `faq` | 4 | General FAQ, Serbian FAQ (Честа питања) |

> **Veltron Industries is fictional.** Every policy above is invented for engineering
> demonstration and is not legally binding. Each chunk carries `synthetic: true` and the
> disclaimer string, and the API returns `synthetic_data_notice` on every grounded answer so
> the flag cannot be silently dropped.

### 1.1 Chunking

Heading-aware with breadcrumb provenance:

* Headings open a new section and update a path (`Warranty and RMA > Turnaround`).
* Sections are split at `max_chars=1100` with `overlap_chars=150`, because a procedure that
  straddles a boundary must be retrievable from either side.
* Chunks index `title + section_path + tags + text`, so a query naming a product or section
  ranks its chunks even when the words appear only in the heading.
* Documents with no heading structure fall back to fixed windows rather than becoming one
  unusable block.

The breadcrumb is what lets a retrieved chunk cite `Warranty and RMA > Turnaround` instead
of an opaque chunk id.

### 1.2 Ingestion formats

`.md`, `.markdown`, `.txt`, `.json`, `.pdf`, `.html`. A malformed file is logged and
skipped rather than aborting the build. PDFs degrade to a recorded error if `pypdf` is
absent, so a mixed directory still ingests its readable formats.

## 2. Retrieval

Hybrid, because a support corpus needs both:

* **Lexical (BM25)** for exact identifiers — serial prefixes (`VH-9K2M1`), policy numbers,
  error strings — that an embedding blurs together.
* **Dense (hashed word/char n-grams)** for paraphrases sharing no vocabulary.

### 2.1 Why not a neural retriever

The knowledge base is 45 chunks. What "dense retrieval" must supply here is
lexical-overlap sensitivity for near-duplicate phrasings, which hashed n-grams capture
deterministically, with no model download, no semantic drift, and no possibility of the
retriever hallucinating a relationship the documents do not state. The trade-off is real —
see §5.1.

### 2.2 Reciprocal Rank Fusion

```
score(d) = Σ_lists 1 / (60 + rank(d))
```

Rank-based fusion is used instead of summing normalised scores because BM25 scores are
unbounded while cosine similarity lives in [−1, 1]. Their scales are not comparable, and
calibrating them on a handful of documents would be more fragile than ignoring them.

## 3. Absolute confidence

This is the most important design decision in the retrieval layer.

Rank fusion gives the top hit of **every** query a normalised score of 1.0 — including a
query that matches nothing. An earlier version thresholded that number, which meant the
escalation check **could never fire**. Measured: nonsense queries scored 0.90.

`RetrievedChunk` therefore carries two scores:

| Field | Range | Purpose |
|---|---|---|
| `score` | 0–1, top = 1.0 | ordering and display only |
| `confidence` | 0–1, **absolute** | "do we actually know the answer?" |

```
lex = bm25_score / max_bm25_score          in [0, 1]
vec = max(0, (sigmoid(z) - 0.5) * 2)       z = z-score across this query's candidates
confidence = 0.70 * lex + 0.30 * vec
```

Two details that were bugs first:

1. **A raw cosine is unusable.** On a 45-chunk corpus every chunk shares common n-grams, so
   raw cosines clustered at 0.78–0.89 regardless of relevance and carried almost no
   signal. Centering against the candidate set turns the number into "more similar than
   typical for this query".
2. **`sigmoid(0) = 0.5`**, so an uncentred logistic handed every candidate half a point of
   vector credit. A nonsense query produces nothing but average similarities, which then
   looked like weak evidence. Subtracting the midpoint makes "no better than typical"
   score exactly zero.

### 3.1 Measured separation

| Query | Top confidence |
|---|---:|
| `warranty coverage period months` | **0.878** |
| `How long is the warranty?` | 0.910 |
| `How much is the Plus plan?` | ~0.92 |
| `zzzqqq xyzzy nonsense gibberish` | **0.238** |
| `What is the airspeed velocity of an unladen swallow?` | 0.210 |

`MIN_RETRIEVAL_SCORE = 0.18`. A documented question clears it by 5×; an undocumented one
lands just above the floor but well below any documented query.

### 3.2 Taxonomy augmentation must not manufacture evidence

The pipeline appends vocabulary from the classified intent
(`technical_issue` → `offline error fault not working reset troubleshooting`). This widens
recall, but if it also fed the confidence score, an off-topic question would inherit every
injected term and score as well as an on-topic one. Measured: the swallow question rose
from 0.210 to 0.725 through augmentation alone.

`Retriever.search(confidence_query=...)` therefore scores confidence on the **rewritten
customer query** and uses the augmented form only for candidate selection.

### 3.3 Cross-lingual retrieval

The knowledge base is mostly English; Serbian customers write in Serbian. "Koliko traje
garancija?" shares **zero** tokens with the English warranty document.

`expand_query` appends English equivalents of Serbian domain terms from a hand-maintained
bilingual map (`гаранција → warranty, rma`; `не ради → not working, offline,
troubleshooting`; and ~50 more). The list is narrow and domain-specific on purpose: a
general dictionary would add noise faster than recall.

Measured effect: `Koliko traje garancija?` → `policy-warranty > Warranty and RMA > Coverage`.

## 4. Retrieval quality

`scripts/verify_rag.py`, 14 held-out queries, top-5 with `per_doc_cap=2`:

| Metric | Result |
|---|---|
| **hit@1** | **14/14 = 100%** |
| **recall@3** | **14/14 = 100%** |

Regression-tested in `tests/unit/test_rag.py`:
`test_retrieval_finds_the_right_document`, `test_serbian_query_reaches_english_document`,
`test_confidence_is_absolute_not_normalised`,
`test_nonsense_query_confidence_is_below_threshold`,
`test_per_document_cap_limits_dominance`.

### 4.1 Weighting was set by measurement, not taste

`0.70 lexical / 0.30 vector` was chosen after observing that at `0.50/0.50` a 2% BM25
shortfall on *"how much is the Plus plan?"* was overturned by the embedding, dropping the
pricing document to rank 2. Before the title boost, the same query retrieved
`policy-privacy-data > Retention` (0.952) over `product-veltron-cloud` (0.781).

### 4.2 Title/heading boost

Query terms found in a chunk's title, section path or tags receive a `TITLE_BOOST = 2.5`
multiplier on their IDF mass. *"how much is the Plus plan"* shares nearly as many body
tokens with a paragraph about subscription retention as with the pricing table; the heading
**"Veltron Cloud plans"** is what settles it.

## 5. Grounding and citations

### 5.1 What "groundedness" means here

Groundedness is measured as **lexical containment of required facts**: does the answer
contain the string `24 month` when the reference says 24 months?

This is a **necessary but not sufficient** condition. An answer can contain the right fact
and also invent a refund policy. The complementary signal is the negative one:
`unsupported_claims` — a number-plus-unit in the answer that does not appear in the
retrieved context — treated as an invented policy, price or deadline. Both are reported
together; neither is called "grounding" on its own.

### 5.2 Citation validation

The system prompt asks the model to end with `Sources: [n]`. Asked for such a line, a model
will sometimes invent `[4]` when only three chunks were retrieved, so the line is **parsed
and validated**, not trusted:

| Scenario | Claimed | Grounded | Fabricated | All valid |
|---|---|---|---|---|
| `Sources: [1]` | [1] | [1] | — | yes |
| `Sources: [1] [2]` | [1,2] | [1,2] | — | yes |
| `Sources: [1] [2] [3]` | [1,2,3] | [1,2] | **[3]** | **no** |
| `Sources: [9]` | [9] | — | **[9]** | **no** |
| no source line | — | — | — | yes (absence is not fabrication) |

Serbian `Izvori: [1]` is recognised too.

## 6. Escalation

`decide_escalation` is the single place where the system decides to hand off to a human.
Every branch names a reason from `ESCALATION_REASONS`, so escalations can be counted and
audited rather than being implicit in a prompt.

| Reason | Trigger |
|---|---|
| `prompt_injection_detected` | Injection marker in the message |
| `requests_credentials_or_third_party_data` | Secret request or third-party data |
| `security_incident_category` / `escalation_category` | Category forces a human |
| `no_retrieval_above_threshold` | Top confidence < 0.18, or nothing retrieved |
| `model_low_answer_confidence` | Generator confidence < 0.15 |
| `low_classification_confidence` | Intent confidence < 0.18 |
| `out_of_warranty_quote_required` | Available for policy rules |

Templates (English / Serbian):

```
"I don't have enough verified information to answer this safely, so I won't guess.
 I'm escalating this to a human support agent who can review it directly."

"Не могу да нађем тај одговор у доступној документацији Veltron podršке, па нећу да
 измишљам. Просим да овај случај преузме лудски оператер."
```

## 7. Query rewriting

**Only filler is removed.** An earlier version also stripped the customer's framing
(`my VeltronHub X1 is not working`), which deleted the product name and the symptom — the
two most informative terms in the sentence. Measured regression:

| Input | Earlier (broken) | Current |
|---|---|---|
| `Hi, my VeltronHub X1 is not working, please help` | `working, please help` | `my VeltronHub X1 is not working` |
| `Zar ne mogu da se ulogujem?` | `Zar ne mogu da se ulogujem?` | unchanged |

Framing carries signal in support queries. The vocabulary bridging that actually helps is
`expand_query` and the intent augmentation, not deletion.

## 8. Limitations

1. **45 chunks.** Retrieval quality is excellent *on this corpus* and says nothing about a
   50,000-document knowledge base. No larger retrieval benchmark has been run.
2. **Hashed embeddings, not semantic.** A question sharing no vocabulary with the document
   and containing no Serbian term in the bilingual map will fail. There is no true
   paraphrase model here.
3. **`MIN_RETRIEVAL_SCORE = 0.18` is tuned on one corpus** of one domain. It has no
   theoretical justification and would need re-tuning per knowledge base.
4. **Bilingual map is hand-maintained** and covers ~50 terms. Serbian coverage is
   necessarily partial.
5. **Groundedness is lexical.** An answer that contains the right fact while also inventing
   a policy can still score as grounded; `unsupported_claims` catches only numeric claims.
6. **Reranking is a diversity cap, not a learned cross-encoder.** Ordering is BM25 + n-gram
   similarity.
7. **No query decomposition.** A multi-part question ("refund policy *and* how do I return
   the device?") retrieves for the whole string and may miss the second part.
8. **The knowledge base is synthetic**, so retrieval is validated against invented policies.
   Real policy text is longer, more cross-referential and less uniformly structured.