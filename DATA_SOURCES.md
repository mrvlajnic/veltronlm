# Data Sources and Licences

Every corpus VeltronLM trains on is declared here with its licence **before** it is
downloaded. A source not in `veltron/data/sources.py` cannot be fetched by the acquisition
pipeline.

## Register

| Source | Licence | SPDX / URL | Commercial use | Redistribution | Attribution |
|---|---|---|---|---|---|
| Wikipedia (en) | CC BY-SA 4.0 | [link](https://creativecommons.org/licenses/by-sa/4.0/) | permitted | **share-alike** | required |
| Wikipedia (sr, sh) | CC BY-SA 4.0 | [link](https://creativecommons.org/licenses/by-sa/4.0/) | permitted | **share-alike** | required |
| Project Gutenberg | Public Domain | [link](https://www.gutenberg.org/policy/license.html) | permitted | permitted | not required |
| CPython stdlib | PSF-2.0 | [link](https://github.com/python/cpython/blob/main/LICENSE) | permitted | permitted | required |
| Flask, Click, Werkzeug | BSD-3-Clause | [link](https://spdx.org/licenses/BSD-3-Clause.html) | permitted | permitted | required |
| requests | Apache-2.0 | [link](https://spdx.org/licenses/Apache-2.0.html) | permitted | permitted | required |
| urllib3 | MIT | [link](https://spdx.org/licenses/MIT.html) | permitted | permitted | required |
| NumPy | BSD-3-Clause | [link](https://spdx.org/licenses/BSD-3-Clause.html) | permitted | permitted | required |
| Rust, Node.js, Go, Black READMEs | MIT / Apache-2.0 / BSD-3-Clause | per-repository | permitted | permitted | required |
| Veltron synthetic knowledge base | CC0-1.0 | [link](https://creativecommons.org/publicdomain/zero/1.0/) | permitted | permitted | not required |

## Share-alike obligation

**Wikipedia is CC BY-SA 4.0.** Redistribution of derivatives — including model weights
trained on it — carries a share-alike obligation in most jurisdictions. Two practical
consequences:

1. A release trained on the full `dataset-v1` mixture should be released under terms
   compatible with CC BY-SA 4.0.
2. A release trained on a **public-domain-only** mixture (Project Gutenberg + synthetic KB
   + permissively-licensed code) can be released under Apache-2.0 alone.

This is a legal question, not a technical one. VeltronLM records it rather than resolving
it. **Obtain advice before releasing weights.**

## Excluded

| Source | Licence | Why excluded |
|---|---|---|
| `git/git` documentation | GPL-2.0-only | Copyleft would attach to the weights as a derivative work. Enforced in code, not just documented. |

## Per-file code provenance

Code files carry their SPDX identifier individually in `RepoFile(repo, ref, path, spdx)`, so
the licence of every downloaded byte is known rather than inferred from the repository.
194 files: 122 PSF-2.0, 46 BSD-3-Clause, 12 MIT, 9 Apache-2.0, 5 MIT-or-Apache-2.0.

## Synthetic data

`veltron/support/knowledge_base.py` contains 15 documents describing a **fictional**
company, Veltron Industries. Every policy is invented for engineering testing and is not
legally binding. Each document carries `synthetic: true` and a disclaimer string, and the API
returns `synthetic_data_notice: true` on every grounded answer.

## Personal data

None. The corpus contains no private customer information. The acquisition pipeline applies
12 PII patterns with a deliberate drop/redact split before any text reaches a shard:

* **Dropped outright:** private keys, AWS keys, GitHub/Slack tokens, JWTs, credit cards,
  IBANs, US SSNs, and any document with 3+ emails or 4+ phone numbers
* **Redacted:** a single email address or phone number, which a support log may legitimately
  contain

Measured on `dataset-v1`: 56 documents (4.1%) dropped for PII, 0 redacted documents retained
with identifiable credentials.

## Attribution

When redistributing anything derived from this corpus:

```
VeltronLM contains material from Wikipedia (CC BY-SA 4.0), the Project Gutenberg
Library (public domain), CPython (PSF-2.0), and the Pallets/requests/urllib3/NumPy
projects (BSD-3-Clause, Apache-2.0, MIT). See DATA_SOURCES.md for per-file details.
```

Individual authors and titles are preserved in the corpus provenance records
(`datasets/*/provenance.json`) and in the source URLs recorded per document.

## Verification

`python -c "from veltron.data.sources import licence_report; import json; print(json.dumps(licence_report(), indent=2))"`

Regenerate: `python scripts/build_dataset.py` writes `per_source` counts and `provenance`
into `datasets/*/manifest.json`.
