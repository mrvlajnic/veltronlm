# Contributing to VeltronLM

VeltronLM is a research project. The contribution bar is therefore **evidence**, not
ambition: a change that makes a system do more is worth less than a change that makes it
possible to tell whether it works.

## The one rule

**Never describe an unmeasured capability as working.**

If a number appears in a document or commit message, a command produced it. If a feature
has no test, it does not have a test. If an experiment is planned, it belongs in
`docs/roadmap.md`, not in `experiments/`.

## Workflow

```powershell
python -m pytest tests -q                    # 181 tests, ~13 s
ruff check veltron tests scripts
python -m veltron.cli info                   # what exists on this machine
```

## Adding a model component

1. Add it to `ModelConfig` with a default, so existing configs are unchanged.
2. Extend `param_counts()` with the analytic shape.
3. **Update `test_analytic_param_count_matches_meta_device`** — it compares your algebra
   against a real `meta`-device instantiation and will fail if you miss a tensor.
4. Add a test that fails without the component, not just one that passes with it.

## Adding a data source

1. Add a `Source` entry to `veltron/data/sources.py` **before** writing the fetcher. The
   licence must be an actual SPDX identifier, not a guess.
2. Reject GPL in the fetch loop if the corpus is not GPL-compatible.
3. Add the filtering stages your content needs, and extend `quality_report` if it needs a
   new signal. Add the fixture to `tests/unit/test_data.py`.
4. Record provenance in the manifest automatically — do not bypass it.

## Adding an experiment

Copy `experiments/EXPERIMENT_TEMPLATE.md`. Change **one variable**. Record failures;
`experiments/EXP-0007` (two false canary failures) and `EXP-0010` (summaries of
Gutenberg boilerplate) are the most useful records in the directory precisely because
they went wrong.

## Style

* Comments explain *why*, not *what*. The interesting decisions in this codebase have
  comments recording the alternative that was rejected and the measurement that decided it.
* No `except Exception: pass`. Catch the specific error, fix the cause, or let it
  propagate. A silently swallowed exception in a training loop costs a night.
* No hard-coded paths. Everything goes through config or the environment.
* Docstrings state the reasoning behind non-obvious choices.

## Pull requests

* Say which commands you ran and paste the output.
* If you claim a performance number, state the hardware and whether the GPU was idle.
  See `experiments/EXP-0013.md` for why that matters.
* If you change the corpus or tokenizer, regenerate the manifest and note the new hash.

## Licence

By contributing you agree your work is licensed under Apache-2.0, and that you have the
right to license any data you contribute. Do not contribute data you cannot redistribute.
