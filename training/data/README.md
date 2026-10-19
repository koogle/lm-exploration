# Training datasets

[EU institutional corpus exploration](eu/README.md) is a separate discovery and
extraction project with ten source candidates, bounded crawlers, agent research
notes and Parquet export. Its artifacts are not registered training inputs.

For the new frozen-tokenizer corpus, use the separate
[Simple LLM streaming pipeline](SIMPLE_LLM_STREAMING.md). Its local preparation
CLI lives in [`prepare_simple_llm.py`](prepare_simple_llm.py); run
`uv run lm-prepare-simple-llm --help` (or
`uv run python -m training_data.prepare_simple_llm --help`) from `training/`.
The corresponding CPU Modal launcher lives in
[`../modal_scripts/modal_prepare_simple_llm.py`](../modal_scripts/modal_prepare_simple_llm.py).

`datasets.json` records each dataset's source, license review and eligibility;
[`registry.py`](registry.py) validates it with pydantic.
Paths in it are relative to this `data/` directory. [Recipes](../recipes/README.md)
select a dataset by `id` from the `DATASETS` table in `trainer/recipe.py`, which maps
it to its fetch module. Raw text is never baked into the Modal image: each worker
runs that module, which downloads and verifies the file.


## Registered datasets

Both registered datasets are character-level Shakespeare learning corpora for
exercising training, checkpointing and sampling. Their text is not committed to
Git. Modal workers fetch them automatically; to inspect them locally, run from
`training/`:

```sh
uv run python -m training_data.fetching.fetch_tinyshakespeare
uv run python -m training_data.fetching.fetch_shakespeare_complete
```

- `tiny-shakespeare` (~1.1 MB): the char-rnn benchmark, saved to
  `data/raw/tinyshakespeare.txt`.
- `shakespeare-complete` (~5.4 MB): Project Gutenberg eBook #100, saved to
  `data/raw/shakespeare-complete.txt` with the Gutenberg header and license footer
  removed and line endings normalized. Gutenberg revises this eBook in place, so
  a new revision fails the pinned check until the hashes are deliberately updated.

Each fetcher verifies a pinned SHA-256. Source and license notes are in
`datasets.json`. These are not selected research corpora or evidence of useful
language capability. Research corpus choices and language mixtures remain undecided.

## Add an eligible local text dataset

1. Establish the source's eligibility and any attribution, retention, redistribution
   and removal obligations before enabling training. Public availability alone is
   not eligibility. Keep record-level source/revision/license/filtering provenance
   separately when aggregating documents, and inspect for credentials and personal
   information. Do not commit private or unreviewed corpus content.
2. Write a fetch module in `fetching/` whose `main()` downloads one nonempty UTF-8
   text file to its `DESTINATION` under `raw/` and verifies a pinned SHA-256 (see
   the existing fetchers).
   `raw/` and `cache/` are ignored by Git and excluded from the Modal image. The
   trainer reads the whole file into RAM; this is not a large-corpus ingestion
   system. Use enough text for the trainer's context window and held-out split.
3. Add an object to the `datasets` array in `datasets.json` with all required fields:

   | Field | Meaning |
   | --- | --- |
   | `id` | Unique lowercase slug: letters, digits, `_` and `-`, at most 64 characters |
   | `name` | Short display name |
   | `path` | Relative path under `data/`, no absolute paths, `..`, or escaping symlinks |
   | `description` | Content, intended use and limitations |
   | `source` | Authorship or source URL/repository and pinned revision; reference retained record-level provenance when applicable |
   | `license_notes` | Actual terms and eligibility review, obligations and limits; do not invent approvals |
   | `languages` | Nonempty array of language identifiers, e.g. `["en"]` |
   | `training_eligible` | Boolean; `true` only after the above review for the intended use |

   Do not add an undecided candidate source as an eligible entry.
4. Add the `id` to `DatasetName` and `DATASETS` in `trainer/recipe.py`, mapped to
   the fetch module. Recipes refuse datasets that are not `training_eligible`. Reference the `id` from a recipe and launch a `--smoke` run. The trainer
   fingerprints the exact bytes it trained on in each run's `config.json` and
   checkpoints.

Dataset definitions and fetching scripts are tracked; `raw/`, `cache/` and
`../runs/` are not. Each research dataset needs its own evidence; this README is not a license
allowlist or an approval of the candidate sources in the training plan.
