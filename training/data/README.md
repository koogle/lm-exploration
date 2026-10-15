# Local training datasets

`datasets.json` is the UI's curated registry. The trainer itself accepts a plain
UTF-8 text file and does not depend on this registry. Paths in the registry are
relative to this `data/` directory, not to the shell's working directory.

The UI launches the shared `training/trainer/` implementation through `cli.py`
with the fixed `training/100m/model.py` architecture (the trainer reports the
exact parameter count). It does not accept a model/executable path or expose an
architecture picker. From `training/`, start it with `uv run lm-ui` and open
http://127.0.0.1:5174. Trainer cwd stays `training/`; inputs stay under `data/` and
run artifacts under `runs/`.

## Registered datasets

Both registered datasets are character-level Shakespeare learning corpora for
exercising training, checkpointing and sampling. Their text is not committed to
Git. From `training/`, fetch them before starting the UI:

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
2. Prepare one nonempty UTF-8 text file under this directory. Local source material
   can live under `raw/`; derived files under `cache/`. Both directories are ignored
   by Git. The initial UI limits each file to 64 MiB and each run to small local
   experiments; this is not a large-corpus ingestion system. Use enough text for
   the trainer's context window and held-out split.
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

   The file must exist even for an ineligible entry. Ineligible entries remain
   visible but cannot be selected for a run. Do not add an undecided candidate
   source as an eligible entry.
4. Restart the UI to reload and validate the registry. Invalid metadata or missing
   files prevent startup with an error rather than silently changing the registry.
   On Start, the UI validates UTF-8 and stores the selected metadata, file size and
   SHA-256 alongside the run in `runs/<id>/ui.json`. Do not modify a dataset while
   a run is starting or running: the trainer independently fingerprints its input
   in `config.json` and checkpoints, and owns resume compatibility checks.

Dataset definitions and fetching scripts are tracked; `raw/`, `cache/` and
`../runs/` are not. Each research dataset needs its own evidence; this README is not a license
allowlist or an approval of the candidate sources in the training plan.
