# SFT: code repair, CIG and math

Post-training for the 124M Simple LLM models: supervised fine-tuning on code repairs or
math word problems, starting from an explicit `scholarly-ddp-v1` pretraining checkpoint
on the private `lm-exploration-training` Modal volume. Model weights, architecture and
the parent's frozen tokenizer are inherited; AdamW starts fresh. Nothing launches
automatically, and no checkpoint is published.

## What SFT trains

Every task trains the same way ([`../trainer/sft.py`](../trainer/sft.py)). An example is
one prompt and one complete answer that people or another model wrote; the model being
trained never generates during training. The model reads `BOS + prompt` and is trained
by teacher forcing to predict each answer token, and EOS, from the prompt and the answer
tokens before it. The prompt carries no loss. Each step samples `batch_size` examples
with replacement and minimizes the mean answer-token cross-entropy; validation reports
the same quantity per bucket.

Two settings make training gentler on what pretraining learned:

- **Replay** (`"replay": true` with the `replay_*` settings,
  [`../trainer/replay_hook.py`](../trainer/replay_hook.py), the hook every post-training
  loop shares): each step also trains on `replay_batch_size` windows of
  `replay_context` tokens of pretraining text from a raw snapshot on the data volume
  (`replay_snapshot`). The pretraining loader
  ([`../data/raw_token_stream.py`](../data/raw_token_stream.py)) reads them, mixing
  sources by the `replay_sources` weights. Replay windows use the ordinary pretraining
  loss on every token. The step backpropagates `(1 - replay_weight) * answer loss`, then
  the hook adds the gradient of `replay_weight * replay loss` before the optimizer
  step. Validation adds replay loss per source on each source's first validation
  windows.
- **Schedule**: `"schedule": "cosine"` warms the learning rate up over `warmup_steps`
  and decays it by a cosine to `min_lr_ratio * lr`; the default is a constant rate.

`settings.task` picks the dataset ([`../data/sft.py`](../data/sft.py) holds the shared
`SFTDataset`; each task's subclass only says how a row becomes a prompt and an answer):

| Task | Prompt | Answer the model learns to write | Data |
| --- | --- | --- | --- |
| `repair` (default) | language, task and broken code | the complete corrected code | [`../data/repair_sft.py`](../data/repair_sft.py) |
| `math` | `Question: {question}\nAnswer:` | the full worked solution, ending in `#### N` | [`../data/math_sft.py`](../data/math_sft.py): GSM8K and MetaMathQA solutions |

The math runs in [MATH_EXPERIMENTS.md](MATH_EXPERIMENTS.md) used `math`; the earlier
repair runs in [EXPERIMENTS.md](EXPERIMENTS.md) used `repair`.

- **Repair SFT** (`"task": "repair"`): validation reports answer-token cross-entropy
  overall and per language.
- **CIG-SFT** ([`../trainer/cig.py`](../trainer/cig.py), `"objective": "cig"`): the
  token-weighted objective from
  [What Should SFT Learn? (arXiv:2610.12995v1)](https://arxiv.org/abs/2610.12995v1),
  equations 7 and 19–20. Before any update, the frozen parent scores every training
  answer twice: with the full prompt, and with instruction and broken code emptied
  (language and scaffolding kept; answer tokens identical). The detached information
  ratio sets projected-target weights with proximity lambda `cig_proximity`
  (default 0.3). Weighted losses divide by the supervised-token count, not the weight
  sum. Validation always uses ordinary cross-entropy, so both objectives are comparable.
- **Math SFT** (`"task": "math"`, format `math-sft-v1`): the prompt is also the zero-shot
  format of [`analysis/evaluation`](../analysis/evaluation/README.md). Validation reports
  answer-token cross-entropy per source.

## Data format

UTF-8 JSONL, one repair per line ([`../data/repair_sft.py`](../data/repair_sft.py),
format `repair-sft-v2`):

```json
{"id":"repo-a-fix-1","group":"repo-a","split":"train","language":"python","instruction":"Return the sum of a and b.","broken":"def add(a, b):\n    return a - b\n","fixed":"def add(a, b):\n    return a + b\n","source":"repository URL + commit","license":"MIT"}
```

- `id`, `group`, `split` (`train`/`validation`), `language`, `instruction`, `broken`,
  `fixed`, `source` and `license` are required nonempty strings. `language` is an
  explicit lowercase identifier (`python`, `cpp`, `bash`, ...); every training
  language needs validation examples.
- `group` is the repository or change family. A group, and any normalized
  `broken`/`fixed` code, may not appear in both splits. Repairs must change code.
- Examples longer than the context fail instead of truncating the answer.
- Preparation never executes supplied code; `source` and `license` are declarations.
  Sampling is uniform over examples with replacement, so set mixtures through counts.

Builders (all write the format above):

| Builder | Input | Notes |
| --- | --- | --- |
| [`repair_curriculum.py`](../data/repair_curriculum.py) | none | Original CC0 arithmetic repairs in nine languages, 576 train / 36 validation. Absolute-difference and maximum families are held out. Rows carry executable `cases` and `family` for the evaluator. A pipeline check, not repository repair. |
| [`commitpack_sft.py`](../data/commitpack_sft.py) | [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) at pinned revision `fc56fe33…` | Commit subject + old file → new file. License allowlist, generated-path and boilerplate filters. Repository aliases are joined before a ~95/5 repository split, and validation code is hash-blocked from training. Tests not rerun. |
| [`private_patch_sft.py`](../data/private_patch_sft.py) + [`download_private_patches.py`](../data/download_private_patches.py) | a sources manifest of pinned HF dataset revisions | Exact before/after buffers checked against Git blob SHA-1s, cut into changed regions with six context lines. Splits are commit-disjoint within repositories, validation excerpts are hash-blocked, and common credential patterns are filtered (not a full secret/PII audit). An optional replay slice comes from an earlier dataset. |

A private-patch sources manifest is a JSON list of
`{"repo": "<owner/dataset>", "revision": "<40-hex commit>", "default_license": "<optional>"}`.
The downloader selects `changes/*.jsonl.gz` shards, up to 24 per repository unless
`--full` is set. `default_license` labels rows whose commit carries no
repository license. License evidence is always retained by hash.

## Run

From `training/` with `uv sync --locked --extra corpus --extra research` and the
CA setup in [`../distributed/README.md`](../distributed/README.md). All launchers
follow the repository's Modal limits: no retries, function timeouts of six hours or
less, and a recorded launch deadline 24 hours ahead that every remote call checks.
Training stops taking updates 30 minutes before that deadline, then evaluates and
saves. Use fresh run and data IDs; outputs never overwrite an existing volume folder.

### Prepare data

```sh
# Execute-verify every broken/fixed curriculum pair in Modal CPU containers; writes locally.
modal run -m modal_scripts.modal_sft_data::curriculum --output artifacts/sft/repairs.jsonl
# CommitPackFT to /runs/<data-id>/{examples.jsonl,report.json}; block the curriculum's held-out code.
modal run -m modal_scripts.modal_sft_data::commitpack --data-id <fresh-data-id> \
  --exclude-dataset artifacts/sft/repairs.jsonl --report artifacts/sft/<fresh-data-id>.json
# Private patches; uses the Modal secret simple-llm-hf. A failed shard download aborts.
modal run -m modal_scripts.modal_sft_data::private_patches --sources <sources.json> \
  --data-id <fresh-data-id> --reserved <earlier-data-id>,... --replay <earlier-data-id> \
  --report artifacts/sft/<fresh-data-id>.json
```

Validate a local JSONL without Modal:

```sh
uv run --extra corpus python -c 'from training_data.repair_sft import REPAIR; from training_data.subword import load_frozen; print({k: len(v) for k, v in REPAIR.prepare("artifacts/sft/repairs.jsonl", load_frozen(), 1024)[0].items()})'
```

Math word problems (`math`) come from GSM8K's training split and MetaMathQA's
GSM-derived rows (rephrasings, answer augmentation, self-verification and FOBAR
rewrites; its MATH-derived rows are not used), both MIT and pinned to HF commits:

```sh
modal run -m modal_scripts.modal_sft_data::math --data-id <fresh-data-id> \
  --report artifacts/sft/<fresh-data-id>.json
```

The first 500 GSM8K training problems are validation, and MetaMathQA rewrites of them
are dropped. GSM8K's `<<a*b=c>>` calculator annotations are removed and MetaMathQA's
closing `The answer is: N` becomes `#### N`. Any problem sharing a 13-word n-gram with a
GSM8K or SVAMP test question is dropped. Each row's `group` is its original problem, so
a problem and its rewrites never cross splits.

### Train

[`../modal_scripts/modal_sft_train.py`](../modal_scripts/modal_sft_train.py) runs one
recipe on one L40S or H100. A recipe is JSON
([`../trainer/sft_recipe.py`](../trainer/sft_recipe.py) validates it):

```json
{
  "name": "example",
  "run_id": "repair-sft-example-r01",
  "parent": "<pretraining-run>/step-NNNNNNNNN.pt",
  "data": {"local": "artifacts/sft/repairs.jsonl"},
  "gpu": "L40S",
  "settings": {"steps": 200, "batch_size": 4, "context": 1024, "lr": 1e-5, "objective": "sft"}
}
```

Recipes for the math runs are in [`recipes/`](recipes/).
`data` is `{"local": path}` (validated locally, uploaded, at most 20 MiB) or
`{"volume": data-id}` (a prepared volume folder; its hash must match `report.json`).
`settings` are `trainer.sft.Settings`: `steps` (total target), `batch_size`,
`context`, `lr`, `seed`, `evaluate_every`, `save_every`, `max_seconds` (training-loop
limit, at most 14,400), `objective` (`sft` or `cig`) and `cig_proximity`. `resume`
continues from an SFT checkpoint with identical identity. Adding `"extend": true`
also permits a larger step target and new cadence/time limit, but no change to
learning, data, model, objective or trainer source; lineage goes to `continuation.json`.

```sh
# Exact resume qualification: uninterrupted run vs. stop, fresh-container restore, continue.
modal run -m modal_scripts.modal_sft_train::qualify --recipe <recipe.json> --run-prefix <fresh-prefix>
# Train; command-line values override the recipe's parent, data, run ID and resume.
modal run -m modal_scripts.modal_sft_train::main --recipe <recipe.json> \
  --run-id <fresh-run-id> --parent <run>/step-NNNNNNNNN.pt [--dataset <jsonl> | --data-id <id>]
```

The run folder on the volume holds the manifest (dataset, parent, tokenizer, source and
settings hashes), the exact examples, tokenizer and source snapshots, `metrics.jsonl`
(per-step loss and periodic validation), CIG reference scores and periodic full
checkpoints (model, optimizer, sampler and RNG state). The local result goes to
`artifacts/sft/<run-id>.json`. A time-limited run can stop below its step target;
resume it into a new run ID.

### Evaluate executable repairs

[`../modal_scripts/modal_sft_eval.py`](../modal_scripts/modal_sft_eval.py) greedily
decodes (BF16, at most 128 new tokens) two held-out curriculum families per language
for each named checkpoint, pretraining or SFT. The grader is first calibrated on
known fixed/broken code. Each answer then runs in its own network-blocked Modal
Sandbox without volumes or secrets. Only printed outputs leave the sandbox, and
they are compared outside it against the expected values.

```sh
modal run -m modal_scripts.modal_sft_eval --dataset artifacts/sft/repairs.jsonl \
  --checkpoints base=<run>/step-NNNNNNNNN.pt,sft=<sft-run>/step-NNNNNNNNN.pt \
  --output artifacts/sft/evaluation.json
```

These 18 tasks are a small diagnostic, not a repository-repair benchmark. Track
EOS, build/run stage, and pass rate together with held-out loss.

## Limits

- Single GPU only; the `repair-sft-v2` checkpoint format is not accepted by the
  pretraining publication scripts.
- Commit-derived data is not execution-verified, and no builder audits semantic
  clones or pretraining contamination.
- Source changes intentionally break exact resume of older SFT checkpoints; use the
  source recorded in a run's manifest to continue it.
