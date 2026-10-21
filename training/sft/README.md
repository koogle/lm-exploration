# Multi-language code repair SFT pilot

The [first executed pilot](../results/repair-sft-20261020/README.md) completed
200 updates after nine-language data verification and exact GPU resume qualification.
Held-out loss worsened; this is a pipeline qualification, not an improved-model claim.

Stacked on PR #26, initialized from its completed 124M / 10.879B checkpoint:
`simple-llm-code60-fresh-5b-20261019-r02-s02/step-000066065.pt`.
This is a new post-training experiment: AdamW starts fresh; pretraining optimizer
moments are intentionally not carried into SFT. Model weights, architecture, and
frozen tokenizer are inherited. Nothing is launched automatically.

## Data contract

Supply a UTF-8 JSONL file, one repair per line:

```json
{"id":"repo-a-fix-1","group":"repo-a","split":"train","language":"python","instruction":"Return the sum of a and b.","broken":"def add(a, b):\n    return a - b\n","fixed":"def add(a, b):\n    return a + b\n","source":"repository URL + base/fix commit IDs","license":"MIT"}
```

Set `language` explicitly: for example `python`, `javascript`, `typescript`, `go`,
`rust`, `c`, `cpp`, `java`, or `bash`. Other lowercase language identifiers are
accepted; preparation and the training objective are language-independent. Use one
consistent identifier per language. The prompt names that language, and each
training language must have validation examples. Mixed-language batches work
without changing the model or tokenizer. Existing rows need a language field;
there is no implicit Python default. Version 1 SFT checkpoints are not resumable
under this changed prompt/data contract; start from the PR #26 parent instead.

Include separate `validation` records with different groups and code. Use repository
or related-change family as `group`; assign splits before generating variants.
Required fields are nonempty strings. Both splits must be present. Prompt text and
broken code are context only; corrected code and EOS are supervised. Padding is
ignored. Overlength examples fail rather than silently truncating solutions.
Source and license are provenance declarations, not automated eligibility checks.

For PR-derived data, separate implementation edits from test edits. Select short,
single-file repairs whose relevant test fails at the base commit and passes
with the reference repair. Record this verification alongside the source dataset.
The JSONL importer does not execute arbitrary supplied code or certify correctness;
only feed it independently verified repairs. Use language-appropriate verification:
pytest for Python, Node/TypeScript tests, `go test`, `cargo test`, C/C++ build and
test commands, Java tests, and shell behavior fixtures. These commands describe
corpus verification requirements, not generic execution backends for imported PRs. An original executable starter
curriculum is described below; no external corpus is bundled. Set mixture proportions through input example counts; sampling
is uniform over examples, not balanced automatically across languages.

## Validate and launch

Run from `training/` with the corpus and research extras installed. Validate without
allocating a GPU:

```sh
uv run --extra corpus python -c 'from training_data.repair_sft import prepare; from training_data.subword import load_frozen; print({k: len(v) for k,v in prepare("repairs.jsonl", load_frozen(), 1024)[0].items()})'
```

Explicit bounded launch (one L40S, default 200 updates, batch 4, LR 1e-5,
1,024-token examples, 1,200-second training-loop limit; worker timeout 3,600 s):

```sh
uv run --extra corpus --extra research modal run modal_repair_sft.py \
  --dataset repairs.jsonl --run-id repair-sft-pilot-01
```

Use `--steps 2 --batch-size 1` for GPU qualification before a substantive pilot.
Inputs are capped at 20 MiB for this initial in-memory workflow. No publication or
hub upload occurs. Private run-volume outputs contain the exact examples, source
snapshots, dataset/parent/tokenizer/source hashes, settings, per-step answer-token
loss, aggregate and per-language validation answer-token loss before and after training, and periodic
full checkpoints. Unique run IDs prevent overwriting existing runs.

Resume a time-bounded run in a **new** output run ID using `--resume
repair-sft-pilot-01/step-000000050.pt`, with the same parent, dataset, source and
settings. Steps are the total target, not additional steps. Resume restores model,
optimizer, random example sampler, CPU/CUDA RNG and answer-token counters.
Each draw samples training examples uniformly with replacement; exposure is not a
unique-data count. A time limit is checked between updates, not inside validation.

## Evaluation and current limits

The training loop evaluates held-out **answer-token loss**. The separate starter
curriculum diagnostic below measures executable repair success for base/SFT.
Broader repository evaluation and matched Qwen comparison remain follow-on work.
Track syntax validity, EOS termination, test pass rate, and edit size. Never run
untrusted generated code in the training container or let it edit grading tests.
The SFT checkpoint format is `repair-sft-v2`; the existing pretraining publication
scripts do not yet accept it. Distributed SFT and automated PR mining remain follow-on work. The sandbox
evaluation below is specific to the starter curriculum, not arbitrary repositories.

## Verified starter curriculum and GPU qualification

`data/repair_curriculum.py` generates an original CC0 synthetic arithmetic-repair
curriculum in nine languages. It has 576 training and 36 validation examples;
absolute-difference and maximum task families are held out across every language.
The 612 pairs are small and highly templated: they qualify the pipeline and teach
short repair/termination behavior, not broad repository repair competence.

From `training/`, prepare and execute-verify every broken/fixed pair on Modal CPU:

```sh
modal run modal_repair_data.py --output /tmp/repair-sft-data/repairs.jsonl
modal run modal_repair_sft.py::qualify \
  --dataset /tmp/repair-sft-data/repairs.jsonl --run-prefix repair-qual-UNIQUE
modal run modal_repair_sft.py::main \
  --dataset /tmp/repair-sft-data/repairs.jsonl --run-id repair-pilot-UNIQUE
```

Qualification compares two uninterrupted updates against one update followed by
fresh-container restore and the second update. Model, optimizer, counters, sampler
and CPU/CUDA RNG must match exactly. A stop-step argument permits qualification
without changing the total target or checkpoint identity. Every worker uses a fresh
container. Use the installed environment/extras and system CA setup described in
`distributed/README.md` in the managed workspace.

After training, `modal_repair_eval.py` compares the parent and selected SFT checkpoint
on two held-out repair families per language using greedy decoding capped at 128
new tokens. It calibrates the grader with known fixed/broken examples first. Each
candidate runs in a separate network-blocked Modal Sandbox without volumes or
secrets. Expected answers stay outside the sandbox; a successful exit alone does
not earn a pass. This is an 18-task diagnostic, not a benchmark capability claim.

```sh
modal run modal_repair_eval.py --dataset /tmp/repair-sft-data/repairs.jsonl \
  --checkpoint repair-pilot-UNIQUE/step-000000200.pt \
  --output /tmp/repair-sft-data/generation-evaluation.json
```

## Larger commit-derived experiment

`data/commitpack_sft.py` prepares the nine language subsets of
[CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) at immutable
revision `fc56fe33c030c6daa414c2b112c932b8eed085e6`. Inputs are the commit subject and
complete old file; the supervised answer is the complete new file. This includes
repairs and other instruction-driven edits. Repository tests have **not** been
rerun, so these are not executable-verified repair pairs.

Preparation preserves commit URLs, file paths, per-record licenses and source
hashes. It restricts licenses to the allowlist in the builder, filters generated
paths and boilerplate instructions, and rejects examples exceeding 2,048 tokens
instead of truncating answers. Repository aliases are joined across all nine
languages before a deterministic approximately 95/5 repository split. All
validation code is excluded from training by normalized exact hashes. The original
synthetic validation code is also excluded. This is not semantic clone detection,
nor evidence that the base model never saw these repositories during pretraining.

The deterministic sample caps each language at 6,000 training and 64 validation
examples. Languages with fewer eligible records retain all eligible records.
Sampling in training remains uniform over examples, not uniform over languages.

```sh
modal run modal_commitpack_sft.py \
  --data-id repair-commitpack-UNIQUE \
  --old-dataset /tmp/repair-sft-data/repairs.jsonl \
  --report-path /tmp/repair-sft-data/commitpack-report.json
modal deploy modal_repair_scale.py
```

Deploying does not train. Explicitly submit the durable job after successful data
preparation (use the environment's documented proxy/CA setup):

```python
import modal

worker = modal.Function.from_name("simple-llm-repair-sft-scale", "worker")
call = worker.spawn("repair-scale-UNIQUE", "repair-commitpack-UNIQUE")
print(call.object_id)  # retain for status/result retrieval
```

The larger worker reads data from the private volume, verifies its report hash,
then runs the same qualified answer-only SFT loop from the original PR #26 parent.
It uses one H100, batch 8, context 2,048, LR 1e-5, a 5,000-update total target, and
validation/checkpointing every 250 updates. A four-hour loop limit and five-hour
worker timeout bound each invocation. A time-limited run can finish below the
update target; inspect `result.json`. Full checkpoints support continuation in a
new run ID with identical settings and `resume="old-run/step-NNNNNNNNN.pt"`.

Poll `modal.FunctionCall.from_id(call_id).get(timeout=0)` for completion, and use
`modal app logs simple-llm-repair-sft-scale` for progress. Per-step metrics,
provenance, data, source snapshots, checkpoints and final result stay on the volume.
Longer training is an experiment, not evidence of improved held-out repair success.

## Private patch SFT

`data/private_patch_sft.py` turns exact before/after buffers from the user's two
private HF collections into focused excerpt repair examples. Pinned revisions
are in `private-patch-sources.json`. The downloader samples up to 24 shards per
repository by deterministic filename hash; it does not assume the collection's
README list means every planned repository was collected.

Each buffer must match its Git blob SHA-1. Examples contain a changed region plus
six context lines, the original commit instruction and file path, and the exact
corrected excerpt. Binary/missing/oversized buffers, generated paths, unchanged
pairs, common credential patterns and examples over 2,048 tokens are filtered.
Root license evidence is retained privately and referenced by hash. These checks
verify source integrity, **not** build or test correctness, and are not a complete
secret/PII audit. Only the corrected answer receives training loss.

Splits are by commit within repositories. All excerpts from a commit remain in
one split. All eligible validation excerpt hashes are blocked from training,
including validation examples beyond the sampled evaluation cap. Previous
CommitPack validation repositories and synthetic held-out code are excluded.
Normalized exact duplicate pairs are removed. There is no semantic clone audit
or claim of repository-disjoint or pretraining-clean evaluation.

The private sample caps training at 180,000 examples, 30,000 per language and
12,000 per repository, with at most 128 validation examples per language. A
bounded replay slice adds up to 2,000 previous CommitPack training examples per
language plus its previous validation examples. Replay excludes private
validation repositories and exact held-out code. Unsupported sparse languages
without private validation are omitted from private training.

From `training/`, after installing the documented extras and CA setup:

```sh
modal deploy modal_repair_scale.py
modal run modal_private_patch_sft.py::main \
  --sources sft/private-patch-sources.json \
  --data-id repair-private-patches-UNIQUE \
  --report-path /tmp/private-patch-report.json
modal run modal_private_patch_sft.py::launch \
  --data-id repair-private-patches-UNIQUE \
  --run-id repair-private-sft-UNIQUE \
  --report-path /tmp/private-patch-report.json
```

Preparation uses Modal CPU and the existing `simple-llm-hf` secret. A source
download failure aborts preparation instead of silently shrinking coverage.
The explicit launch requires at least 60,000 training examples across nine
languages. It submits a durable job to the deployed GPU worker: original PR #26
parent, one H100, batch 8, context 2,048, LR 1e-5, 20,000 total updates, evaluation
every 500 and checkpoints every 1,000. Each invocation has a four-hour training
loop limit, so inspect its actual completion count. Raw examples, detailed source
inventories and license evidence remain on the private Modal volume. Only
concise aggregate summaries belong in Git; complete generated reports belong on HF.

## Full private patch snapshot

`private-patch-full-sources.json` pins the expanded October 20 snapshots (1,025
patch shards). The CPU preparation command below processes every patch shard,
removes total/language/repository training caps, retains commit-disjoint splitting
and blocks the earlier private validation code hashes. Validation remains capped
at 128 excerpts per language plus CommitPack validation. Quality, provenance,
context-length and duplicate filters still apply; raw source files are not
silently treated as supervised repair examples. These are observed commit edits,
not execution-verified bug fixes. The training mixture follows eligible example
counts and is not automatically balanced across languages.

From `training/`, after deploying `modal_repair_scale.py`:

```sh
modal run --detach modal_private_patch_sft.py::main \
  --sources sft/private-patch-full-sources.json \
  --data-id repair-private-full-20261020-v1 \
  --report-path /tmp/private-full-report.json --full \
  --train-run-id repair-sft-private-full-20261020-r01
```

This explicitly launches a durable preparation job and, only after successful
strict validation and volume commit, submits one H100 training run. The run starts
from the original PR #26 model with fresh optimizer state, targets 40,000 updates
(batch 8, context 2,048, LR 1e-5), and has a four-hour training-loop limit. Sampling
is with replacement; 320,000 example draws do not guarantee one pass over every
eligible example. Checkpoints and validation occur every 1,000 updates. The
volume dataset directory contains `report.json`, `selection.json` and
`launch.json`; the latter records the actual training call ID. A preparation
failure never submits training. An early training time limit saves a checkpoint;
finishing the update budget would require an explicit matching resume.
