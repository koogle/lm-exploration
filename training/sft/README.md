# Python repair SFT pilot

Stacked on PR #26, initialized from its completed 124M / 10.879B checkpoint:
`simple-llm-code60-fresh-5b-20261019-r02-s02/step-000066065.pt`.
This is a new post-training experiment: AdamW starts fresh; pretraining optimizer
moments are intentionally not carried into SFT. Model weights, architecture, and
frozen tokenizer are inherited. Nothing is launched automatically.

## Data contract

Supply a UTF-8 JSONL file, one repair per line:

```json
{"id":"repo-a-fix-1","group":"repo-a","split":"train","instruction":"Return the sum of a and b.","broken":"def add(a, b):\n    return a - b\n","fixed":"def add(a, b):\n    return a + b\n","source":"repository URL + base/fix commit IDs","license":"MIT"}
```

Include separate `validation` records with different groups and code. Use repository
or related-change family as `group`; assign splits before generating variants.
Required fields are nonempty strings. Both splits must be present. Prompt text and
broken code are context only; corrected code and EOS are supervised. Padding is
ignored. Overlength examples fail rather than silently truncating solutions.
Source and license are provenance declarations, not automated eligibility checks.

For PR-derived data, separate implementation edits from test edits. Select short,
single-file Python repairs whose relevant test fails at the base commit and passes
with the reference repair. Record this verification alongside the source dataset.
The importer does not execute arbitrary supplied code or certify correctness;
only feed it independently verified repairs. No external corpus is bundled.

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
loss, full validation answer-token loss before and after training, and periodic
full checkpoints. Unique run IDs prevent overwriting existing runs.

Resume a time-bounded run in a **new** output run ID using `--resume
repair-sft-pilot-01/step-000000050.pt`, with the same parent, dataset, source and
settings. Steps are the total target, not additional steps. Resume restores model,
optimizer, random example sampler, CPU/CUDA RNG and answer-token counters.
Each draw samples training examples uniformly with replacement; exposure is not a
unique-data count. A time limit is checked between updates, not inside validation.

## Evaluation and current limits

This initial loop evaluates held-out **answer-token loss**, not executable repair
success. Before selecting a useful model, run generated repairs in isolated
sandboxes against fixed hidden tests and compare base/SFT/Qwen on identical tasks.
Track syntax validity, EOS termination, test pass rate, and edit size. Never run
untrusted generated code in the training container or let it edit grading tests.
The SFT checkpoint format is `repair-sft-v1`; the existing pretraining publication
scripts do not yet accept it. Distributed SFT, generation evaluation, sandbox test
execution, and automated PR mining are follow-on work, not implemented features.
