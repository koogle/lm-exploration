# Training workflows

A model goes through optional stages: data preparation and pretraining, [mid-training],
[SFT], [RL]. Every training launch is a workflow file with a `stage`, run by one
controller, `workflows.modal_train`, under one 24-hour deadline and an independent
supervisor ([`modal_app.py`](modal_app.py)) that stops the launch's calls and sandboxes:

| Stage | Workflow file | Runs |
| --- | --- | --- |
| `pretrain` | `workflows/*.json` ([below](#pretraining)) | [`trainer.loop`](../trainer/README.md): prepare or reuse a dataset, qualify, resume bounded sessions; continued from a checkpoint with a new data mix, it is mid-training |
| `sft` | an SFT stage file, e.g. [`../sft/recipes/`](../sft/recipes/) ([format](../sft/README.md#train)) | [`trainer/sft.py`](../trainer/sft.py) on one L40S or H100, or its resume qualification |
| `rl` | an RL stage file, e.g. `../rl/recipes/` ([format](../rl/README.md#code-layout)) | [`trainer/rl/`](../trainer/rl/__init__.py) on one H100 with the code sandbox |
| `pipeline` | post-training stages one after another ([below](#pipelines)) | each stage from the previous stage's final checkpoint |

```sh
# Any stage: inspect the resolved workflow locally (no Modal, data or GPU), then launch.
uv run python -m workflows --workflow sft/recipes/math-metamath-replay-20b.json
uv run --all-extras modal run -m workflows.modal_train --workflow sft/recipes/math-metamath-replay-20b.json
```

## Post-training stages

A stage file names one run of one post-training stage
([`stages/`](stages/__init__.py) validates every stage's file the same way): `stage`,
`run_id`, `parent` and `settings`, optional `name`, `description` and `provenance`, plus
the stage's own inputs (SFT: [`stages/sft.py`](stages/sft.py), RL: [`stages/rl.py`](stages/rl.py)). The `parent` may be a
checkpoint of any stage: [`../trainer/checkpoints.py`](../trainer/checkpoints.py) reads
pretraining and post-training checkpoints alike, and a parent is refused only when its
model code or tokenizer differs from the trainer's. Post-training checkpoints record
their `stage` and their parent.

Launches ([`modal_post_train.py`](modal_post_train.py)) take `--run-id` to replace the
file's run ID, SFT also `--resume <run>/step-NNNNNNNNN.pt` and `--qualify N` (an
uninterrupted N-step run against a stop at N − 1, fresh-container restore and continue,
compared exactly). Their local result goes to `artifacts/<stage>/<run-id>.json`, and the
launch status to `/runs/<run-id>/workflow.json`.

## Pipelines

A pipeline ([`pipeline.py`](pipeline.py)) runs post-training stages one after another
from any checkpoint:

```json
{
  "stage": "pipeline",
  "name": "math-sft-20b",
  "from": "climbmix-124m-wsd20b-02-s02/step-000203724.pt",
  "steps": [
    {"branches": ["sft/recipes/math-metamath-20b.json", "sft/recipes/math-metamath-replay-20b.json"]}
  ]
}
```

([`../sft/pipelines/math-sft-20b.json`](../sft/pipelines/math-sft-20b.json): the r01 and r02
math SFT runs of the 20B end point, side by side.) A longer pipeline lists more steps,
for example SFT then RL.

- `from` is the first stage's parent, so a pipeline can start after pretraining,
  mid-training, SFT or RL.
- Each step is one stage file (relative to `training/`) or `branches`, several stages
  run side by side from the same parent; only the last step may branch.
- A stage's own `parent` and `run_id` are replaced: the parent by the previous step's
  final checkpoint, the run ID by `<pipeline run>-<step>-<file stem>`.

```sh
uv run python -m workflows --workflow PIPELINE.json --run-id ID   # the resolved plan
uv run --all-extras modal run -m workflows.modal_train --workflow PIPELINE.json --run-id ID
```

A launch that ends before the pipeline does (its deadline, a failure) leaves
`/runs/<pipeline run>/pipeline.json` with each finished stage and its checkpoint.
Launching the same pipeline file with the same run ID continues with the unfinished
stages; a stage that did not finish is rerun from its parent as `<run ID>-a<attempt>`.
The plan must be unchanged.

## Pretraining

A pretraining workflow selects a model configuration and dataset configuration, then
owns the execution settings.

| Files | Owns |
| --- | --- |
| `*.json` | Model/dataset selection, GPU family/count, token target, data-review opt-in, pilot mode, session/qualification/continuation, preparation concurrency and overlap with training |
| [`model_configs/`](model_configs/README.md) | Architecture dimensions, context/batching, optimizer, learning-rate schedule, checkpoint/evaluation settings |
| `datasets/*.json` | Dataset root on the Modal volume, loader, and either a prepared snapshot or a corpus preparation recipe |

There is one owner per setting. Model configs cannot contain `world_size`,
`target_tokens`, `pilot`, `allow_unreviewed_data` or `data_loader`. The resolver
combines the three selections into a validated `DistributedConfig` for the trainer.
Changing GPU count still requires compatible global/microbatch sizes.

## Inspect and run

Run commands from `training/`. Inspect the complete resolved configuration locally;
this does not import Modal, download data or launch work:

```sh
uv run python -m workflows --workflow workflows/h100x1-pilot.json
uv run python -m workflows --workflow workflows/h100x1-pilot.json \
  --model-config model_configs/modern-124m.json \
  --dataset-config datasets/pilot-tokens.json
```

Model/dataset references and selector overrides are relative to the workflow file's
directory. Preparation recipes are relative to `training/`. Config files are loaded
explicitly; continuation jobs receive their frozen contents and SHA256 provenance
rather than reloading whichever files happen to exist later.

The generic Modal entrypoint accepts the same selections:

```sh
# Existing prepared token snapshot, one GPU.
uv run modal run --detach -m workflows.modal_train \
  --workflow workflows/h100x1-pilot.json --run-id my-pilot-01

# Prepare/reuse a raw corpus: select a workflow whose dataset uses prepare_raw.
uv run modal run --detach -m workflows.modal_train \
  --workflow workflows/MY-WORKFLOW.json --run-id my-training-01 --version my-data-v1
```

`h100x1-pilot.json` and `h100x2-pilot.json` share one model config and dataset config.
No shipped workflow prepares a raw corpus: the 12-source scholarly/code recipe and
its `small-5b.json` workflow were removed with that mixture.
`h100x3-20b.json` is a template: replace its dataset's placeholder snapshot with a
prepared, reviewed, capacity-qualified snapshot before launch. These configurations
are not newly measured or deployed results.

## Select datasets

- `mode: "snapshot"` requires `snapshot: "VERSION/snapshots/NAME"` and a loader of
  `raw` or `token_shards`. `root` selects the directory under the mounted data
  volume. Format, tokenizer, review status, train/validation availability and
  non-pilot capacity are checked before scheduling a GPU.
- `mode: "prepare_raw"` requires `loader: "raw"` and `recipe`, a project-relative
  corpus recipe accepted by `training_data.simple_llm.load_recipe`. Supply a fresh
  `--version` when changing staging identity. The workflow reuses an exact compatible
  snapshot or stages/exports a new one. The selected recipe's contents are frozen
  in the resolved workflow, including its pinned upstream sources.

Create another dataset JSON to select another snapshot or supported corpus recipe;
no controller edit is needed. Both modes currently use the project's frozen 32K
tokenizer and existing data adapters. Arbitrary tokenizer formats, source schemas
and character training are not implemented by this configuration change.

## Continuation and recovery

Qualification uses the selected model's final training schedule with a short step
limit, then continues the same checkpoint lineage. There is no separate active
qualification recipe. Set `qualification_segments: 0` to skip that phase.

The controller schedules the workflow's GPU count/family and timeout for each
segment, checks progress and validation continuity, and continues up to
`max_segments`. The model configuration and dataset selection stay fixed.
To resume a stopped checkpoint explicitly, select its prepared snapshot dataset
and launch with a fresh run ID and `--resume OLD_RUN/step-NNNNNNNNN.pt`.
The trainer still requires identical settings, source, dataset and PyTorch version.

To continue pretraining on new data or with a new schedule, select the new phase's
snapshot dataset and model config and launch with a fresh run ID and
`--phase-from OLD_RUN/step-NNNNNNNNN.pt`. The launcher first runs `seed_phase` (CPU,
one-hour timeout, no retries) in the training image. It writes
`RUN_ID-seed/step-000000000.pt` with the parent's weights, optimizer state and RNG, then
trains from that seed. The lineage is recorded in `RUN_ID/workflow.json`. See
[continued pretraining](../trainer/README.md#continued-pretraining) for what carries over.

Interrupted preparation can use `--resume-preparation` with its original run ID,
version and identical resolved selection, provided training has not begun. With
`overlap_preparation`, preparation that stops after its training started also resumes
this way. A growing snapshot can be selected for `--resume` once its preparation has
written `complete.json`.
Historical recipe files are retained under `model_configs/historical/`; the old
`workflows.modal_small_5b` module and standalone recipe-launch CLI are replaced by
`workflows.modal_train`. Existing remote resource names and measured artifacts
remain unchanged. The new controller requires explicit deployment for deployed use.
