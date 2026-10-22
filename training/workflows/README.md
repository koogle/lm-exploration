# Training workflows

A workflow selects a model configuration and dataset configuration, then owns the
execution settings. The same controller can prepare or reuse a dataset, qualify
training, and resume bounded sessions through [`trainer.loop`](../trainer/README.md).

| Files | Owns |
| --- | --- |
| `*.json` | Model/dataset selection, GPU family/count, token target, data-review opt-in, pilot mode, session/qualification/continuation and preparation concurrency |
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
uv run python -m workflows --workflow workflows/small-5b.json
uv run python -m workflows --workflow workflows/small-5b.json \
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

# Prepare/reuse the selected raw corpus, then train the selected small model.
uv run modal run --detach -m workflows.modal_train \
  --workflow workflows/small-5b.json --run-id my-training-01 --version my-data-v1
```

`h100x1-pilot.json` and `h100x2-pilot.json` share one model config and dataset config.
`small-5b.json` preserves the existing three-H100/5B workflow's settings.
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
version and identical resolved selection, provided training has not begun.
Historical recipe files are retained under `model_configs/historical/`; the old
`workflows.modal_small_5b` module and standalone recipe-launch CLI are replaced by
`workflows.modal_train`. Existing remote resource names and measured artifacts
remain unchanged. The new controller requires explicit deployment for deployed use.
