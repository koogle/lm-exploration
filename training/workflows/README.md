# Training workflows

A workflow file describes how a model is made, as a list of stages:

| Stage | Does | Module |
| --- | --- | --- |
| `data` | Prepare training data for a later pretraining or mid-training stage | [`stages/data.py`](stages/data.py) |
| `pretrain` | Train a model from scratch | [`stages/pretrain.py`](stages/pretrain.py) |
| `midtrain` | Continue the previous checkpoint on new data, context or schedule | [`stages/pretrain.py`](stages/pretrain.py) |
| `sft` | Supervised fine-tuning of the previous checkpoint | [`stages/sft.py`](stages/sft.py) |
| `rl` | Reinforcement learning from the previous checkpoint | [`stages/rl.py`](stages/rl.py) |

```json
{
  "name": "math-sft-20b",
  "family": "subword",
  "from": "climbmix-124m-wsd20b-02-s02/step-000203724.pt",
  "stages": [
    {"stage": "sft", "branches": [{"name": "metamath", ...}, {"name": "metamath-replay", ...}]}
  ]
}
```

A workflow may hold every stage or any part of them. One workflow goes from corpus to
pretrained model. Another holds only SFT stages that start from a pretrained
checkpoint, `from`. Each stage starts from the previous training stage's final
checkpoint. The first training stage that isn't pretraining starts from `from`. A
pretraining or mid-training stage names the data stage it trains on in `data`. The
last stage may list `branches`: stages of one kind that run side by side from the same
parent, such as SFT with and without replay. [`plan.py`](plan.py) resolves a workflow
file into its plan.

Every stage of a workflow belongs to one `family`:

| Family | `model_config` | Data stage | Sessions | Run folders |
| --- | --- | --- | --- | --- |
| `subword` | `workflows/model_configs/*.json` | `dataset_config` (`prepare_raw`) and `version`: a raw snapshot on `lm-exploration-mixture-data` | `trainer.subword` | `lm-exploration-training` |
| `hybrid` | `hybrid/configs/*.json`; owns its token target | `recipe`: token shards on `lm-exploration-hybrid-data` ([`hybrid/modal_data.py`](../hybrid/modal_data.py)) | `model_hybrid.train` | `lm-exploration-hybrid-runs` |

Post-training stages (`sft`, `rl`) exist for `subword`. File paths in a workflow are relative to
`training/`.

## Inspect and run

Run commands from `training/`. Inspect the resolved plan locally; this does not import
Modal, download data or launch work:

```sh
uv run python -m workflows --workflow workflows/climbmix-124m-wsd-20b.json --run-id my-run-01
```

Launch it on Modal. One launch runs the selected stages in order, within one 24-hour
deadline:

```sh
uv run modal run --detach -m workflows.modal_train::main \
  --workflow workflows/climbmix-124m-wsd-20b.json --run-id my-run-01 \
  [--start <stage>] [--end <stage>] [--checkpoint <run>/step-NNNNNNNNN.pt]
uv run modal run -m workflows.modal_train::stop --run-id my-run-01
```

- Stage `<name>` runs as `<run id>-<name>`. A relaunch of a stage that a launch left
  unfinished adds `-a2`, `-a3`, ...
- `/runs/<run id>/stages.json` on `lm-exploration-training` records each finished
  stage's output. Launching the same workflow and run ID again skips finished stages
  and continues the rest. A pretraining stage longer than one launch resumes its last
  checkpoint. A data stage continues its preparation. The plan's SHA256 must match:
  start a changed workflow under a new run ID.
- `--start` and `--end` select stages by name. `--checkpoint` is the parent of the
  first selected stage when its parent stage hasn't finished in this run, for
  example to run a workflow's SFT stages from another pretrained model.
- `stop` asks the running training session to save a checkpoint and exit. The
  workflow pauses, and the next launch continues it.
- An independent supervisor cancels every call and sandbox of the launch at its
  deadline ([`modal_app.py`](modal_app.py), [`deadline.py`](deadline.py)). A stage
  starts only if at least 30 minutes remain.

## Pretraining and mid-training stages

A stage entry holds the execution settings of `PretrainConfig`
([`config.py`](config.py)): GPU family and count, token target, data-review opt-in,
pilot mode, session length, qualification and continuation, and preparation
concurrency and overlap with training. It also holds `model_config` and its data.

| Files | Owns |
| --- | --- |
| `*.json` | Workflows: stages and their execution settings |
| [`model_configs/`](model_configs/README.md) | Architecture dimensions, context/batching, optimizer, learning-rate schedule, checkpoint/evaluation settings |
| `datasets/*.json` | Dataset root on the Modal volume, loader, and either a prepared snapshot or a corpus preparation recipe |

There is one owner per setting. Model configs cannot contain `world_size`,
`target_tokens`, `pilot`, `allow_unreviewed_data` or `data_loader`. The resolver
combines the selections into a validated `DistributedConfig` for the trainer, and
the plan carries their frozen contents and SHA256 provenance. Changing the GPU count
needs compatible global and microbatch sizes.

The controller runs bounded sessions (`torchrun -m <family trainer>` over
[`trainer.loop`](../trainer/README.md), each in a fresh `<stage run>-sNN` folder that
resumes the previous session's final checkpoint). It checks progress and validation
continuity between sessions, up to `max_segments`. Qualification uses the model's
final schedule with a short step limit, then continues the same checkpoint lineage.
`qualification_segments: 0` skips it.

Data:

- A subword stage without a data stage selects a `snapshot` dataset config:
  `snapshot: "VERSION/snapshots/NAME"`, a `raw` or `token_shards` loader and the
  `root` folder on the data volume. Format, tokenizer, review status,
  train/validation splits and non-pilot capacity are checked before a GPU is
  scheduled.
- A subword data stage prepares a `prepare_raw` dataset config: `loader: "raw"` and a
  corpus `recipe` (`training_data.simple_llm.load_recipe`). It goes into
  `<root>/<version>/snapshots/<stage run>`. The pretraining stage that names it sizes
  and checks the preparation. The two run together, and with `overlap_preparation`
  training starts on the growing snapshot. An exact compatible snapshot is reused.
- A hybrid data stage prepares its recipe's token shards. The training stage reads
  them under the recipe's name.

Continuing a checkpoint:

- `resume` in a pretraining stage continues a pretraining checkpoint exactly, for
  example a WSD decay branch (`climbmix-124m-wsd-20b-decay-5b.json`). The trainer
  requires identical settings, sources, dataset and PyTorch version.
- `midtrain` continues the previous checkpoint with new data, context or schedule.
  - `subword`: `seed_phase` (CPU, one-hour timeout) writes
    `<stage run>-seed/step-000000000.pt` with the parent's weights, optimizer state
    and RNG, then trains from it. See
    [continued pretraining](../trainer/README.md#continued-pretraining).
  - `hybrid`: starts from the parent's exported weights (`init_from`) with a fresh
    optimizer, so its run config may change the context length or drop MTP heads.

`h100x1-pilot.json` and `h100x2-pilot.json` share one model config and dataset config.
`h100x3-20b.json` is a template: replace its dataset's placeholder snapshot with a
prepared, reviewed, capacity-qualified snapshot before launch.
`model_configs/historical/` keeps earlier recipe files as configuration provenance.

## Post-training stages

An SFT stage entry holds its `data` (`{"volume": data-id}`, or `{"local": path}`
uploaded at launch), `gpu` (`L40S` or `H100`), an optional `resume` of an SFT
checkpoint with `extend`, and the trainer's `settings` ([`trainer/sft.py`](../trainer/sft.py)).
See the [SFT guide](../sft/README.md). The parent is a checkpoint of any stage, and
the trainer checks that its architecture and tokenizer fit
([`trainer/checkpoints.py`](../trainer/checkpoints.py)).

An RL stage entry holds its `pool`, its `gpus` (1–8 H100s, one process each) and the
trainer's `settings` ([`trainer/rl/`](../trainer/rl/__init__.py)); it runs model-written
code only in a network-blocked sandbox. A run that stops short of its step target leaves
the stage paused, and the next launch resumes it in the same run.
The parent checkpoint's architecture picks the model family's adapter
([`trainer/families.py`](../trainer/families.py)). Subword runs use the subword image and
the training volume; hybrid runs use the hybrid image, with parents and runs on the
hybrid runs volume and pools on the training volume. Subword workflows take SFT and RL
stages, hybrid workflows RL stages (hybrid chat SFT is a `midtrain` stage with a chat
run config). See the [RL guide](../rl/README.md).

`math-sft-20b.json` and `math-sft-5b.json` run math SFT of the WSD end points, with
and without pretraining replay. `hybrid-720m-rl.json` runs chat SFT of the hybrid 720M 40B
export, then RL from it.
