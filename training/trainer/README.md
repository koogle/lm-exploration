# Trainer implementation

[`loop.py`](loop.py) is the single active training loop for one, two or three GPUs
on one host. It owns model/optimizer construction, token-weighted gradient
accumulation, evaluation, logging, durable checkpoints and exact resume checks.
Both raw and seekable-token mixtures use this loop. Training runs on Modal CUDA
workers; local tests check mathematical correctness separately from GPU performance.

[`../distributed/`](../distributed/README.md) owns shared configuration, per-rank
runtime/rendezvous settings, the Modal image, GPU qualification and the launcher.
Its worker runs `torchrun -m trainer.loop`; the trainer accepts an explicit
`DistributedRuntime`. Training recipes live in [`../workflows/model_configs/`](../workflows/model_configs/README.md),
and preparation/qualification/continuation live in [`../workflows/`](../workflows/README.md).

Run commands from `training/`:

```sh
uv run python -m trainer.loop --help
uv run python -m unittest discover -s tests -p 'test_distributed_*.py' -v
```

See the [distributed launch guide](../distributed/README.md) for job and resume
commands. No training behavior or checkpoint compatibility checks were relaxed by
moving the loop; source identity must still match for exact resume. Use original
source and settings for historical runs.

## Evaluation during training

Every evaluation row in `validation.jsonl` (and `initial_validation`/`final_validation` in
`report.json`) has `per_source` and `weighted_loss`, plus `per_stratum` losses,
`target_counts`, `stratum_target_counts` and `windows`. The config's `heldout` setting
chooses the evaluation set explicitly:

- `derived` (default): the first `validation_windows_per_source` windows of each source's
  validation split in our snapshot. Diagnostics, not capability evaluation.
- `frozen`: the file the snapshot manifest's `heldout` entry names. Fixed windows labelled
  with a source and a stratum such as `code/python/tests`, checked by checksum, tokenizer
  and coverage ([`data/heldout.py`](../data/heldout.py) writes and loads it).

A mismatch between the setting and the manifest fails instead of falling back. `--stop-tokens N` ends a session at N targets with stop reason
`evaluation_boundary` and leaves the schedule on `target_tokens`, so the checkpoint
resumes the same run.

## Other modules

`gpu_benchmark.py` and `ddp_benchmark.py` are bounded throughput experiments,
separate from the training loop. The character training loop, its recipe schema,
format-1 checkpoint loader and the `lm-generate` CLI have been removed (last present
at revision `5083671`). Historical Shakespeare recipe files and measured evidence are
preserved.

Muon/AdamW implementation lives in `optimizers.py`; see the
[matched comparison](../distributed/MUON_COMPARISON.md) for measured context.

## Continued pretraining

[`continued_pretraining.py`](continued_pretraining.py) starts a new data phase from a
checkpoint of this trainer. It writes a seed checkpoint that the loop resumes like any
other. What the seed keeps and resets:

| Kept from the parent | Reset or taken from the new config |
| --- | --- |
| Weights | Data cursors and exposures (new snapshot) |
| Complete optimizer state, as saved: AdamW `exp_avg`, `exp_avg_sq` and `step`; for Muon, momentum buffers plus the auxiliary AdamW state | `step` and `tokens` (0), so warmup and cosine decay restart over the new `target_tokens` |
| Every rank's Python, NumPy, CPU and CUDA RNG state | Optimizer hyperparameters (weight decay, betas, Muon momentum and LR ratio) |
| | `config`, `source_sha256`, `torch_version` and `data_identity` |

The parent must use the same architecture, world size and optimizer as the new config,
the frozen tokenizer, and finite weights and optimizer state. The seed path must not
exist. `lineage` records the parent checkpoint and hash, its step, config, data and
source identity, and `parent_tokens`, the lifetime targets before this phase. Later
checkpoints keep the lineage, and `report.json` adds `lifetime_loss_tokens`.

Through the workflow launcher, `--phase-from OLD_RUN/step-NNNNNNNNN.pt` seeds the new
phase in the training image and then trains it (see [workflows](../workflows/README.md)).
Locally, with the PyTorch build of the training image:

```sh
uv run python -m trainer.continued_pretraining --parent OLD/step-000123456.pt \
  --output NEW-seed/step-000000000.pt --manifest SNAPSHOT/manifest.json --config phase.json
```
