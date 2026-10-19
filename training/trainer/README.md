# Trainer implementation

The trainer is an installed Python package within the `training/` uv project.
Training runs only on Modal GPU workers: [`../modal_train.py`](../modal_train.py)
validates a [recipe](../recipes/README.md) locally and calls `recipe.run_recipe()`
on the worker, which supplies the architecture (e.g. `../10m/model.py`) to the loop.
There is no CPU or MPS training path.

## Responsibilities

- `../modal_train.py`: the Modal app, image, budget reservation and launcher.
- `../generate.py`: model-agnostic local `lm-generate`; loads the architecture named
  by a checkpoint's `model_name` and streams samples.
- `config.py`: the pydantic `TrainConfig` (validated hyperparameters) and the fixed CUDA device.
- `loop.py`: run setup, model/optimizer construction, evaluation and the core
  optimizer-step loop, metrics, preview generation, optional wall-time bounds and cooperative SIGINT/SIGTERM
  handling at optimizer-step boundaries. Checkpoint operations are delegated to
  `checkpoints.py`.
- `checkpoints.py`: trusted checkpoint loading, resume compatibility checks,
  restoring model/optimizer/RNG state, checkpoint saving and the source hashes,
  git commit and runtime details stored in `config.json` and each checkpoint.
- `recipe.py`: the pydantic `Recipe`/`Backend`/`MetricRow` models, the fixed `MODELS` and `DATASETS` tables,
  dataset fetching, best-checkpoint selection and the run report.
- `sampling.py`: shared sampling for training previews and `lm-generate`.

Checkpoint format version 1 is preserved. The loop still supports resume
(`train(..., resume=<checkpoint>)`), which requires identical data, tokenizer, model and fixed
optimizer/batch/seed/device settings, but no launcher exposes it yet. Only load
trusted checkpoints: RNG restoration requires Python objects, so checkpoint
loading is not safe for untrusted files.

## Distributed subword training

[`DISTRIBUTED.md`](DISTRIBUTED.md) documents the separate one/two/three-GPU
NCCL trainer, token-budget stopping, durable checkpoints and explicit cross-job
resume through `../modal_distributed_train.py`. It uses the frozen 32K tokenizer
and prepared mixture snapshots. The character trainer above is unchanged.

The [fresh Muon/AdamW comparison](MUON_COMPARISON.md) adds a matrix-only Muon
optimizer with an AdamW companion and a matched, bounded eight-H100 launcher.

New distributed training configs default to **Muon for hidden matrices plus
fused AdamW for embeddings, biases and norms**. Use `optimizer="adamw"` for the
all-AdamW baseline. See [distributed training](DISTRIBUTED.md#optimizer-default)
for the optimized eight-H100 recipe and resume behavior.
