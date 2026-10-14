# Trainer implementation

The trainer is an installed Python package within the `training/` uv project.
The local CLI/UI uses `../100m/model.py`. The bounded Shakespeare Modal worker
explicitly supplies `../10m/model.py` through the architecture-module argument.
Tiny test configurations remain internal; see [10M training](../10m/README.md).

## Responsibilities

- `../cli.py`: thin `lm-train` and `lm-generate` entry points. Parses `--config` and `--resume`,
  loads settings and calls the trainer. New runs always go to `runs/<uuid>/`;
  the UI passes a hidden `--run-dir` for the folder it pre-creates. Importing it or asking for help does not
  load the training loop or start a run.
- `config.py`: `TrainConfig`, training-setting validation and the training root.
  Dataset and output paths are anchored to `training/`, not this package folder.
- `loop.py`: run setup, model/optimizer construction, evaluation and the core
  optimizer-step loop, metrics, preview generation, optional wall-time bounds and cooperative SIGINT/SIGTERM
  handling at optimizer-step boundaries. Checkpoint operations are delegated to
  `checkpoints.py`.
- `checkpoints.py`: trusted checkpoint loading, resume compatibility checks,
  restoring model/optimizer/RNG state, checkpoint saving and the source hashes,
  git commit and runtime details stored in `config.json` and each checkpoint.
- `sampling.py`: shared sampling for training previews and `lm-generate`.

## Commands

Run from `training/`:

```sh
uv sync --locked
uv run lm-checks
uv run python -m training_data.fetching.fetch_tinyshakespeare
uv run lm-train

# Optional JSON settings, also used by the UI.
uv run lm-train --config path/to/training.json

# Resume a trusted checkpoint with an increased total step target.
uv run lm-train --config path/to/training.json --resume runs/<id>/checkpoints/step-000100.pt
```

`uv run python -m cli` invokes the same command. Edit `TrainConfig` in
`config.py` for normal settings and `100m/model.py` for architecture dimensions.
The default benchmark is the ignored `data/raw/tinyshakespeare.txt`.
Full-sized training needs several GiB; `lm-checks` uses tiny CPU models and a
meta-device parameter count rather than a full-sized training run.

Checkpoint format version 1 is preserved. Resume requires identical data,
tokenizer, model and fixed optimizer/batch/seed/device settings. Total steps may
increase; logging, evaluation and checkpoint cadence can change. Only load
trusted checkpoints: RNG restoration requires Python objects, so checkpoint
loading is not safe for untrusted files. Exact CPU resume is checked within the
same runtime; cross-device reproducibility is not promised.
