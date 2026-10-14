# Trainer implementation

The trainer is an installed Python package within the `training/` uv project.
It uses the fixed architecture in `../100m/model.py`; internal tiny-model test
configurations are not exposed through the CLI or UI.

## Responsibilities

- `../cli.py`: thin `lm-train` and `lm-generate` entry points. Parses `--config` and `--resume`,
  loads settings and calls the trainer. Importing it or asking for help does not
  load the training loop or start a run.
- `config.py`: `TrainConfig`, training-setting validation and the training root.
  Dataset and output paths are anchored to `training/`, not this package folder.
- `loop.py`: run setup, model/optimizer construction, evaluation and the core
  optimizer-step loop, metrics, preview generation and cooperative SIGINT/SIGTERM
  handling at optimizer-step boundaries. Checkpoint operations are delegated to
  `checkpoints.py`.
- `checkpoints.py`: trusted checkpoint loading, resume compatibility checks,
  restoring model/optimizer/RNG state and atomic checkpoint saving.
- `sampling.py`: shared sampling for training previews and `lm-generate`.
- `artifacts.py`: atomic writes for `config.json`, `sample.txt` and
  `checkpoint.pt`, plus the source/runtime provenance stored in `config.json`.

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
uv run lm-train --config path/to/training.json --resume runs/<id>/checkpoint.pt
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
