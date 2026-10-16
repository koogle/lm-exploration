# Training recipes

A recipe is one JSON file that is validated by the pydantic `Recipe` model in
[`trainer/recipe.py`](../trainer/recipe.py):

| Key | Meaning |
| --- | --- |
| `model` | `"10m"`, `"100m"` or `"100m_tied"`: a key of `MODELS`. See the [embedding-tying experiment](../100m_tied/README.md). |
| `dataset` | `"tiny-shakespeare"` or `"shakespeare-complete"`: a key of `DATASETS` |
| `training` | `TrainConfig` fields: `steps`, `batch_size`, `lr`, `seed` and eval/log/checkpoint cadence |
| `model_overrides` | Optional `ModelConfig` overrides (e.g. `block_size`, `n_layer`). `vocab_size` always comes from the dataset |
| `backend` | Optional Modal resources: `gpu` (one of `T4`, `L4`, `A10G`, `L40S`, `A100`, `A100-80GB`, `H100`, `H200`, `B200`; default `L4`), `cpu`, `memory_mib`, `max_seconds` (cooperative training limit, default 1,200) and `timeout_seconds` (hard worker limit and budget reservation, default 1,500) |
| `description` | Optional free text |

Recipes are validated locally before anything starts: unknown keys, wrong types,
an unlisted model, dataset or GPU, invalid model dimensions, bad settings and
datasets not marked `training_eligible` fail without a Modal call.
`--smoke` keeps the model and dataset but uses 5 updates of batch 2, a 60-second
training limit and a 180-second worker timeout.

## Launch

Run from `training/`. Each run needs a new run ID.

```bash
uv run --with 'modal[api-proxy-support]==1.6.1' modal run modal_train.py \
  --recipe recipes/shakespeare-complete-10m.json --run-id shakespeare-complete-10m-smoke-01 --smoke
uv run --with 'modal[api-proxy-support]==1.6.1' modal run --detach modal_train.py \
  --recipe recipes/shakespeare-complete-10m.json --run-id shakespeare-complete-10m-train-01
```

The worker fetches and verifies the dataset, trains on one GPU without retries,
selects the saved checkpoint with the lowest validation loss as `best.pt`, and
writes `report.json` (including the resolved recipe) to `<run-id>/` on the
`lm-exploration-training` volume. The launcher also saves the report to
`results/<run-id>.json`. The image copies the whole uv project. To add a model or
dataset, add its entry to `MODELS` or `DATASETS` and its name to `ModelName` or
`DatasetName` in `trainer/recipe.py` (a new
model folder also needs a `pyproject.toml` package entry).

Before launching, the launcher reserves the full worker timeout at the ledger's
conservative GPU-hour rate in the ignored `runs/modal-budget.json`. It refuses
reused run IDs and reservations over `cap_usd`. This is reservation accounting,
not an actual bill. Restore the ledger from a previous checkout, or create it
with an approved cap:

```json
{"cap_usd": 10.0, "conservative_gpu_hour_usd": 5.0, "reservations": []}
```

## Recipes

- `shakespeare-10m.json`: the settings of the published `shakespeare-10m-train-01`
  run (executed with the former `modal_shakespeare.py`, commit `ec5ebd1`).
- `shakespeare-complete-10m.json`: the same architecture and settings on Complete
  Shakespeare. **Not yet run.**

## Current limits

The trainer is generic over architectures that expose `GPT(config)` and a
`ModelConfig` with `vocab_size` and `block_size`, and over any single UTF-8 text
dataset. It is still fixed to a character tokenizer, one dataset per run (no
mixtures), a contiguous 90/10 split, FP32 AdamW with fixed betas, weight decay and
clipping, no learning-rate schedule, and one GPU.

`100m` and `100m_tied` use the [modern architecture](../100m/README.md) by default
(RMSNorm, RoPE, SwiGLU). Set `model_overrides: {"architecture_version": 1}` for
historical vanilla blocks. Older checkpoints without a version load as version 1;
new checkpoints record the version and cannot resume across different versions.
