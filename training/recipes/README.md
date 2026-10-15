# Training recipes

A recipe is one JSON file that names everything a training run needs:

| Key | Meaning |
| --- | --- |
| `model.architecture` | Installed architecture package, e.g. `model_10m` (`10m/model.py`) or `model_100m` |
| `model.config` | Overrides for that architecture's `ModelConfig` (e.g. `block_size`, `n_layer`). `vocab_size` always comes from the dataset tokenizer |
| `dataset` | A training-eligible `id` from [`data/datasets.json`](../data/datasets.json) that has a `fetch` module |
| `training` | `TrainConfig` fields: `steps`, `batch_size`, `lr`, `seed` and eval/log/checkpoint cadence |
| `backend` | Only `"kind": "modal"`. `gpu` (default `L4`), `cpu`, `memory_mib`, `max_seconds` (cooperative training limit, default 1,200) and `timeout_seconds` (hard worker limit and budget reservation, default 1,500) |

Recipes are validated locally before anything starts: unknown keys, invalid model
dimensions, ineligible datasets and bad settings fail without a Modal call.
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
`results/<run-id>.json`. The image copies the whole uv project, so new
architectures and fetchers need no launcher changes. They need a `pyproject.toml`
package entry.

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
