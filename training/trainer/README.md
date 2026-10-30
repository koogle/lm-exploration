# Trainer implementation

[`loop.py`](loop.py) is the training loop for every model family on one host (1–8
GPUs): rank-zero data reading and batch scatter, token-weighted gradient
accumulation under DDP, the learning-rate factor, clipping, evaluation, logging,
durable checkpoints and exact resume checks. A family adapter supplies the model,
data, loss, evaluation and exports (the interface is in the module docstring):

| Family | Adapter and entry point | Settings | Token budget |
| --- | --- | --- | --- |
| Subword GPT (10M/100M/124M) | [`subword.py`](subword.py), `torchrun -m trainer.subword` | `distributed.config.DistributedConfig` | loss targets |
| Hybrid KDA/attention | [`../hybrid/train.py`](../hybrid/train.py), `torchrun -m model_hybrid.train` | `model_hybrid.train.RunConfig` | window tokens |

[`optimizers.py`](optimizers.py) is the one optimizer: Muon for the matrices a
module declares `muon` and AdamW for the rest (`adamw_decay`, `adamw_no_decay`),
from the `optimizer()` declarations of the modules that own the parameters
([`roles.py`](roles.py)); each group keeps a base rate and the loop applies one
schedule factor to all of them. SFT and RL ([`sft.py`](sft.py), [`rl/`](rl/__init__.py))
use it too, and pick the model family from the parent checkpoint's architecture
([`families.py`](families.py): `SubwordModel` in `subword.py`, `HybridModel` in
[`../hybrid/post_train.py`](../hybrid/post_train.py)).

A run folder holds `config.json`, `metrics.jsonl` (one row per step),
`validation.jsonl` (start, every `evaluate_every` steps, end), `step-<step>.pt`
checkpoints with `latest.json`, and `report.json`. A session resumes from an
explicit checkpoint or from the folder's `latest.json`, only with the same
settings, GPU count, data identity, source files (loop, optimizer, family, model,
loaders) and PyTorch version, or a change the family accepts (a subword WSD decay
branch).

[`../distributed/`](../distributed/README.md) owns the subword configuration, per-rank
runtime/rendezvous settings, the Modal image, GPU qualification and the launcher;
its worker runs `torchrun -m trainer.subword`. Training recipes live in
[`../workflows/model_configs/`](../workflows/model_configs/README.md), and
preparation/qualification/continuation live in [`../workflows/`](../workflows/README.md).

Run commands from `training/`:

```sh
uv run python -m trainer.subword --help
uv run python -m unittest discover -s tests -p 'test_distributed_*.py' -v
```

See the [distributed launch guide](../distributed/README.md) for job and resume
commands. Source identity must match for exact resume; use original source and
settings for historical runs.

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

## Training loss

Training and evaluation use `loss.cross_entropy`, which avoids Softmax Collapse
([reading note](../references/softmax-fix-2026-10-24.md)). On CUDA it runs the fused
kernel in `kernels/stable_ce.cu` (wrapped as a custom op by `stable_ce_cuda.py`); on
CPU it runs the PyTorch port in `stable_softmax.py`. The dense loss in
`backward_batch`, the model's chunked loss (`loss_fn` argument) and `evaluate_sources`
all go through it. The shared Modal image (`../distributed/image.py`) is a CUDA devel
image that prebuilds the kernel into `/opt/stable_ce`; a source change that does not
match that build is JIT-compiled instead. Measured on the 124M recipe with three H100s:
8.7% more tokens/s and 8 GiB less peak memory per GPU than `F.cross_entropy`, with
matching losses (details in the reading note).

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

A workflow's `midtrain` stage seeds the new phase from the previous stage's checkpoint
(or the workflow's `from`) in the training image and then trains it (see
[workflows](../workflows/README.md)).
Locally, with the PyTorch build of the training image:

```sh
uv run python -m trainer.continued_pretraining --parent OLD/step-000123456.pt \
  --output NEW-seed/step-000000000.pt --manifest SNAPSHOT/manifest.json --config phase.json
```
