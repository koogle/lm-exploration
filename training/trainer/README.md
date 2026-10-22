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

`gpu_benchmark.py` and `ddp_benchmark.py` are bounded throughput experiments,
separate from the training loop. `sampling.py` supports local generation.
`recipe.py`, `config.py` and `checkpoints.py` retain historical character-recipe
validation and format-1 checkpoint interpretation for existing models and the
[`lm-generate` CLI](../generate.py). The character training loop, checkpoint writer
and Modal recipe launcher have been removed. Historical Shakespeare recipe files
and measured evidence are preserved.

Muon/AdamW implementation lives in `optimizers.py`; see the
[matched comparison](../distributed/MUON_COMPARISON.md) for measured context.
