# Resumable subword training on multiple Modal GPUs

`distributed_training/modal_train.py` runs the modern 124.4M model on **one, two or three
GPUs on one host**, using PyTorch NCCL DistributedDataParallel. It trains a shared
model with synchronized gradients. It is separate from the historical character
trainer and from the throughput-only benchmarks.

## One implementation, including one GPU

All subword training code, Modal launchers, the shared image and recipes live in
this package. The duplicate `modal_simple-llm_pilot.py` / `trainer/subword_pilot.py`
path has been removed; historical pilot results remain in `results/` and its
implementation remains in Git history. Character training and throughput-only
benchmarks are separate experiments, not alternate subword training paths.

For one GPU, set `world_size` to 1: the very same `train()` and `backward_batch()`
run under one-rank DDP, with the same model, loader, global batch, valid-token loss
normalization, optimizer, LR schedule and checkpoint format. There is no special
single-GPU loop. With matched settings, multiple GPUs divide that same global
batch, subject to floating-point reduction differences. This is not a claim of
bitwise equality across GPU counts or reproduction of the removed pilot's
step-based warmup/constant learning-rate schedule.

```sh
# One-GPU bounded pilot, using the same implementation as the two-GPU recipe.
modal run --detach -m distributed_training.modal_train \
  --config distributed_training/recipes/h100x1-pilot.json \
  --run-id my-single-gpu-pilot-01 \
  --snapshot simple-llm-124m-20261016/snapshots/pilot-001-seekable \
  --gpu H100 --max-steps 2 --max-seconds 300
```

## How the GPUs work together

![Single-host data-parallel training: rank zero scatters a global batch to three full model replicas, DDP synchronizes gradients, and a durable checkpoint enables explicit resume.](images/distributed-training.png)

The diagram shows the three-H100 configuration. One and two GPUs use the same
flow. This is **data parallelism**: every GPU holds the entire model and its own
AdamW optimizer state. Adding GPUs divides the batch computation; it does not
split the model across devices or increase the configured global batch.

1. The [Modal launcher](modal_train.py) allocates one host and
   calls the remote worker through the Modal Python API. The worker supervises a
   `torchrun` subprocess, which starts one Python process per GPU. Each process
   calls `train()` and is a *rank*; NCCL carries communication between ranks.
   The worker captures logs and enforces a process timeout, terminating the
   process group if needed.
   The training entrypoint resolves `RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
   `MASTER_ADDR` and `MASTER_PORT` once into a typed `DistributedRuntime` and
   passes it explicitly to `train(..., runtime=runtime)`. Training uses its
   explicit rank, GPU index, world size and TCP rendezvous endpoint. The shared
   JSON config supplies the requested GPU count; the runtime count must match it.
   Per-process ranks and rendezvous addresses belong to the launch context and
   are not saved as shared training settings.
2. Rank zero alone reads the deterministic token mixture and constructs the
   global input/target batch. For the illustrated configuration, 48 windows of
   context 2,048 are scattered as 16 windows per GPU. Rank zero also counts all
   valid loss targets; padding is excluded.
3. Each rank runs forward and backward passes on its slice. If that slice needs
   multiple microbatches, `no_sync()` accumulates local gradients until the last
   microbatch triggers DDP synchronization. Computation uses BF16 autocast while
   model parameters remain FP32.
4. **Before backward**, each microbatch's summed loss is multiplied by
   `world_size / global_valid_tokens`. DDP then averages gradients across ranks.
   For `W` ranks, `N` valid targets and local summed losses `S_r`, the resulting
   gradient is `(1/W) * sum_r grad(W * S_r / N) = grad(sum_r S_r / N)`.
   This is the global mean token-loss gradient even when ranks have different
   amounts of padding. Step 2 in the diagram places this scaling before backward;
   its factor of 3 is the illustrated three-GPU world size.
5. All ranks clip the synchronized gradient and apply the same AdamW update,
   keeping their model replicas aligned. Learning rate and cumulative progress
   use actual valid-token counts. The cycle repeats with the next global batch.

The production implementation is
[`backward_batch()` and `train()` in train.py](train.py).
Input shards remain on the shared `/data` volume and are read only by rank zero
within a job. Independent jobs can read the same immutable snapshot; each keeps
its own cursor and output directory. The launcher copies source, tokenizer assets
and the manifest for provenance, not the corpus. Training checks file sizes and
manifest identity at startup, then seeks directly into binary token shards using
the saved pass/shard/document/token cursor. Full-file checksum verification is an
explicit preparation/transfer operation; see [snapshot conversion and verification](../data/SIMPLE_LLM_STREAMING.md#seekable-snapshots-and-verification).

Rank zero also owns logging and diagnostic validation. At checkpoint boundaries,
all ranks contribute their RNG state; rank zero saves the shared model/optimizer,
data cursor and counters alongside those states, atomically renames the file,
and commits the Modal Volume. An explicitly launched fresh job restores them
with the same world size and compatible configuration, data and source identity.
There is no automatic job chain; hard termination can lose updates after the last
committed checkpoint.

## Start a job

Run from `training/`, with Modal credentials and prepared data on
`lm-exploration-mixture-data`. The configuration is a JSON `DistributedConfig`,
not the character trainer's `Recipe` format.

```sh
# First convert the existing gzip pilot using the linked conversion instructions.
# Bounded smoke run against that converted snapshot; creates a checkpoint.
modal run --detach -m distributed_training.modal_train \
  --config distributed_training/recipes/h100x2-pilot.json \
  --run-id my-ddp-pilot-01 \
  --snapshot simple-llm-124m-20261016/snapshots/pilot-001-seekable \
  --gpu H100 --max-steps 2 --max-seconds 300

# Substantial-run settings: requires a training-ready snapshot with capacity.
modal run --detach -m distributed_training.modal_train \
  --config distributed_training/recipes/h100x3-20b.json \
  --run-id my-pretraining-segment-01 \
  --snapshot APPROVED_VERSION/snapshots/APPROVED_SNAPSHOT \
  --gpu H100 --max-seconds 3000
```

The second example is a template, not a launched full run. Snapshot audit and
capacity gates apply before model allocation. Set `allow_unreviewed_data: true`
explicitly to permit snapshots with pending audits; the default is `false`.
This permission is independent of architecture, token budget and `pilot` mode.
Pilot mode alone does not permit unreviewed data, and neither flag resets
exhausted sources or removes the loader's two-pass source limit.

`DistributedConfig` validates general settings, not an architecture allowlist or
experiment token ceiling. Architecture dimensions and token budgets are recipe
choices; pilot mode has no separate token ceiling. Model/runtime compatibility
and the loader's two-pass source limit still apply. Non-pilot runs also check
mixture capacity before model allocation. This does not authorize a run.

Recipes now record data permission explicitly. Historical configs and measured
artifacts are unchanged; exact resume from older checkpoints requires their
original source and settings because config and source identity are checked.

Supported GPU families are `H100`, `L40S` and `A10G`; `world_size` in the JSON
chooses the number of GPUs. Global windows must be divisible by
`world_size * microbatch`. The following settings preserve the benchmark's
48-window global batch at context 2,048:

| GPU family | GPU count | Microbatch/GPU | Accumulation |
| --- | ---: | ---: | ---: |
| H100 | 1 | 24 | 2 |
| H100 | 2 | 24 | 1 |
| H100 | 3 | 16 | 1 |
| L40S | 1 / 2 / 3 | 4 | 12 / 6 / 4 |
| A10G | 1 / 2 / 3 | 8 | 6 / 3 / 2 |

Rank zero constructs one deterministic token-weighted global batch and scatters
it. Gradients accumulate with `no_sync()` before the final microbatch. DDP's rank
averaging is compensated by `world_size / global_nonpadding_tokens`, so padded
windows and unequal valid-token counts across ranks do not bias the objective.
The trainer uses BF16 autocast, FP32 parameters, causal SDPA, fused AdamW
(betas 0.9/0.95), gradient clipping at 1 and a token-based warmup/cosine schedule.
The supplied schedule is a configurable starting point, not a quality-qualified
20B recipe. Learning rate is calculated at the end-token count of each update.

## Resume in a fresh Modal job

Artifacts live under `lm-exploration-training/<run-id>/`. `latest.json` names the
latest checkpoint in that job. Checkpoints use `step-NNNNNNNNN.pt` and are written
to a temporary file, renamed, then committed to the Modal Volume before training
continues. Every graceful stop saves an optimizer-boundary checkpoint.

```sh
modal run --detach -m distributed_training.modal_train \
  --config distributed_training/recipes/h100x2-pilot.json \
  --run-id my-ddp-pilot-02 \
  --snapshot simple-llm-124m-20261016/snapshots/pilot-001-seekable \
  --resume my-ddp-pilot-01/step-000000002.pt \
  --gpu H100 --max-steps 2 --max-seconds 300
```

`--max-steps` counts **additional steps in this job**; `target_tokens` is the
cumulative ceiling across the entire checkpoint lineage. A batch that would
exceed the ceiling is rolled back without training, so completion can fall short
by less than one global batch. Exhaustion similarly rolls back a partial batch.

Checkpoints contain model/optimizer state, cumulative steps and loss tokens, the
shared mixture cursor and per-source exposures, and Python/NumPy/CPU-Torch/CUDA
RNG state for **every rank**. LR progress is derived from the saved token count.
Resume rejects changes to training settings (including world size and batch),
data/tokenizer identity, training source hashes or PyTorch version. Historical
pilot and character checkpoints use different formats and cannot resume here.
Only load trusted checkpoints from this project's run volume.

A resumed job uses a new directory and records its parent checkpoint; it does not
overwrite the previous job. The training source, frozen tokenizer, manifest and
launcher provenance are copied to the output directory. Reports and metrics are
JSON; model checkpoints and corpus text are not committed to Git.

## Stop and operational limits

The generic worker has a 24-hour hard limit, no automatic retry, and one container
per app. A job's cooperative time limit defaults to 3,000 seconds and is at most
82,800 seconds, leaving a reserve for checkpointing. `--max-steps`, time limits, token limits, source exhaustion and
SIGINT/SIGTERM stop at optimizer boundaries. Signal handling is best effort:
hard termination or a failed rank can lose work after the last committed
checkpoint. Torchrun stops peer ranks on failure; rank-zero loader/save errors
are broadcast so peers do not wait indefinitely in the next collective.

There is **no automatic chain of jobs**. Resume explicitly from the last committed
checkpoint into a fresh run ID. The worker commits artifacts in `finally`, and
preserves torchrun logs if training fails. Checkpoints are retained; budget storage
and remove old run artifacts deliberately. The example 20B configuration saves
every 1,000 steps. Each full-model optimizer checkpoint is roughly 1.5 GB.

Validation reports one fixed window per source and its mixture-weighted loss at
startup, configured intervals and the end. This is a diagnostic, not production
capability evaluation or automatic best-checkpoint selection. Full-scale corpus
preparation/audits remain separate work. Seekable token shards remove compressed
cursor replay; full-corpus restart latency, throughput and multi-day reliability
have not been measured. Old gzip snapshots need one-time conversion for new runs;
old checkpoints require their original code and snapshot. Moving the trainer
into this package changes source hashes, so checkpoints from before the move
also require their original code for resume.

## Raw-corpus experiments

The [raw-corpus preparation guide](../data/RAW_CORPUS.md) covers reusable SQLite snapshots,
preparation throughput evidence and the dedicated modern 9.85M / 5B workflow.
It uses this package's worker and explicit runtime contract, not an alternate
trainer. The generic trainer permits sessions up to 82,800 seconds; the worker's
86,400-second hard limit leaves time for checkpointing. Individual experiment
launchers may impose shorter bounds.

## Verification

```sh
uv run python -m unittest discover -s tests -p 'test_distributed_*.py' -v
modal run --detach -m distributed_training.modal_checks --run-id UNIQUE_CHECK_RUN_ID
```

The local checks compare the production one-rank accumulation loss, gradients
and AdamW update with a plain full-batch model, including fully masked
microbatches. They also assert the one- and two-GPU pilot recipes differ only in
GPU count. These mathematical checks use Gloo/FP32 locally, not CUDA/BF16
performance qualification.

The bounded CUDA integration suite exercises 1/2/3 GPUs, the production gradient
accumulation path against a single-model reference with unequal padding (including
fully masked microbatches), interrupted-versus-uninterrupted model/optimizer/data/
RNG restoration, exact token-budget accounting, two-pass exhaustion and invalid
resume/audit rejection. Floating model/optimizer tensors use FP32 tolerances
(rtol 1e-5, atol 1e-7); data and RNG state must match exactly. Three-rank NCCL
reduction order can change low bits across fresh jobs.

Measured run evidence is recorded in
[`results/simple-llm-ddp-support-20261017.md`](../results/simple-llm-ddp-support-20261017.md).
Those GPU measurements predate the seekable snapshot format. The format change
has local loader/conversion/resume checks; a new CUDA qualification has not been run.
