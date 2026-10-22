# Resumable subword training on multiple Modal GPUs

`distributed/modal_train.py` supplies the worker that runs selected models on **one, two, three or eight
GPUs on one host**, using PyTorch NCCL DistributedDataParallel. It trains a shared
model with synchronized gradients through [`trainer/loop.py`](../trainer/loop.py).
The legacy character loop has been removed; throughput-only benchmarks remain separate.

## One implementation, including one GPU

Shared configuration, per-process distributed runtime, the generic Modal
launcher, shared image and GPU checks live in this package. The actual training
loop, evaluation and checkpointing live in [`../trainer/`](../trainer/README.md). Training settings live in [`../workflows/model_configs/`](../workflows/model_configs/README.md);
preparation and continuation live in [`../workflows/`](../workflows/README.md).
The duplicate `modal_simple-llm_pilot.py` / `trainer/subword_pilot.py`
path has been removed; historical pilot results are in the HF dataset `jfrick/lm-exploration-results` and its
implementation remains in Git history. Historical character recipes are retained
for provenance; their loop and launcher have been removed.

For one GPU, set `world_size` to 1: the very same `train()` and `backward_batch()`
run under one-rank DDP, with the same model, loader, global batch, valid-token loss
normalization, optimizer, LR schedule and checkpoint format. There is no special
single-GPU loop. With matched settings, multiple GPUs divide that same global
batch, subject to floating-point reduction differences. This is not a claim of
bitwise equality across GPU counts or reproduction of the removed pilot's
step-based warmup/constant learning-rate schedule.

```sh
# One-GPU bounded pilot, using the same model config as the two-GPU workflow.
modal run --detach -m workflows.modal_train \
  --workflow workflows/h100x1-pilot.json --run-id my-single-gpu-pilot-01
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
[`backward_batch()` and `train()` in trainer/loop.py](../trainer/loop.py).
Input shards remain on the shared `/data` volume and are read only by rank zero
within a job. Independent jobs can read the same immutable snapshot; each keeps
its own cursor and output directory. The launcher copies source, tokenizer assets
and the manifest for provenance, not the corpus. Training checks file sizes and
manifest identity at startup, then seeks directly into binary token shards using
the saved pass/shard/document/token cursor. Full-file checksum verification is an
explicit preparation/transfer operation; see [snapshot verification](../data/SIMPLE_LLM_STREAMING.md#seekable-snapshots-and-verification).

Rank zero also owns logging and diagnostic validation. At checkpoint boundaries,
all ranks contribute their RNG state; rank zero saves the shared model/optimizer,
data cursor and counters alongside those states, atomically renames the file,
and commits the Modal Volume. An explicitly launched fresh job restores them
with the same world size and compatible configuration, data and source identity.
There is no automatic job chain; hard termination can lose updates after the last
committed checkpoint.

## Optimizer default

New `DistributedConfig` settings default to `optimizer="muon"`: Muon updates
hidden matrices, while fused AdamW updates the tied embedding/output table,
biases and norms. Set `optimizer="adamw"` to use AdamW for every parameter.
`muon_lr=0.02` and `muon_momentum=0.95` configure Muon; `peak_lr=0.0003`
configures the AdamW companion. Both share the token-based warmup/cosine factor.

For new eight-H100 runs, use `workflows/model_configs/historical/h100x8-muon-8k-5b.json`.
It combines Muon with microbatch 2, cropping, cached RoPE and model compilation.
The 5B ceiling is a configuration, not an automatically launched run or a measured
5B result. The [100M-token comparison](MUON_COMPARISON.md#measured-results-run-prefix-muon-124m-100mt-20261019-b)
supports this default; longer-run quality remains unmeasured. See the
[comparison guide](MUON_COMPARISON.md) for the bounded experiment.

Historical recipes explicitly select AdamW to preserve their experiment meaning.
Resume requires the saved optimizer/config/source; changing the default does not
convert AdamW checkpoint state into Muon.

## Start a job

Run from `training/`, with Modal credentials and prepared data on
`lm-exploration-mixture-data`. Select a workflow JSON; it resolves the model and dataset configurations into
one validated trainer request. See the [workflow guide](../workflows/README.md).

```sh
# Bounded smoke run against the seekable pilot snapshot; creates a checkpoint.
modal run --detach -m workflows.modal_train \
  --workflow workflows/h100x2-pilot.json --run-id my-ddp-pilot-01

# Substantial-run settings: requires a training-ready snapshot with capacity.
modal run --detach -m workflows.modal_train \
  --workflow workflows/h100x3-20b.json --run-id my-pretraining-segment-01
```

The second example is a template, not a launched full run. Snapshot audit and
capacity gates apply before model allocation. Set `allow_unreviewed_data: true`
explicitly to permit snapshots with pending audits; the default is `false`.
This permission is independent of architecture, token budget and `pilot` mode.
Pilot mode alone does not permit unreviewed data, and neither flag resets
exhausted sources or removes the loader's two-pass source limit.

`DistributedConfig` validates general settings, not an architecture allowlist or
experiment token ceiling. Architecture dimensions come from the model config and token budgets from the workflow; pilot mode has no separate token ceiling. Model/runtime compatibility
and the loader's two-pass source limit still apply. Non-pilot runs also check
mixture capacity before model allocation. This does not authorize a run.

Recipes now record data permission explicitly. Historical configs and measured
artifacts are unchanged; exact resume from older checkpoints requires their
original source and settings because config and source identity are checked.

### Explicit experiment settings

The selected dataset config supplies `data_loader`: `"token_shards"` or `"raw"`.
The selected loader validates the manifest; an incompatible snapshot fails rather
than selecting another implementation. Both training and validation use that
selection. Raw dataset configs select `loader: "raw"` explicitly.

These settings are validated by `DistributedConfig`, recorded in resolved run
metadata/checkpoints, and included in strict resume compatibility checks:

| Setting | Default | Meaning |
| --- | --- | --- |
| `adam_beta1`, `adam_beta2` | `0.9`, `0.95` | AdamW moving-average coefficients, each in `[0, 1)` |
| `gradient_clip_norm` | `1.0` | Positive maximum gradient norm |
| `validation_windows_per_source` | `1` | Fixed initial windows from each validation source; insufficient data fails |
| `log_every` | `10` | Console step interval; the first step also prints and every step is saved to metrics |

Validation averages by valid target count within each source, then applies source
mixture weights. Vocabulary size comes from the frozen tokenizer definition and
capacity from the validated snapshot's repetition policy. Tokenizer identity,
supported snapshot formats and repetition limits remain data compatibility checks.

The [configuration choices audit](choices.md) records the findings behind these
changes. Earlier measured results retain their original settings and evidence.

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
The trainer uses BF16 autocast, FP32 parameters, causal SDPA, the selected optimizer,
gradient clipping at 1 and a token-based warmup/cosine schedule. The AdamW component
uses betas 0.9/0.95.
The supplied schedule is a configurable starting point, not a quality-qualified
20B experiment. Learning rate is calculated at the end-token count of each update.

## Resume in a fresh Modal job

Artifacts live under `lm-exploration-training/<run-id>/`. `latest.json` names the
latest checkpoint in that job. Checkpoints use `step-NNNNNNNNN.pt` and are written
to a temporary file, renamed, then committed to the Modal Volume before training
continues. Every graceful stop saves an optimizer-boundary checkpoint.

```sh
modal run --detach -m workflows.modal_train \
  --workflow workflows/h100x2-pilot.json --run-id my-ddp-pilot-02 \
  --resume my-ddp-pilot-01/step-000000002.pt
```

Qualification step limits count **additional steps in each job**; `target_tokens` is the
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
per app. The workflow selects each job's cooperative time limit, which is at most
82,800 seconds, leaving a reserve for checkpointing. `--max-steps`, time limits, token limits, source exhaustion and
SIGINT/SIGTERM stop at optimizer boundaries. Signal handling is best effort:
hard termination or a failed rank can lose work after the last committed
checkpoint. Torchrun stops peer ranks on failure; rank-zero loader/save errors
are broadcast so peers do not wait indefinitely in the next collective.

The workflow explicitly chains bounded sessions using fresh segment IDs and
committed checkpoints, up to its configured segment limit. The low-level worker
has no automatic retry. Manual checkpoint resume uses a fresh workflow run ID. The worker commits artifacts in `finally`, and
preserves torchrun logs if training fails. Checkpoints are retained; budget storage
and remove old run artifacts deliberately. The example 20B configuration saves
every 1,000 steps. Each full-model optimizer checkpoint is roughly 1.5 GB.

Validation reports the configured fixed windows per source and mixture-weighted loss at
startup, configured intervals and the end. This is a diagnostic, not production
capability evaluation or automatic best-checkpoint selection. Full-scale corpus
preparation/audits remain separate work. Seekable token shards remove compressed
cursor replay; full-corpus restart latency, throughput and multi-day reliability
have not been measured. Old checkpoints require their original code and snapshot. The active loop now lives in `trainer/loop.py`; resume continues to enforce
source hashes. Historical checkpoints require compatible original code and settings.

## Overlapped data preparation

[`modal_streaming_train.py`](../modal_streaming_train.py) starts training from an
initial upstream prefix instead of preparing the full corpus first: a rank-zero
producer thread fetches pinned source shards, tokenizes them into global batches
and keeps up to `prefetch_batches` (1–64; 0 keeps the synchronous path) ready
while the GPUs train. Checkpoints save only the acknowledged optimizer cursor, so
a restart discards and rebuilds speculative batches; live manifests require
`allow_unreviewed_data`. CUDA equivalence and throughput have not been measured.

## Eight-GPU experiment

The dedicated [eight-H100 / 8K / 5B workflow](SIMPLE_LLM_8K.md) uses the frozen raw
snapshot and publishes inference checkpoints to HF. Its recipe and resource bounds
are explicit; eight-GPU performance requires separate qualification.

## Raw-corpus experiments

The [raw-corpus preparation guide](../data/RAW_CORPUS.md) covers reusable SQLite snapshots,
preparation throughput evidence and the dedicated modern 9.85M / 5B workflow.
It uses this package's worker and explicit runtime contract, not an alternate
trainer. The generic trainer permits sessions up to 82,800 seconds; the worker's
86,400-second hard limit leaves time for checkpointing. Individual experiment
launchers may impose shorter bounds.

The [generic controller](../workflows/modal_train.py) accepts an explicit workflow
path, model config and dataset config. The [workflow guide](../workflows/README.md)
documents raw preparation, prepared-snapshot selection, qualification and continuation.
`workflows/small-5b.json` retains the small-model experiment's execution settings.
The target token count and GPU allocation now belong only to the workflow;
the dataset owns loader selection and the model config owns architecture/optimizer settings.

## Verification

```sh
uv run python -m unittest discover -s tests -p 'test_distributed_*.py' -v
modal run --detach -m distributed.modal_checks --run-id UNIQUE_CHECK_RUN_ID
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
[`simple-llm-ddp-support-20261017.md`](https://huggingface.co/datasets/jfrick/lm-exploration-results/blob/main/training/results/simple-llm-ddp-support-20261017.md).
Those GPU measurements predate the seekable snapshot format. The format change
has local loader/resume checks; a new CUDA qualification has not been run.
