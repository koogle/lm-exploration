# Resumable subword training on multiple Modal GPUs

`modal_distributed_train.py` runs the modern 124.4M model on **one, two or three
GPUs on one host**, using PyTorch NCCL DistributedDataParallel. It trains a shared
model with synchronized gradients. It is separate from the historical character
trainer and from the throughput-only benchmarks.

## Start a job

Run from `training/`, with Modal credentials and prepared data on
`lm-exploration-mixture-data`. The configuration is a JSON `DistributedConfig`,
not the character trainer's `Recipe` format.

```sh
# Bounded smoke run against the existing pilot snapshot; creates a checkpoint.
modal run --detach modal_distributed_train.py \
  --config recipes/distributed/h100x2-pilot.json \
  --run-id my-ddp-pilot-01 \
  --snapshot scholarly-124m-20261016/snapshots/pilot-001 \
  --gpu H100 --max-steps 2 --max-seconds 300

# Substantial-run settings: requires a training-ready snapshot with capacity.
modal run --detach modal_distributed_train.py \
  --config recipes/distributed/h100x3-20b.json \
  --run-id my-pretraining-segment-01 \
  --snapshot APPROVED_VERSION/snapshots/APPROVED_SNAPSHOT \
  --gpu H100 --max-seconds 3000
```

The second example is a template, not a launched full run. Snapshot audit and
capacity gates apply before model allocation. The `pilot: true` override permits
pending audits only with a target no larger than 5M tokens; it never resets or
replays exhausted sources. The two-pass source limit remains in force.

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
modal run --detach modal_distributed_train.py \
  --config recipes/distributed/h100x2-pilot.json \
  --run-id my-ddp-pilot-02 \
  --snapshot scholarly-124m-20261016/snapshots/pilot-001 \
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

Each worker has a one-hour hard limit, no automatic retry, and one container per
app. A job's cooperative time limit is at most 3,000 seconds, leaving a reserve
for checkpointing. `--max-steps`, time limits, token limits, source exhaustion and
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
preparation/audits remain separate work. The existing gzip loader replays to saved
cursor offsets during resume; full-corpus restart latency and multi-day reliability
have not been measured.

## Verification

```sh
python -m unittest discover -s tests -p test_distributed_config.py -v
modal run --detach modal_distributed_checks.py --run-id UNIQUE_CHECK_RUN_ID
```

The bounded CUDA integration suite exercises 1/2/3 GPUs, the production gradient
accumulation path against a single-model reference with unequal padding (including
fully masked microbatches), interrupted-versus-uninterrupted model/optimizer/data/
RNG restoration, exact token-budget accounting, two-pass exhaustion and invalid
resume/audit rejection. Floating model/optimizer tensors use FP32 tolerances
(rtol 1e-5, atol 1e-7); data and RNG state must match exactly. Three-rank NCCL
reduction order can change low bits across fresh jobs.

Measured run evidence is recorded in
[`results/scholarly-ddp-support-20261017.md`](../results/scholarly-ddp-support-20261017.md).

## Prepare once, reuse across runs

The `modal_small_5b.py` launcher checks `<version>/snapshots/*/manifest.json`
before opening staging or fetching upstream data. Use the same data `version`
and a fresh training `run_id` to reuse a prepared corpus. A matching snapshot
must have the same pinned source recipe, tokenizer, split/filter implementation,
source weights and two-pass policy, enough tokens for the new target, and
validation data for each source. Its database checksum is verified before reuse.

When a match is found, the workflow records `data_reused` and starts training
against that exact immutable snapshot. It does not redownload, refilter, recount
or copy the corpus. Pending audit flags are preserved. A larger target or changed
data definition requires additional preparation (changed staging identity requires
a new version). Corrupt compatible snapshots fail explicitly rather than silently
being reused. Raw documents are still tokenized as batches need them.

This reuse change is in PR #13 for future deployments. The active 5B workflow
continues on its original deployed source to preserve checkpoint compatibility;
its eventual snapshot is compatible with this reuse lookup.
