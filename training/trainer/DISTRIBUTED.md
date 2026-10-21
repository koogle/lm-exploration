# Resumable subword training on multiple Modal GPUs

`modal_distributed_train.py` runs the modern 124.4M model on **one, two, three or eight
GPUs on one host**, using PyTorch NCCL DistributedDataParallel. It trains a shared
model with synchronized gradients. It is separate from the historical character
trainer and from the throughput-only benchmarks.

For the authorized eight-H100 / 8K / 5B workflow and periodic Hub publication, see
[Simple LLM 8K](SIMPLE_LLM_8K.md). Eight-GPU performance requires its own qualification.

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
capacity gates apply before model allocation. Set `allow_unreviewed_data: true`
explicitly to permit data with pending audits; it defaults to `false` and is
independent of architecture, token budget and `pilot`. Pilot mode alone does not
permit unreviewed data. Token budgets are recipe choices; pilot mode has no
separate token ceiling. Neither flag resets exhausted sources or removes the
loader's two-pass source limit.

`DistributedConfig` validates general settings (types, positive dimensions,
even integral RoPE head dimensions, batches and optimizer/schedule consistency),
not an experiment architecture allowlist or token ceiling. Dedicated workflow
policies live in `run_validation.py`: every entrypoint of `modal_small_5b.py`
requires the modern 9.85M architecture,
three GPUs and a non-pilot target no larger than 5B tokens. Those workflow checks
run before corpus preparation, continuation scheduling or training invocation.
The overnight continuation additionally validates its 124M architecture, eight GPUs
and non-pilot target no larger than 10B at checkpoint initialization and segment
execution. These experiment policies do not constrain shared config construction.
The data opt-in remains independent of these policies. Preparation and loaders
share the same review gate: reused reviewed snapshots need no override, while new
or pending-audit
snapshots require explicit permission before upstream ingestion or GPU scheduling.

This replaces the former experiment-scoped `experimental_data` setting. Update
active recipes to the new key. Historical result/config artifacts are unchanged;
exact resume from those checkpoints requires their original source and settings,
not this refactored trainer.

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
python -m unittest discover -s tests -p test_run_validation.py -v
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

The reuse lookup and persistent-stream preparation fix are deployed in PR #13.
The active 5B preparation was resumed from committed cursors with the updated
launcher before GPU training began. Its eventual snapshot is reusable by later
runs. No model-training or raw-corpus identity code changed in this update.


## Preparation throughput

`PersistentStream` keeps the current upstream connection and decompressor open
between 1,000-record durable chunks. Previously each chunk reopened gzip and
reread every earlier row to reach the cursor, multiplying download/decompression
work inside each shard. Now that replay happens only after a restart, rollback,
source/shard change or stream failure. Database transactions and volume checkpoints
retain their existing behavior. The staging identity and accepted records are
unchanged; the helper is recorded in run source provenance.

Tests compare the original reader and the persistent reader for identical stored
documents, counts and cursors, and verify reopening at a saved cursor after rollback.
Future improvements include bounded parallel fetching and batched tokenization;
these require coordinating a single database writer and deduplication decisions.
No measured end-to-end speedup is claimed until live progress supports it.

### Batched, parallel preparation

The preparation worker now requests 16 CPUs / 16 GiB RAM. Up to three source
workers fetch and tokenize concurrently; the tokenizer uses eight Rust threads.
Each worker emits at most 64 documents or roughly 4 MiB of text per batch. A
single SQLite writer commits the documents, filtering decisions, repetition
counts and source cursor atomically. It commits source batches in deterministic
round-robin order, so network timing cannot decide duplicate ownership. Existing
stored documents retain their source assignment. New-source ownership can differ
from the former whole-source-at-a-time schedule when content overlaps.

Trigger-maintained token totals replace repeated full-corpus scans. Existing
staging is migrated with one initial count scan; its identity and document format
are preserved. Trigger changes roll back with a failed write transaction.

Compressed JSON shards are cached locally, retaining one shard per active source.
Download interruptions retry a batch up to four attempts from its committed
cursor. Partial prefetched batches are never recorded as trained/prepared data.
Volume snapshots and workflow status update after approximately 20 seconds of
processing; initial downloads can take longer. Discarded prefetch may be fetched
again on restart; committed accepted records are reused.

Controlled Modal CPU benchmarks measured 4.3–4.6x faster processing with identical
stored outputs on two synthetic trials. Those figures exclude upstream downloads,
volume commits and the one-time count migration. See the retained benchmark JSON
for exact timing and matched-output fingerprints. They do not establish the
end-to-end preparation speedup on the live corpus.
