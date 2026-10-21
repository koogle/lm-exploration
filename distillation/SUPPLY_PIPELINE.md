# Eight-GPU student with a throughput-sized teacher pool

The 215,910,656-parameter shared-Qwen student retains its 4.5B fresh-token target.
The user requested eight H100s for student training and enough independent
teacher instances that generation does not determine training speed.

## Student

`modal_billion.train` requests one `H100:8` host and launches eight NCCL DDP ranks
with torchrun. All ranks resume the same weights, AdamW state and consumed-token
cursor. Rank zero writes checkpoints and runs validation; per-rank RNG states
are saved. The checkpoint format accepts the existing single-GPU checkpoint and
records the GPU-count migration. This is an implementation and deployment
configuration; GPU allocation and measured throughput must be confirmed remotely.

Each GPU processes four microbatches of four windows, giving 128 global windows
instead of the previous 16. Thus an update consumes about eight times as many
tokens; this changes the optimization batch, not only wall-clock speed. The
learning-rate schedule remains tied to the exact consumed-token count and is not
multiplied by eight. Empty final-rank slices use ignored-label padding so all ranks
execute matching collectives. DDP's gradient averaging is compensated by
`world_size / global_valid_targets`, preserving the global mean-token objective.

Each rank uses the same frozen shard-list snapshot and partitions disjoint
windows from the same global batch. Checkpoint cursor advancement and the 4.5B
boundary include real targets only, never padding. CPU validation tests cover
unequal targets and empty ranks; actual multi-GPU qualification is recorded
separately after allocation.

## Teacher capacity and buffering

Each teacher replica uses one H100 running Qwen3-8B with vLLM, producing durable
accepted problem/code/explanation shards. A teacher does not need eight GPUs.
The pool grows according to:

```
teachers = ceil(1.25 * measured_student_tokens_per_second
                / median_accepted_tokens_per_second_per_teacher)
```

Only actual eight-GPU training metrics after at least 5M segment tokens drive
scaling. Student rate includes in-segment validation/checkpoint overhead; teacher
rates count accepted Qwen targets, not rejected or raw output tokens. Pool size
can grow without changing previously published data order. For example, a
100K-token/sec student and 2.5K-token/sec teacher call for 50 replicas. This is an
example, not a measured result or a launch count.

Teachers work in resumable 10M-token segments. The controller stops admitting new
segments when the ready-data buffer exceeds 100M tokens, and resumes them when
training consumes it. In-flight segments can overshoot the buffer; they are not
wasted or regenerated. At the global 4.5B ready-target budget, remaining teacher
calls are cancelled and their last committed shards retained. The student still
stops at exactly 4.5B targets; there may be a small unused committed tail.

## Ordered dataset and reuse

`dataset-plan.json` freezes the already-available original corpus prefix; its
exact Qwen-token count is measured once. The student need not wait for the older
500M generation job to finish. New continuation shards are admitted into an
append-only `manifest.json` in commit-observation order. Faster teachers can
supply data immediately, without waiting for a slow teacher at a round-robin
barrier. Previously admitted shard identities are immutable and checksum checked.
There is no dataset replay. The checkpoint pins the dataset-plan hash.

Accepted examples persist on `lm-exploration-distillation` with text, shared-Qwen
IDs, original IDs, content hashes and prompt indices. A separate CPU archive job
combines those shards into reusable gzip JSONL parts and commits them to a private
Hugging Face dataset named `<HF owner>/python-math-qwen3-distillation-20261021`.
It publishes the exact token IDs used by training, rather than re-tokenizing
text. `dataset-archive.json` on the run volume records the repository, immutable
commit, index path/checksum, and archived counts. That pinned index provides each
part's SHA256, source-shard mapping and the student's consumed-token prefix.

Load the JSONL data with Hugging Face Datasets, or read the gzip records directly.
Always pin the recorded commit for reproducibility. The accepted corpus remains
syntax-checked rather than independently execution-verified. Per-producer exact
deduplication does not establish semantic or global uniqueness.

## Operations

Run commands from the repository root. The deployed app is
https://modal.com/apps/radiantai/main/deployed/lm-distillation-215m-4p5b.
The archiver uses the named `distillation-huggingface` secret; credentials are not
included in images, source snapshots or datasets.

For a migration, stop the previous controller and its student call, deploy, run
`configure_supply.remote()` once to freeze the dataset prefix and retire legacy
producer calls, then `prepare_unique_policy.remote()` to refresh the durable
student position, and finally spawn the deployed `supervise` function. Do not
run migration helpers against a live controller. Subsequent controller resumptions
reattach saved student, teacher and archive call IDs.

All 24 local distillation tests pass, including distributed target normalization,
dynamic publisher prefix preservation, corruption rejection and archive token-ID,
checksum and immutable-commit preservation. Ruff checks pass. Live GPU results
must not be inferred from these CPU tests.

## Observed deployment status, 2026-10-21

The initial prefix was frozen at 103,669,090 Qwen targets. The deployed controller
reattached four continuation teachers and requested the 8-H100 student; four
original-corpus teachers remain separate. At the last check the student request
was still queued, so no eight-GPU throughput or GPU correctness result is claimed.
Teacher expansion remains conditional on the first measured eight-GPU segment.

The archive committed 11,396,274 targets / 31,006 examples to the private dataset
`jfrick/python-math-qwen3-distillation-20261021`. The downloaded index's SHA256 was
verified against the recorded commit. See [the pinned artifact index](dataset-artifact-index.json)
for the immutable revision/checksum; ongoing archive commits are tracked on the
run volume. The HF/Xet cache is placed outside the training volume so open log
files do not prevent the archiver from reloading newly published shards.
