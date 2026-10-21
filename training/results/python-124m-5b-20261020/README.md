# Python 124M / 5B execution record

**Current status (2026-10-21): resumed CPU preparation and expanded documentation acquisition; no GPU training or model publication.**

The previous ephemeral app was cancelled at 2026-10-20 22:54 UTC. Its final
saved counts reached 555,882,002 implementation, 292,688,273 tests/examples,
and 29,876,984 documentation training targets; the source census was incomplete.
The cancellation is visible in Modal logs; its initiator is not established.

Preparation now runs through deployed functions invoked with `spawn`, independent
of the local CLI. Existing source cursors are retained. The user's collected
private code datasets were rechecked and have not changed since the first pass;
their complete pinned repository files remain preferred over older duplicates.

- [Resumed corpus census](https://modal.com/apps/radiantai/main/deployed/simple-llm-python-124m-5b)
- [Parallel Python documentation acquisition](https://modal.com/apps/radiantai/main/deployed/simple-llm-python-documentation)
- [Dependent merge and snapshot finalization](https://modal.com/apps/radiantai/main/deployed/simple-llm-python-finalize)

The added source is `common-pile/stackexchange`, pinned at
`5ec3aa2e65b7c61b7c18f1ff18ccbc4d3279a23c`. Its Stack Overflow subset has
95 compressed shards (27,392,996,805 bytes). Only complete explanatory Python
threads with Python examples, sufficient prose and per-document CC-BY-SA
license/attribution metadata are accepted. Explicit non-Python code signatures,
missing provenance, repetitive content and error-dump-like documents are rejected.
This expands the documentation category to include explanatory technical Q&A;
it does not change the 10% target. Filters are heuristics, not a guarantee that
all examples are correct or that every embedded snippet is Python. Attribution,
source shard hashes and creation metadata remain with private records.

Eight shard workers measure capacity independently. The deployed coordinator
waits for both source jobs, merges with exact deduplication and held-out hash/
repository exclusions, and freezes only if all six categories meet their budgets
within two passes. Candidate shard totals are not globally deduplicated capacity.
No GPU launch is attached to these preparation functions.

The raw loader now supports exact per-category target budgets, masks only excess
final labels, and returns a padded partial final batch. Tests verify exact counts
and identical continuation after restore. One budget/resume test, the existing
raw-loader test, and nine Python corpus/documentation/snapshot tests pass locally.
The snapshot builder preserves tokenizer identity and freezes representative
validation windows from distinct held-out parents. New-corpus GPU qualification,
training, executable evaluation and publication remain pending.

The sections below retain the previous measurements as historical evidence.

[Draft PR #30](https://github.com/koogle/lm-exploration/pull/30) remains stacked on
#27. It incorporates the tested Muon implementation from #24, at
`1b9a7379e361d92f7e2ace0b15758a7e6eeed2f5`. The requested six-category mixture and
5,000,000,000 nonpadding-target budget remain unchanged. All code is Python;
initialization is fresh when training is eventually qualified.

## Executed work

GitHub, Modal and Hugging Face authentication succeeded. Both private code
collections were inspected: they contain complete source snapshots as well as
patch records. Preparation reads complete blobs from their 58 stored Git
repositories, pins commits, and prioritizes more recent snapshots. It does not
train on patch strings or execute repository code. Not every stored snapshot is
recent; commit dates remain in the private provenance.

The initial per-file Git reader was replaced with a batched blob reader, retaining
files marked `export-ignore`. Completed repositories were resumed rather than
recollected. Five repository license checks were unresolved and excluded.

The completed repository-only census contains **109,455,272 unique training
targets** across implementation, tests/examples and documentation. These counts
include one EOS target per document and exclude validation, filtered files, exact
duplicates and per-repository caps. They are **not** a 5B-capable corpus. Further
cross-source held-out exclusions can reduce them during supplemental preparation.
See [capacity table](repositories-capacity.md) and [machine-readable counts](repositories-capacity.json).

- [Initial repository preparation](https://modal.com/apps/radiantai/main/ap-vvHeCt6S23k42AiZKcL0wt): stopped to replace inefficient file-at-a-time reads.
- [Resumed, completed repository preparation](https://modal.com/apps/radiantai/main/ap-qCWLuFTlBWJHHHv7gP4dfY).
- [Detached supplemental preparation](https://modal.com/apps/radiantai/main/ap-SN8iRtna31TPxCgvyq6WVy): CPU-only, bounded continuation with persisted cursors.

The Stack, The Stack Dedup and StarCoderData authenticated file probes returned
`GatedRepoError`: the account is not authorized to retrieve their Python shards.
Existing pinned CodeParrot pools and supporting-text collections are being
inspected instead. A dataset's advertised size is not usable Python capacity.

## Checks and limits

Six corpus tests and eight configuration/resume-policy tests passed locally.
Ruff and whitespace checks passed. Prior eight-H100 compiled Muon qualification,
optimizer restore and full-model continuation results are retained in the
[original measured report](../muon-124m-100mt-20261019.md); those are prior-run
results, not GPU measurements from this experiment.

The new preparation preserves source text, comments, indentation and docstrings;
assigns each accepted file to one category; rejects generated/vendored/minified
and heavily repetitive files; and excludes files rejected by the Python 3.12 parser explicitly. Valid syntax introduced in later Python versions may therefore be excluded.
Path-based task classification and root-license detection are heuristics, not an
exhaustive per-file license or semantic-quality audit. Exact deduplication does
not remove all near clones. Larger files and some fixtures are conservatively
excluded. Supporting sources may be older. No benchmark-clean claim is made.

Private source text, repository-level notices and detailed provenance remain on
the private Modal data volume. Only aggregate capacity is committed here.

## Pending, not claimed

Corpus sufficiency, a frozen representative validation set, new-corpus GPU
qualification, tiny-set fitting/completion, training/generation logit agreement,
full 5B training, isolated executable evaluation, checkpoint publication, curves,
unselected generations and final Hub hash verification remain pending. No new
GPU optimizer step, training checkpoint, model repository or training cost is
claimed. A queued controller or preparation worker is not a training run.

## Latest measured capacity and remaining blocker

The [supplemental snapshot](partial-capacity.md) is **partial**, not a completed
corpus census. The detached CPU controller continues bounded preparation with
persisted cursors. It has no automatic GPU launch or publication stage. No
training data has been consumed by a model; actual training reuse is zero.

The available documentation pool contains **29,876,984 unique training targets**,
versus at least **250,000,000** needed under the two-pass limit. Independently,
all 59,316 records of `codeparrot/github-jupyter-parsed`, pinned at
`131d18ace1c171cfe3cb2b2a889678b614783580`, were token-counted with the frozen
tokenizer. Joining every record's cells with double newlines gives an optimistic
**186,211,988-target upper bound**, or **130,777,521** for the inspected permissive
license allowlist. These counts include all notebook categories and exclude no
validation, duplicates or low-quality records, so they are NOT usable training
capacity. Even adding the entire upper bound to current documentation yields
only 216,088,972 targets, below the 250M minimum. No notebook content was added
to the training corpus. See [raw aggregate measurement](notebook-upper-bound.json).

The official `HuggingFaceTB/smollm-corpus` Python-Edu subset at
`3ba9d605774198c5868892d7a8deda78031a781f` provides file IDs, repository names and
scores, not source text. Its documented content retrieval depends on the
Software Heritage S3 collection and refers to Stack v2 licensing. It was not
counted as available code. These observations do not prove that no suitable
corpus exists elsewhere; they establish that the bounded, inspected sources do
not yet satisfy this experiment. Additional verified documentation is required.

Runtime/cost: no new GPU runtime or GPU charge was incurred by this experiment.
A current 8-core/32-GiB CPU preparation worker plus its 1-core/1-GiB controller
costs approximately **$1.54/hour** at the retrieved public Modal CPU/memory rates,
excluding storage and other charges. Preparation through this report is estimated
at **$2–5**, not a measured invoice. The bounded CPU workflow may continue to
accrue charges after this snapshot. No 5B training runtime or measured cost exists.
