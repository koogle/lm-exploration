# KDA foundation pretraining constraints

The selected model and stages are in [the training plan](CODING_REASONER_1B.md);
[the mixture study](KDA_DATA_STUDY.md) explains the 50/40/10 decision and its
accepted retention cost. This document defines the scale-up requirements.

## Corpus

Plan a conditional single-pass 100B-token schedule: 50B ClimbMix, 40B permissive
GitHub and 10B Nemotron CC-Math score-3 tokens. Freeze source revisions,
tokenizer identity, filtering and train/held-out partitions in the preparation
manifest. Require measured usable capacity of at least target tokens times
source weight; estimates size preparation but cannot authorize training. Do not
fill a short source with silent repetition or substitute a new source.

Require content-hash train/validation separation across sources and repository
separation for GitHub. Screen conservative 13-token overlaps against the frozen
benchmark manifests, including the six LiveCodeBench releases used in the study.
Inspect hits and rejection rates: this screen does not establish complete
contamination freedom. Preserve source-license restrictions, language coverage,
source exposure and repetition audits. Bound math preparation by Parquet row
groups and persist progress through the shared workflow infrastructure.

Global near-deduplication, fork-aware separation and generated/vendor-code quality
remain explicit gaps. Archive prepared-corpus audits before describing the entire
100B-token corpus as qualified.

## Training and budget

Use exactly eight 80GB H100s, the shared trainer, frozen model/tokenizer and MTP
objective. Select execution details from matched real-data correctness and
performance checks. Full pretraining remains held during performance
qualification; hardware and MFU experiments belong to the stacked implementation
PR and do not alter this mixture decision.

Start with a bounded qualification segment and verify finite losses, per-source
held-out evaluation, checkpoint/resume and source consumption before extending
that checkpoint lineage. Preserve periodic checkpoints and numerical evaluation;
preview generation is optional and must not invalidate performance measurement.
No midtraining, SFT or RL stage starts automatically with foundation training.

Keep the main-pretraining allocation at $5,000 within the overall plan. Set the
final token commitment from actual dollars per token, including preparation,
CPU, memory, storage, evaluation and checkpoint overhead. Reconcile previous
spend before each continuation and preserve the later-stage reserve. A 100B-token
target is conditional on capacity, observed learning and remaining budget.

Every launch must have a server-side timeout, zero retries and an independent
remote deadline covering queued work and children, no later than 24 hours after
launch. Preserve shorter diagnostic limits. Publish measurements, negative
results and checkpoints in the existing HF/Modal artifact stores, with source
provenance; executable launch/configuration details belong to the implementation
review rather than this planning-only change.
