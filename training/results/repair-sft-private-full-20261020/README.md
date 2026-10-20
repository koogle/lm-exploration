# Full private patch SFT — preparation running

The expanded source snapshots contain 1,025 patch shards (504 collected-agent
repository shards and 521 curated-OSS shards). All shards are selected; eligible
training excerpts have no total, language or repository count cap. Other quality,
context, provenance and deduplication filters remain. This is repair SFT from
commit messages and before/after excerpts, not raw-code continued pretraining.

Implementation commit: `c2da7f1`. Source pins:
[`private-patch-full-sources.json`](../../sft/private-patch-full-sources.json).

Preparation started on Modal CPU (2 CPUs, 16 GiB):
https://modal.com/apps/radiantai/main/ap-IWJ3i7V2bXuvvXSR1E0wKd
The worker has emitted its start marker and shard-download progress. Final
example/language counts are not available yet. This detached worker will submit
training itself only after strict dataset validation and committing the data.
No H100 optimizer updates have been verified for this run yet.

- Dataset ID: `repair-private-full-20261020-v1`
- Planned training ID: `repair-sft-private-full-20261020-r01`
- Parent: original PR #26 `step-000066065.pt`, fresh SFT optimizer
- Budget: one H100, 40,000 updates, batch 8, context 2,048, LR 1e-5;
  four-hour training-loop limit; evaluation/checkpoints every 1,000 updates
- Sampling: uniform with replacement; at most 320,000 example draws
- Minimum launch gate: 79,441 training examples and 12 languages
- Volume: `lm-exploration-training`; dataset `report.json` stores final counts
  and dataset hash, `selection.json` records source coverage, and `launch.json`
  records the actual spawned training call and settings

The prior private run completed 20,000 updates with validation answer-token loss
0.834494 → 0.567412. That result is not a result for this expanded run. Earlier
validation code hashes remain excluded from training; commit assignment remains
stable. Validation sampling can change as the corpus grows, so aggregate losses
across datasets are not a controlled comparison. Executable repair correctness
has not yet been measured for either private-corpus model.

Validation: 18 CPU tests passed, including configurable selection caps and
exclusion of reserved held-out code; changed Python files pass Ruff.
