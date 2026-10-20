# Private patch SFT — 2026-10-20

The private patch pipeline and launch workflow are on PR #28, stacked on PR #26.
This experiment starts from the same original PR #26 checkpoint as the previous
runs. The completed CommitPack SFT checkpoint is not used as the parent.

## Prepared data

[Aggregate data report](data-summary.json): **79,441 training examples**, comprising
62,088 private patch excerpt repairs and 17,353 CommitPack replay examples.
Validation has 1,617 examples: 1,056 private patch edits and 561 previous CommitPack
examples. The training corpus contains 44,774,878 total sequence tokens, including
20,641,168 supervised answer tokens.

| Language | Training | Validation |
|---|---:|---:|
| Bash | 2,501 | 79 |
| C | 6,012 | 192 |
| C++ | 8,718 | 177 |
| Go | 5,384 | 192 |
| Java | 2,481 | 98 |
| JavaScript | 2,485 | 114 |
| Python | 5,766 | 192 |
| Ruby | 919 | 40 |
| Rust | 14,521 | 192 |
| Swift | 595 | 21 |
| TypeScript | 26,345 | 192 |
| Zig | 3,714 | 128 |

All 226 selected source shards downloaded successfully on Modal. Source snapshots
are pinned in [private-patch-sources.json](../../sft/private-patch-sources.json).
Private patches span 15 training repositories and 13 validation repositories;
these repository counts exclude the replay corpus. This is commit-disjoint,
**not repository-disjoint**, validation. Previous CommitPack validation repositories
are excluded from the private patch inputs. All eligible private validation code
hashes are blocked from training, even when beyond the sampled validation cap.
Replay training also excludes private validation repositories and exact held-out
code. Semantic clones and pretraining overlap are not ruled out.

The pipeline verifies Git blob hashes and extracts exact changed regions with
six context lines. Targets are corrected excerpts; they are not necessarily
standalone compilable files. Project tests have not been rerun. License evidence,
raw examples and the detailed source inventory remain on the private volume under
`repair-private-patches-20261020-v1/`; only aggregates are committed here.
Dataset SHA-256:
`6197c4178e737172184d830ed52a3da5a3251ba642a03e20d336d05143b93702`.

Preparation completed at:
https://modal.com/apps/radiantai/main/ap-edX1KgdRWuTGZkYNC3Zi7j.
Earlier 16-CPU and 8-CPU preparation requests were stopped while awaiting capacity;
the successful run used two CPUs and 16 GiB RAM. The earlier workspace outage
occurred before this successful preparation and did not start a private SFT run.

## Training submission

[Launch record](launch.json), run `repair-sft-private-patches-20261020-r01`:
one H100, batch 8, context 2,048, constant LR 1e-5 and 20,000 total updates.
The target is 160,000 example draws with replacement, not distinct examples.
Evaluation runs every 500 updates and full checkpoints every 1,000. A four-hour
training-loop limit and five-hour worker timeout bound each invocation; check
`result.json` for the actual completion count. Resume requires a new output run ID
with identical settings and source/data identity.

The durable deployed worker is at:
https://modal.com/apps/radiantai/main/deployed/simple-llm-repair-sft-scale.
Submission is not proof of completed training or improved repair correctness.
The previous run's final answer loss was 0.34446 on its own validation corpus;
losses on this larger, different validation corpus should not be compared directly.

Validation: 18 unit/integration tests passed; Ruff and diff checks passed. An
additional local check used actual cached source shards from both private
collections. The full prepared dataset passed the trainer's strict tokenization,
identity, language coverage, group separation and normalized duplicate checks.
