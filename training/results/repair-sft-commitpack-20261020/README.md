# Larger multi-language SFT — 2026-10-20

The 576-example arithmetic pilot improved termination and runnable output but
scored 0/18 held-out repairs before and after training. Its validation answer loss
worsened from 0.8614 to 1.3072. See the [actual generations](../repair-sft-20261020/examples.md).
This experiment broadens the data and lengthens training; it does not establish
that more training fixes generalization.

## Data prepared

[Preparation report](data-report.json): immutable CommitPackFT revision
`fc56fe33c030c6daa414c2b112c932b8eed085e6`, 37,388 training examples and 561 validation
examples spanning 21,429 repository/alias groups. Training contains 18,556,003
input/answer sequence tokens, including 9,112,832 supervised answer tokens.

| Language | Training | Validation |
|---|---:|---:|
| Python | 6,000 | 64 |
| JavaScript | 6,000 | 64 |
| TypeScript | 3,276 | 64 |
| Go | 2,984 | 64 |
| Rust | 1,353 | 64 |
| C | 3,676 | 64 |
| C++ | 2,099 | 49 |
| Java | 6,000 | 64 |
| Bash | 6,000 | 64 |

Examples pair commit instructions with complete old/new files. They include fixes
and other edits; repository tests were not rerun. Repository aliases are grouped
before splitting across all languages. Normalized code duplicates and original
synthetic validation code are excluded across splits. Semantic duplicates and
pretraining overlap are not ruled out.

Private dataset: `lm-exploration-training:/repair-commitpack-20261020-v1/`.
The report records the immutable upstream revision, per-language download hashes,
builder hash and exact generated JSONL hash. Raw third-party code stays on the
private volume with per-record provenance and license metadata.

## Submitted training

[Durable deployed worker](https://modal.com/apps/radiantai/main/deployed/simple-llm-repair-sft-scale);
submission identity and settings are in [launch.json](launch.json).

- Run: `repair-sft-commitpack-20261020-r01`.
- Parent: original PR #26 `simple-llm-code60-fresh-5b-20261019-r02-s02/step-000066065.pt`;
  the overfit arithmetic SFT checkpoint is not the parent.
- One H100, 5,000 total updates, batch 8, context 2,048, constant LR 1e-5.
- 40,000 example draws with replacement if all updates finish; these are not
  40,000 distinct examples. The dataset is approximately 65× larger, the update
  target 25× longer and the example-draw target 50× larger than the pilot.
- Validation and full optimizer checkpoints every 250 updates.
- Four-hour training-loop bound, five-hour worker timeout, no automatic retry.
- Same answer-only loss and exact-resume engine qualified during the pilot;
  this change extends the time bound and adds a volume-backed larger worker.

At submission, no new quality result is available. Check the call ID for completion,
the app logs for progress and volume `metrics.jsonl` for validation. On normal
completion the worker writes `result.json`, including the actual update count,
answer-token count and checkpoint. Do not interpret job submission as completion
or loss improvement as executable repair success.

Validation of this change: 15 local tests passed (data contracts, token masking,
padding, curriculum, repository alias grouping and provenance filters); Ruff and
`git diff --check` passed. The CPU preparation job completed successfully:
https://modal.com/apps/radiantai/main/ap-XD63Aw2eQjSmw66YmQUUsM.
