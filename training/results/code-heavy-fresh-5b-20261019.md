# Simple LLM 124M fresh-data continuation

Launched October 19, 2026 at 09:54 Cyprus time (06:54 UTC) on Modal app
`simple-llm-code-heavy-5b`. Runtime source commit: `02d2c00`; stacked PR #26.
The controller is running CPU preparation. No new training updates are claimed.

Requested: another 5B loss-bearing targets, coding 60% (3B), all other original
shares multiplied by 4/7 (2B). Continue the latest EU checkpoint, preserving
weights, Adam moments and rank RNG. Context 8K, eight H100s, optimized compiled
microbatch two from PR #17, constant LR 3e-5.

Only previously unused records count as new training data. The preparation ledger
excludes every original and EU snapshot hash, including held-out records. Existing
upstream cursors resume beyond old data; final export checks zero prior-hash
training overlap. Minimum fresh capacity is 2.5B unique targets under the retained
two-pass limit, with 5% reserve requested. A source running out stops preparation;
old records are not substituted. Capacity and throughput remain to be measured.

The durable workflow queues fresh-process start/restore qualification (two steps
each) and then the complete training phase after preparation passes its gates.
Completion queues the existing expanded 7.62M-target / 2,782-family validation
sample, comparing the new checkpoint with both previous checkpoints. All outputs
remain private on Modal. See [workflow details](../trainer/CODE_HEAVY_CONTINUATION.md)
and the adjacent JSON for call IDs and reproducible settings.

CPU tests: nine passing across fresh preparation, fast ingestion and checkpoint
migration, plus Ruff and whitespace checks. New-data GPU qualification is queued,
not yet completed. Earlier kernel correctness/performance results remain those
recorded for PR #17; no new throughput or end-to-end completion estimate is claimed.

Current status can be read with `modal_code_heavy.inspect(run_id)` or the deployed
`inspect` function. The run's `workflow.json` records stage/errors/checkpoint;
staging `progress.json` records source capacity/cursors. This document records
launch state, not a live completion assertion.
