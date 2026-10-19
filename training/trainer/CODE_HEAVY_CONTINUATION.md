# Fresh-data 5B continuation, 60% coding

Continue the latest EU-trained 124M Simple LLM checkpoint for another 5B
loss-bearing tokens. Coding rises from 30% to 60%; every other original source
share is multiplied by 4/7. The [plan](../plans/code-heavy-continuation-5b-20261019.json)
records exact rational weights and integer budgets (3B coding, 2B other sources).

| Category | New token share | Phase target |
|---|---:|---:|
| Python-heavy code | 60% | 3B |
| Educational web | 5.714286% | ~285.7M |
| Mathematics | 11.428571% | ~571.4M |
| Scholarly papers | 8.571429% | ~428.6M |
| Books/textbooks | 14.285714% | ~714.3M |

## Freshness and preparation

The user selected fresh data on October 19. No old training records count toward
this phase's capacity. Clone the immutable original snapshot into separate staging
and resume its upstream cursors, keeping the original source revisions, adapters,
filters, tokenizer and parent-level splits. The cloned records serve as a global
exclusion ledger; only new records enter the new training snapshot. Include all
EU document hashes in the exclusion ledger too, protecting previous training and
held-out records across both phases. Previously used validation remains held out.

Keep the existing maximum of two passes **within the fresh corpus**. This requires
at least 2.5B new unique targets, including 1.5B coding targets. Preparation aims
for an additional 5% capacity reserve. Thus 5B training exposures are not a claim
of 5B unique new tokens. Exact hashes are checked; global near-duplicate audits
remain pending, as in the existing authorized experimental-data runs.

CPU preparation uses six parallel fetch/tokenize workers, bounded batches,
compressed-shard caching, and one transactional SQLite writer. Counters track
fresh eligible tokens only; conflict quarantine decrements them. Cursors and data
commit together, including across fresh-process preparation chunks. The final
snapshot contains only fresh training records plus held-out validation records;
export verifies zero prior-hash training overlap and exact agreement with capacity
counters. Immutable old snapshots and checkpoints are never rewritten.

If a pinned source exhausts before its fresh quota, the workflow stops and reports
that source. It does not recycle old data, increase pass limits, or change shares.
Upstream capacity is a measured preparation result, not an assumption.

## Training and validation

Parent: `simple-llm-124m-eu-onepass-20261018-s02/step-000003838.pt`, with
5,878,591,538 lifetime targets. The checkpoint migration checks model ordering,
architecture, framework/tokenizer and finite weights/Adam moments, preserves
Adam/RNG, and resets phase counters/data cursors. Nested lineage keeps lifetime
counts correct. Ordinary resume verification remains strict.

Execution: eight H100s, 8K context, global batch 32, compiled microbatch two,
cropped padding and cached RoPE from PR #17; constant 3e-5 LR. Start with two
updates and then two resumed updates in a new process, requiring identical restore
validation. Then train in bounded six-hour segments, checkpointing every 1,000
updates. The last incomplete optimizer batch is left unused and reported.

`modal_code_heavy.py` deploys `simple-llm-code-heavy-5b`. Its `start` CPU controller
runs restartable preparation chunks before spawning an independent training
controller, so GPU allocation cannot happen before capacity and freshness gates.
Check `inspect(run_id)` for committed preparation/workflow status. Preparation
may use up to five three-hour chunks, each with a four-hour function timeout;
training allows two qualification segments plus at most three six-hour segments.
Failures preserve progress and never trigger automatic duplicate training.

After training, automatically score the existing expanded validation sample:
7,620,187 targets / 2,782 source-family selections, covering all 12 original
sources plus ECB and EMA. Compare against both the original 5B and EU checkpoints
using paired family-bootstrap intervals. Existing baselines are reused. Model
checkpoints and validation samples stay private on Modal.

Deploy from `training/` with `modal deploy modal_code_heavy.py`. Launch once via
`modal.Function.from_name("simple-llm-code-heavy-5b", "start").spawn(run_id, provenance)`.
The provenance must identify the committed runtime source. Deploying alone does
not launch data preparation or GPUs. Actual launch/status belongs in the run record.
