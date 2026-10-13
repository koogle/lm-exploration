# Open work

The large-data curriculum training is **complete**. Useful-drafter research and
sealed-test qualification remain unfinished. Use uv; direct main commits/pushes
remain authorized. No active training workers or coordinator remain.

## Completed result

[Final findings](results/large-curriculum-final.md),
[curves](results/large-curriculum-final.png),
[full reports](results/large-curriculum-final.json).

Both separate 79M drafters for frozen Qwen2.5-0.5B finished 100000 updates,
268.8M supervised NEW-token exposures each, random initial draft weights and
fresh optimizers. Train first on 65536 prose + 65536 code examples, then add
translation to reach balanced 196608, restoring own weights/optimizer/RNG and
keeping the global schedule. Dataset transitions occur at 17985/11472 updates
(short/long cap), so domain-exposure timing is a comparison confound.

- 64-token cap: best NEW2.19454 at51475; final2.10009. Best domains
  prose2.55469/code2.39453/translation1.63439. Clear late overfitting.
- 256-token cap: best NEW4.03251 at70585; final4.00693. Best domains
  prose5.33594/code5.10547/translation1.65613. Late loss overfitting, acceptance
  saturation; selected head improves on historical49k/256 best3.60350.
- Both selected heads pass the preliminary validation macro>=2/every-domain>=1
  criterion. All twelve short verified rollouts per head match target output.
  Test remains sealed; no longer-rollout/full-test qualification exists.

Selected files: `/runs/fresh-grid-196608-r64-08/best.pt` and
`/runs/fresh-grid-196608-r256-11/best.pt`. Freeze/checksum the long selected head
and evaluation configuration next. These paths inherit best weights, not final.
Frozen training package SHA:
`47d71073e55a03e87c4bdb59df6aa684cedc532815d30c9f44167ff97da919ae`.

## Remaining qualification and analysis

- Longer verified rollouts and actual acceptance, with FP32 target arithmetic
  matching the existing correctness reference; parity is required.
- Repetition/position/context-length validation slices. Some greedy teacher
  responses are repetitive; acceptance is not a human answer-quality score.
- Freeze selected checkpoint/config and evaluate the sealed drafting test once.
  Preserve historical already-used conditioning test as a distinct partition.
- Full-corpus fitting diagnostics if needed; current train curves sample only
  twelve responses. Per-layer activation/gradient/projection diagnostics remain
  authorized but unexecuted: [plan](docs/dflash-learning-diagnostics.md).
- Repeated seeds/source-diversity controls remain untested. Do not attribute the
  complete gain purely to data size or claim final statistical significance.
- J-space/layer-selection/dropout, collaborative generation, native pretraining,
  larger-target controls and speed work remain deferred.

## Execution and cost accounting

Coordinator09 returned `training complete; comparison and qualification pending`.
Its remote budget/state are archived in `/runs/fresh-grid-controller-09/`.
Local ledger `artifacts/budget.json` is synced to the authoritative remote ledger.
Ownership marker released only after confirming terminal state/no active workers;
handle archived as `artifacts/budget_remote_controller-09-complete.json`.

Current user-reset epoch reservations: **$162.125/$500**, not actual Modal bill.
Historical $468 ledger stays archived separately; failed/canceled reservations
are retained. Do not reset or drop this accounting without user instruction.
Single-use GPU containers prevent retained mmap handles blocking reload between
workers; see [operations](docs/operations.md). The earlier cache handoff failures
were recovered from saved checkpoints with no optimizer/source resets.

The original five-condition grid was changed by user direction: both49k arms
were early-stopped; intermediate98k launch canceled; only the two large curriculum
arms completed their100k schedules. This is not a completed equal-final-exposure
five-way comparison. Historical evidence and negative results remain linked in
[knowledge index](docs/knowledge/INDEX.md).
