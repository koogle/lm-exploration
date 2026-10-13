# Open work

The large-data curriculum training is **complete**. The selected long-response drafter has now passed longer-rollout and sealed-test
qualification. Further architecture/representation research remains open. Use uv; direct main commits/pushes
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
  [Qualification](results/qualification-findings.md) is complete for the selected
  long-response head:24/24 longer-rollout parity, test NEW3.93836 and every
  domain>=1; test is now consumed.

Selected files: `/runs/fresh-grid-196608-r64-08/best.pt` and
`/runs/fresh-grid-196608-r256-11/best.pt`. The long selected head and evaluation configuration are frozen and checksummed
in `results/qualification-frozen.json`. These paths inherit best weights, not final.
Frozen training package SHA:
`47d71073e55a03e87c4bdb59df6aa684cedc532815d30c9f44167ff97da919ae`.

## Completed qualification and remaining research

- Longer rollout task completed:24/24 outputs and3906 generated tokens match
  FP32 target greedy output. Actual NEW/call prose4.514/code4.300/translation1.195.
- Repetition/position/context-length validation slices. Some greedy teacher
  responses are repetitive; acceptance is not a human answer-quality score.
- Frozen sealed-test task completed once:189 usable prompts (63/domain),
  NEW3.93836/NLL2.09540; two duplicate responses and one short response excluded
  by the fixed policy. No test-based retuning. The drafting test is now used;
  future model choices require validation and a new final holdout. Preserve the
  historical conditioning test as a distinct already-used partition.
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

Current user-reset epoch reservations: **$167.125/$500**, not actual Modal bill.
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

## Authorized qualification execution

User requests both pending tasks. Frozen selection: long-response step70585.
[Protocol](docs/dflash-qualification.md) and `configs/qualification-long-196k.json`
are recorded before GPU execution. Standalone evaluator tests leakage/parity
guards and does not modify the frozen training package. One bounded qualification
input will perform24 longer validation rollouts, then192 reserved test prompts
only if parity passes; GPU execution completed: `qualification-long-196k-01`,240 seconds. See
`results/qualification-findings.md`, full report, chart and conditional
source-group bootstrap intervals. Both requested tasks pass. Only the selected
long-response checkpoint was qualified; the short arm was not selected for test.

## New user-directed stages

Short write-up published: one-page PDF with performance and scoring visuals, linked in
README. Next: actual sparse J-space conditioning on the qualified head, primarily
augmentation alongside raw states, with raw and random-frame controls. J-only is
a later optional ablation after the user clarified the semantic side-channel goal. See
`docs/jspace-capable-drafter.md`. Training-only Jacobian/projection helpers and
local safeguards are implemented; bounded real-model profiling is pending.
No new J comparison or unverified/benchmark result exists yet. Prior consumed
test remains unavailable for selecting these variants.

Write-up corrected: distinguish severe small-set memorization from modest late
validation degradation in the large long-response fit. Middle/right charts now
share consecutive-match metric and 0–7 scale; late NLL rise is explicitly labeled.
The tiny 12-example training diagnostic is not a full-training generalization gap.

One-pager now typeset in LaTeX with automatic wrapping and explicit column break.
Rebuild: `uv run --with matplotlib python scripts/render_drafter_learnings.py`
(requires pdflatex, pdfinfo and pdftoppm). Source saved in
`results/drafter-learnings.tex`; build checks one page and no overfull boxes.

One-pager visuals consolidated: validation curves for prose/code/translation and
equal-domain average, plus illustrative token-level scoring. Domain means are
block-weighted, not means of per-prompt averages; the anchor and tokens after the
first mismatch are excluded. Architecture comparisons remain in the prose.


## Expanded research PDF (2026-10-13)

The report now retains the visual summary as page one and adds two pages covering
the architecture, training curriculum and hyperparameters, verified-generation
integration, qualification evidence, negative findings and remaining J-space work.
The same renderer now checks **three pages** and no overfull boxes; the earlier
one-page build notes above describe the previous version. README links point to
the expanded report. This documentation update did not launch training.
