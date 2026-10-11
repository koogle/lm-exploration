# Small-target drafter improvement series

Predeclared before new runs, 2026-10-11. User requests sustained studies based on
DFlash differences until a useful novel drafter exists for the small frozen Qwen.
The target is Qwen2.5-0.5B, pinned as before. The trainable network is our separate
randomly initialized drafter (31M with two layers), not the frozen 0.5B target.

## Quality gate and isolation

Working usefulness gate: macro average at least 2 consecutive correct NEW tokens
out of 7, with at least 1 in each prose/code/translation domain on validation;
then freeze the checkpoint/configuration and evaluate sealed test once with the
same gate. Require verified greedy output equality on an expanded prompt check.
Report uncertainty and per-position accuracy. This is a study criterion, not a
production quality or speed guarantee. No speed optimization or J-space sweeps.
If the gate is not met, report failure honestly and continue within the existing
$500 conservative allocation cap. Current reservations before this series: $258.25.

## First studies

Keep the 6144 cached mixed-domain teacher responses, seed, 192 validation responses,
anchor+7 masked proposals, batch 3, 8 random anchors/sequence, 6e-4 learning rate,
4% warmup and 20000-update schedule fixed. Test data stays sealed. Existing
`dflash-scale-6144-01` is the historical two-layer/no-QK-norm control.

1. Bounded 300-update/96-response GPU smoke for opt-in query/key normalization;
   this validates the new path, not useful learning.
2. `dflash-fidelity-norm-01`: two layers plus per-head query/key RMSNorm before RoPE.
   Common weights have exactly identical seeded initialization to the old variant;
   new norm scales start at one. Keep QKV biases to isolate normalization alone.
3. `dflash-fidelity-depth-01`: five layers with the same normalization. This isolates
   added depth relative to study 2, not relative to the historical baseline.

Each long worker has a 3600-second hard timeout and $5 reservation. Incomplete
workers resume with optimizer/RNG restored under an absolute 20000-update schedule.
A short smoke reserves $1.25. Run full-training fitting audits of selected variants.
Gradient diagnostics record global norm before clipping and feature-projection norm
post-clipping; a nonzero gradient alone does not establish useful conditioning.

Adapt subsequent studies to these results: fixed five-depth feature conditioning
requires recaching the same token sequences rather than regenerating responses;
then assess optimization exposure and larger/longer mixed-domain teacher corpora.
Keep one-factor comparisons where possible and label any bundled changes. Do not
claim convergence based solely on a fixed screening budget. Add zero/shuffled-memory
controls to test reliance on conditioning. Preserve all negative results and curves.

Reference: [DFlash paper](https://arxiv.org/html/2602.06036v1),
[pinned source and fidelity audit](dflash-fidelity.md). Our QK-only variant is still
an adaptation, not a full reference reproduction.

## Feature-depth extension launched after successful cache check

`dflash-fidelity-cache-01` adds post-block layers 7/12/17 to the archived 6144-response
fit's layers 2/22. The ordered teacher-token SHA matches exactly; existing feature
planes are reused bitwise. New planes use a masked BF16 target forward. No teacher
answers are regenerated. `dflash-fidelity-features-01` uses five draft layers and
five fixed feature layers with QK normalization, matching study 3's data, schedule
and all other training settings. Its larger projection has more parameters; do not
attribute differences solely to feature information independent of capacity.

## Larger pool preparation

Prepare 26400 requested mixed-domain training prompts to support a balanced 24576
usable-response subset. Reuse the original pool's train/validation caches; register
both its manifest and the older pilot before sampling additional source groups.
Hold validation/test prompt counts fixed. Existing response tokens and features
are not regenerated, and validation tensors are hash-checked before/after extension.
The new teacher batch sees only new/previously uncached training groups. Record
short/duplicate exclusions and actual usable counts. This preparation does not
choose the next training architecture or establish sufficient optimization exposure.
