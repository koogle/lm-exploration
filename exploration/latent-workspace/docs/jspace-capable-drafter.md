# J-space conditioning after a capable baseline

User direction: publish the short drafter learnings first, revisit J-space on the
qualified drafter, then investigate unverified two/three-token block generation
and benchmark quality loss against the frozen base model. No native backbone
pretraining, collaborative module or external pretrained draft weights.

## What J-space means in this study

The [paper](https://transformer-circuits.pub/2026/workspace/index.html) defines a
sparse nonnegative cone of token-indexed J-lens directions in the original
residual basis. Its J-space component approximates an activation by a nonnegative
combination of typically25 directions, found with gradient pursuit. The old
experiment instead averaged top25 token vectors after lens readout; that is not
this projection. Preserve the old negative result without treating it as a test
of the paper's full construction on our strong drafter.

Fit average Jacobians from source blocks2/7/12/17/22 to block24, using the pinned
Anthropic estimator: sum target-position cotangents, average source positions,
skip the first16 and final position. Calibrate on1000 fixed, source-distinct,
balanced training-only128-token committed prefixes; never upcoming block labels.
Prompt/committed teacher tokens may form a prefix, but at least8 later tokens
are excluded. Preserve source identities, prefix hashes and calibration counts.
The helper reuses the already tested independent Jacobian estimator in this repo.

For Qwen RMSNorm, vocabulary directions are `(U * final_norm_scale) @ J_l`.
Normalize atoms without changing their cones. Approximate sparse pursuit using
an initial256-atom shortlist and nonnegative Gram-matrix refits, k25. Return the
reconstructed vector in the original layer basis. This is a documented support-
search approximation, not an exact global cone projection or an SAE. Profile
full-vocabulary scoring and per-token projection before selecting training
throughput/storage. Do not silently replace it with the old softmax-vector proxy.

## Planned controlled comparison

Use the qualified step70585 draft as the common competent initialization; all
arms use the same196k teacher sequences and exact sample/anchor streams. The
frozen Qwen target, tokenizer, five draft layers, normalization and block8 remain.
Do not compare a newly untrained J head against an already trained raw head.

Primary question: keep raw conditioning and add a semantic side signal. Three
primary arms:
1. Raw-state conditioning control.
2. Raw states plus the sparse J-space component.
3. Raw states plus an equal-sparsity random-frame projection, controlling added
   nonlinear features/capacity; use fixed orthogonal rotations of each J dictionary.

J-only conditioning is a later optional ablation, not the primary experiment.
User clarifies that J-space should add to the existing hidden-state signal.

For matched trainable capacity, retain the trained raw projection frozen and add
one equally sized projection per arm. The control adds a zero-initialized raw
correction; augmentation arms add zero-initialized J/random corrections. All
three start at identical baseline predictions, with identical trainable counts.
Keep other draft weights trainable identically. Do not replace raw conditioning
in the main comparison. This isolates
conditioning relative to a working trunk; it is not another full from-scratch fit.

Declare30000 updates/80.64M NEW-label exposures per arm, common lower LR and
schedule before fitting, with learning curves/domain/position metrics and paired
source-group uncertainty. Only calibration/projection plumbing is implemented
now; these fits have not run. A negative augmentation result can reflect extraction/calibration or adaptation
difficulty; it is not proof that no future information is present.
A representation derived from raw states does not add independent information;
any gain concerns accessible features/inductive bias and finite-budget training.

Reserve a new untouched evaluation holdout before choosing new hyperparameters;
the prior189-prompt drafting test is consumed. Use validation for choices, then
freeze the new comparison before new test access. Repetition/context-position
slices matter; primary macro/domain gate alone is insufficient for causal claims.
No sealed-test re-evaluation of competing J variants on the already used set.

## Following stage: commit drafts without verification

After the conditioning comparison, freeze the selected method and compare base
AR with unverified blocks of2 and3 TOTAL committed tokens: one base-model anchor
plus1/2 draft predictions. The base still computes context features once per
block; this is not a drafter operating without a context backbone. If testing2/3
new draft tokens instead, label total blocks3/4 explicitly. Use the same trained
head's early positions initially, then decide from validation whether smaller-block
training is necessary. No correctness parity guarantee applies after verifier removal.

First measure long free-running divergence, repetition, invalid code, EOS behavior
and block-position errors. Then freeze prompts/decoding/output budgets and run
independent benchmarks: code functional correctness, translation chrF/BLEU, and
prose/reference-text likelihood under the actual normalized block-generation
factorization. Compare task quality and wall-clock throughput separately against
the same frozen Qwen, with paired cases and uncertainty. Public benchmark overlap
with the target's pretraining is unknown; use identical cases across methods.
No benchmark or unverified-generation run has been executed in this new stage.

## Semantic-side-channel intuition and efficiency

User's hypothesis is that explicit verbalizable concepts help the drafter use
semantic features it has already learned, beyond consuming unstructured residual
states. This is plausible as an inductive-bias/accessibility hypothesis; the
J-space is derived from the same hidden states, not independent new information.
The primary comparison retains the capable raw path and adds a learned,
zero-initialized projection of the sparse semantic component. Random-frame
controls help distinguish semantic alignment from generic nonlinear feature help.

If quality improves, measure accepted NEW/call and end-to-end milliseconds/token
including J extraction, adapter/context-cache updates and target verification.
Cache only features for newly committed tokens at inference, retaining existing
raw/J cache for earlier context. For unverified2/3-token blocks, compare benchmark
quality at identical output budgets alongside total latency. Prediction gain alone
does not establish a speed gain; exact sparse extraction may cost more than it saves.
