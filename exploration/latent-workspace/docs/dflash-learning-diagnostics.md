# Conditional learning diagnostics

User direction: if the substantial fresh data/response grid remains weak, inspect
activations, loss and gradients to determine whether instability or a conditioning
bottleneck is limiting learning. This is authorized follow-up; it does not replace
the ongoing five-condition study or justify a diagnosis before measurements.

## Existing measurements

Current training records loss and accepted NEW tokens by proposal position/domain,
plus sampled global gradient norm before clipping, feature-projection gradient norm
AFTER clipping, learning rate and batch loss at evaluation checkpoints. These are
sparse snapshots, not clipping-frequency measurements or per-layer gradient traces.
They cannot establish that gradients are healthy everywhere. Frozen-target gradients
should remain absent by design; inspect gradient flow in the trainable draft only.

## Trigger and controlled diagnostic batches

After sufficient fresh training, if the quality gate remains unmet or curves show
poor fitting/instability, first inspect the recorded losses and gradient snapshots.
Then use archived early/middle/selected/final checkpoints and fixed mixed-domain
training/validation batches to run a separate diagnostic replay. Preserve checkpoint
weights, optimizer/RNG provenance, masks, feature depths, response caps and labels.
Do not add hooks or change package source midway through active grid fits/resumes.
Additional diagnostic models/probes train only on training data; test stays sealed.

## Measurements

1. **Loss:** weighted training objective and unweighted token NLL separately; domain,
   proposal position and context-position slices; accuracy/confidence/entropy;
   early versus late response anchors and repetitive versus less-repetitive cases.
   Separate failure to fit from a train/validation gap.
2. **Gradient flow:** per-layer and per-module norms before AND after clipping,
   gradient/weight ratios, actual optimizer-update/weight ratios, zero/nonfinite
   fractions, and feature-projection contributions from each target depth. Collect
   dense traces during an isolated replay to estimate clipping behavior; do not
   describe replay frequency as original training history. Account for learning
   rate, label weighting and parameter scales before calling gradients vanishing.
3. **Activations:** RMS, spread, outliers and nonfinite fractions before/after the
   feature projection, normalization, attention and MLP branches; residual/update
   ratios across all five draft layers. Inspect attention-logit scale, entropy and
   allocation to target context versus block tokens on fixed small cases without
   changing the production SDPA path. A large norm alone is not gradient explosion.
4. **Conditioning information:** compare predictive readouts before versus after
   projection/normalization with explicit capacity controls and held-out labels;
   inspect feature contribution scales and effective rank as descriptive evidence.
   Use fixed-input ablations/interventions to check whether target context affects
   proposals. Low activation variance/rank or zero-memory failure alone does not
   prove useful information was destroyed.

## Follow-up decisions

Choose one-factor fixes only after identifying a measured failure: e.g. scaling or
normalization changes for unstable magnitudes; optimizer/LR changes for excessive
clipping or negligible updates; conditioning/projection changes for a supported
information bottleneck. Check suspected reduced-precision faults on identical
inputs in FP32. Validate every remedy against the same held-out metrics and
actual verified rollout acceptance, with adequate optimization exposure.

Diagnostic traces and curves should be saved as artifacts with checkpoint, source,
data and configuration hashes. No useful-drafter success, exploding/vanishing-gradient
finding or projection-information-loss claim is established by this plan.

References: [fresh-grid protocol](dflash-fresh-grid.md),
[DFlash implementation comparison](dflash-fidelity.md),
[existing results](../results/dflash-improvement-findings.md).
