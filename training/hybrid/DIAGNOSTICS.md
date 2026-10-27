# What the completed KDA checkpoints tell us

The strongest supported explanation is additional local feature construction:
short convolutions combine neighboring hidden states before the delta-rule memory
is queried and updated. The trained baseline depends particularly strongly on its
value-path history. RoPE changes q/k addressing, but does not supply this value
mixing. This is a mechanistic hypothesis supported by paired evaluations and
interventions, not proof that one component caused the entire training advantage.

## Quality beyond the original validation prefix

All six final checkpoints reproduced their saved validation CE within the
configured 0.002 tolerance. An additional evaluation sampled 128 distinct held-out
documents per source, restricting to documents with at least 257 tokens. Each
contributed one randomly positioned window of up to 1,024 next-token targets.
The 384 documents supplied 276,736 valid targets, shared exactly across variants
and seeds. This sample differs from the original contiguous validation prefix.

| Source | Baseline | No convolution | RoPE without convolution |
|---|---:|---:|---:|
| ClimbMix | 3.81237 | 3.81229 | 3.81804 |
| Code | 3.42954 | 3.44702 | 3.43558 |
| Math | 2.95115 | 2.96312 | 2.96463 |

Means average the two trained seeds. Reweighting sources 70/20/10, removing
convolution increases CE by 0.00464; RoPE without convolution increases it by
0.00652. Paired document bootstrap intervals are [0.00158, 0.00764] and
[0.00362, 0.00942], respectively. These intervals describe document-sampling
uncertainty conditional on these trained models, not uncertainty across training
seeds. The no-convolution difference is approximately zero in seed 17 and +0.00946
in seed 29; the RoPE difference is positive in both (+0.00463, +0.00842).

Baseline's advantage over the no-convolution control is concentrated in code and
math. On the equally sampled source pool it also has lower loss on newlines,
numbers, punctuation/operators, and word/identifier tokens. Whitespace is a small
exception. These are token-category descriptions, not evidence of improved
mathematical reasoning or executable-code correctness. Decoded examples include
wins and losses for both baseline and RoPE; the archived example selection is
explicitly illustrative, choosing large same-sign differences in both seeds.

## Direct interventions on the trained models

On the same 48-document subset, preserve the current-token convolution coefficient
and zero only the three past-token coefficients, restoring weights after each test.

| Baseline intervention | Mean CE increase | Seed 17 | Seed 29 |
|---|---:|---:|---:|
| Remove q history | 1.17377 | 0.39418 | 1.95336 |
| Remove k history | 0.71109 | 0.87628 | 0.54590 |
| Remove v history | 2.49050 | 2.54383 | 2.43717 |
| Remove all three histories | 2.67925 | 2.73982 | 2.61868 |
| Cast recurrence inputs to BF16 | 0.000057 | 0.000074 | 0.000040 |

The value-path result is large and consistent across seeds. Learned convolution
coefficient RMS, averaged across layers and seeds, has a useful asymmetry:

| Path | Previous token coefficient | Current token coefficient |
|---|---:|---:|
| q | 0.0607 | 0.0971 |
| k | 0.0889 | 0.0693 |
| v | 0.1151 | 0.0722 |

Queries emphasize the current hidden state; keys and values emphasize the preceding
hidden state, with smaller contributions from two and three tokens back. These
are coefficient magnitudes, not a direct measure of activation contribution.
The ablations establish that temporal contributions matter to the fitted model.
The input hidden states already contain context from earlier layers; this is
additional local mixing at every KDA layer, not the model's only access to context.

Disabling RoPE in the trained RoPE model increases CE by 0.20142 and 0.42287.
RoPE is therefore used by that model. Its usefulness does not mean it replaces
convolution's feature construction. Likewise, inference ablations induce a
substantial distribution shift: their large loss changes must not be interpreted
as the expected gap between separately retrained architectures. BF16 inference
sensitivity is tiny here, but this does not rule out effects of precision on the
original training trajectories.

## Gradients and actual AdamW updates

The checkpoints contain weights, optimizer moments, step counts and data cursors;
individual historical gradients were not saved. We recomputed token-mean CE
gradients on the same 12 held-out documents and inspected Adam's stored first and
second moments. We then resumed the optimizer for exactly one new training batch
from the saved data cursor, including z-loss, clipping, and the saved learning
rate. All six next-batch hashes match. These diagnostic updates were not written
to the original checkpoints.

| Variant | Raw gradient norm, seeds 17/29 | Training CE improvement, seeds 17/29 |
|---|---|---|
| Baseline | 4.59 / 4.22 | 0.02426 / 0.02428 |
| No convolution | 3.25 / 3.36 | 0.02451 / 0.02470 |
| RoPE without convolution | 3.14 / 3.17 | 0.02492 / 0.02481 |

All norms exceed the shared clipping threshold of 1. Actual update L2 divided
by parameter L2 ranges from 8.63e-5 to 9.14e-5. All convolution taps receive
nonzero held-out gradients; mean tap-gradient RMS is of order 1e-3. This probe
shows no stalled optimizer or dead convolution taps. Baseline actually has the
larger raw gradient norm, while its one-batch training improvement is slightly
smaller. The results support a representation/generalization explanation more
than an explanation based on failed learning in the alternatives. One batch
cannot diagnose the entire training trajectory or establish optimizer stability.

The held-out loss changes after the fresh update are small and positive for all
six models, approximately 0.00008–0.00095. First-order gradient-dot-update
predictions are archived alongside them; they do not capture curvature and
numerical effects. The variable-length context probe exceeded PyTorch's default
FlexAttention compilation-cache limit, so subsequent gradient/update diagnostics
used its unfused fallback. They are not kernel-throughput measurements. Original
loss reproduction, paired document losses and convolution interventions preceded
that fallback.

## Exploratory recurrent-state probe

A separate probe replayed the exact FP32 reference recurrence on one 256-token
code prefix in all six models. Averaged over KDA layers and seeds, baseline's
adjacent-key cosine was about 0.43, versus 0.55 without convolution and 0.52 with
RoPE. Its prediction-error/value-energy ratio was about 0.56, versus 0.67 and
0.68; write/state-energy ratios were about 0.08 versus 0.11. The learned baseline
therefore produced more distinct neighboring keys and smaller relative memory
corrections in this example. This single-prefix probe is illustrative and does
not establish population-wide memory behavior.

The short-prefix test also shows that all models benefit substantially from
context. Its small selected target set does not establish a general long-context
advantage; truncation resets RoPE positions as in ordinary model calls.

## Next useful experiment

Keep a learned value convolution when adding RoPE, then separately test keeping
key history as well. Train these as matched variants: inference deletion cannot
answer how they will learn. In parallel, fusing the existing short convolution
could preserve the measured baseline behavior while recovering implementation
speed. Neither proposal has been validated by this diagnostic study.

Code and the concise findings are in this branch. Full gradients-by-group,
optimizer-update statistics, token losses, samples, interventions, reference-state
probes, bootstrap results and the figure are archived in the private HF dataset
`jfrick/lm-exploration-results`, folder
`training/results/kda-diagnostics-1793140697`. The original training results are
linked from [the GLM study](GLM_STUDY.md).
