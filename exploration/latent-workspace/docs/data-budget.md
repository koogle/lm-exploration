# Drafter data budget: research and first learning curves

## What is known

The first WikiText pilot supplied each decoder only 24,000 target-token slots
(750 updates × 8 examples × 4 tokens). Context exposures are not output labels;
random masking further reduces the number of supervised positions. Its repeated
punctuation and high held-out loss do not establish a representation limitation.
The readout now includes Qwen's frozen final normalization, with a regression
check against actual Qwen logits. Historical results used the earlier readout.

[DFlash](https://arxiv.org/html/2602.06036v2) is a close architectural reference:
it conditions a lightweight diffusion drafter on multiple target hidden features,
shares frozen embeddings/output weights, and uses target verification. Its
training distribution and verification objective differ from our standalone
four-token generator. Published sequence counts cannot be converted directly into
our number of supervised four-token blocks. We therefore measure learning curves
before choosing a full training budget or attempting a broad conditioning sweep.

## Controlled initial experiment

Use frozen Qwen2.5-0.5B, one fixed diffusion decoder and selected residual layers.
Train every model on an equal mix of prose, code and English-to-German translation.
Nested training subsets contain 768, 3,072 and 12,288 unique four-token windows.
Each starts from identical initialization and gets 100/500/1,000/2,000 update
checkpoints, batch size 16. At the final checkpoint each sees 128,000 target slots;
record the actual masked positions receiving loss separately. Compare dataset
sizes at equal updates, and optimization progress within each dataset size.

The subsets are exploratory brackets, not a claim that 12,288 windows suffice.
Validation has 64 windows per domain; test has another 64 per domain and remains
sealed. These are small screening samples, unsuitable for definitive significance
claims. Split prose by official article partition, code by repository hash, and
translation by official partition with source deduplication. Quarantine normalized
exact duplicates; near-duplicate and pretrained-model contamination are unresolved.

Report per-domain all-masked token NLL, per-position accuracy, iterative block
accuracy, and a balanced training diagnostic. Train on human corpus targets in
this first study. Target-generated continuations/distillation should be a separate
controlled follow-up if agreement with Qwen is the intended drafting objective.
Do not interpret ground-truth agreement as speculative acceptance rate.

## Decisions after measurement

- If training and validation loss still improve with updates, extend optimization
  before concluding that unique data is insufficient.
- If more unique data helps at equal updates, expand the largest bracket by 4×.
- If training improves while validation stalls, prioritize diversity/data volume.
- If even training loss stalls at poor quality, test tiny-set memorization,
  readout calibration, learning rate and conditioning before buying more data.
- Once curves approach a practical plateau, repeat the candidate budget with
  three seeds and larger validation partitions. Select a budget using validation,
  freeze the protocol, then evaluate untouched test once.

Layer scans, all-layer attention, J-lens ablations and larger Qwen models remain
next-stage experiments. This first study does not establish minimum data needs,
quality parity, lossless decoding or a speed advantage.

## First measured outcome

[The completed L40S study](../results/data-study-01.md) shows smaller-set
overfitting and better validation NLL with more unique data. It does not yet
produce useful generation or establish a sufficient budget. Next: a controlled
12,288/49,152-window comparison with longer optimization, preserving this study's
held-out groups and revisions. This recommendation is not an executed result.

## Follow-up diagnostic gate

The high absolute loss and repetitive outputs require a decoder/training diagnosis
before attributing the problem solely to dataset size. A 96-window memorization
run uses the original training pool and frozen validation/test tensors. It records
all-masked training accuracy as well as loss. A zero-training prefix-only baseline
repeats Qwen's next-token distribution at every block position; its first position
is especially useful for detecting loss of pretrained predictive information.

`study_parent` reloads the parent corpus and pins its model/source revisions.
Expansion retains clean original training rows, appends distinct windows, excludes
held-out groups/content hashes and carries validation/test tensors unchanged.
The original full-pool training order is retained as the nested subset's prefix.
The proposed 12k/49k, 8k-update expansion is configured but gated on diagnosis.

The original 96-window memorization run reached about 1.71 training NLL but only
26% all-masked training-token accuracy after 2,000 updates. This fails a practical
memorization sanity check. Qwen's first-token validation NLL is 1.15–2.69 across
domains, versus 6.77–7.37 for the original 12k-window drafter. Later-token losses
from repeating Qwen's distribution are poor, as expected; it is not a solution.

Two explicit diagnostic options are under evaluation before more data spending:
`residual_readout` starts at the frozen final residual state and learns a zero-
initialized correction; this adds final-state conditioning and changes readout
initialization/normalization, so it is not a pure intermediate-layer ablation.
`position_readout` uses a separate output projection at each block position to
check whether shared decoding loses position information. It increases trainable
parameters by about 1.38M for four tokens. Neither is a validated fix yet.

The position-specific 96-window run achieved 99.2% training-token accuracy at
500 updates and 99.5% at 2,000 (NLL 0.035). Validation is deliberately terrible
for this overfit tiny set; this establishes trainability, not generalization.
The residual-only 12k-window run reached validation NLL 5.96 at 4k updates and
preserved much of Qwen's first-token quality, but later positions remained weak.
The combined readout is now being compared at 12k/49k windows and up to 8k updates.

Future evaluations also record greedy prefix agreement with Qwen along the
proposed block: only consecutive matching tokens before the first mismatch count.
This uses no reference continuation labels and is separate from ground-truth
accuracy. It is an evaluation diagnostic, not a deployed verifier, speculative
sampling correctness proof or measured end-to-end speed improvement.

## Window-level audit correction

The first expansion stopped before training because full-document deduplication
left repeated context/target windows. The original 12,288-row pool contains 12,213
distinct windows (75 duplicates), with one exact training/validation overlap and
no training/test or validation/test overlaps. Historical validation results are
exploratory and are marked accordingly. The corrected comparison preserves all
held-out tensors but necessarily changes the training control: remove duplicates
and the overlapping window, then fill the smaller balanced subset from clean data.
Both new sizes share this cleaned nested pool. The corrected collector reserves
held-out windows first and checks window-level duplicates in addition to groups
and document/source hashes. Near-duplicate and pretraining contamination remain
outside this exact-match audit.

The corrected run's audit confirms the overlapping validation window had nine
training copies. Cleanup removes those nine plus 67 other duplicate rows (76
rows total). The original 96-window checkpoint predicts the same token at all
four positions on 100% of training examples, directly confirming position collapse.

The cleaned 12k/49k comparison completed: 8k-update NLL 6.001/5.738; Qwen greedy
prefix agreement 0.760/0.901 tokens out of four. The paired domain-stratified
source-group bootstrap gives larger-minus-smaller NLL -0.263 [-0.402, -0.133],
conditional on this single seed and small validation set. This supports checking
196,608 windows, not declaring the decoder useful. The next run retains the same
architecture, model, corpus revisions, and held-out tensors. Feature caching now
stores only the residual planes consumed by this head and avoids whole-pool copies;
it preserves training tensors and arithmetic while reducing host memory use.

A separate final-layer-only control tests whether intermediate conditioning is
responsible for poor loss. Saved-head inference also tests a Qwen-provided first
token followed by three drafted tokens, inspired by the known-token anchor in
[DFlash §4](https://arxiv.org/html/2602.06036v2). This is reported separately from
four newly drafted tokens and does not introduce a verification service.

## Clean size and optimization follow-up

Completed seed-42 controls (same frozen validation tensors):

| Windows | Batch | Best observed update | Best validation NLL |
| ---: | ---: | ---: | ---: |
| 12,288 | 16 | 4,000 | 5.921 |
| 49,152 | 16 | 8,000 | 5.738 |
| 196,608 | 16 | 8,000 | 5.690 |
| 49,152 | 128 | 2,000 | 5.511 |
| 196,608 | 128 | 4,000 | 5.264 |

[Combined curves](../results/data-scaling.png) distinguish unique corpus size,
optimization and generation agreement. Batch-size comparisons change token
exposures as well as gradient variance; they do not isolate batch size alone.
At 4k updates, batch 128 has 2.048M target slots and 1,279,980 actual masked
labels. The 196k pool has 786,432 distinct target slots; context tokens are not
labels. At batch 16 / 8k updates only 128k windows are sampled with replacement,
so not every available window was necessarily observed.

49k overfits by 4k updates (NLL 5.670). Comparing each run's best checkpoint gives
a 0.247 NLL advantage to 196k; comparing final checkpoints alone exaggerates it.
Even the larger run agrees with Qwen for only 1.057 consecutive tokens out of
four. Quality is not solved. These losses are all-masked marginals, not sequence
perplexity and not directly comparable with autoregressive teacher-forced loss.

Negative diagnostic controls: final-layer-only conditioning reached NLL 6.002
at batch 16 / 4k, versus multi-layer 5.816. Parameter-free token-input RMSNorm
reached 5.540 at batch 128 / 2k versus 5.511 without it. Normalization remains
off by default. Known first-token anchoring and left-to-right unmasking of the
49k final checkpoint gave only about 0.3–0.5 additional consecutive Qwen matches;
teacher-forcing earlier block labels also left large later-position losses.
These checks do not support solving the problem with decoding order alone.

Current bounded runs extend the 196k batch-128 checkpoint to 8k/16k updates and
repeat 49k/196k through 4k with seeds 43/44. Roughly 200k windows is a candidate
pilot pool, not an established minimum or a production-quality recommendation.
All comparisons remain exploratory on 192 validation windows and sealed test.

## Completed repeated-seed outcome and budget decision

Seeds 42/43/44 all favor 196k over 49k at a maximum of 4k batch-128 updates.
Best observed checkpoints are at 2k for 49k and 4k for 196k in every seed:

| Seed | 49k best NLL | 196k best NLL | Difference |
| ---: | ---: | ---: | ---: |
| 42 | 5.511 | 5.264 | -0.247 |
| 43 | 5.639 | 5.280 | -0.359 |
| 44 | 5.557 | 5.348 | -0.209 |

Mean paired difference is -0.272. These share the same small validation set;
three training seeds do not remove validation-selection bias or uncertainty
about new source groups. Exact window audits pass for every new run.

Seed-42 continuation preserves optimizer and RNG state. At 8k/16k updates NLL
is 5.222/5.201, versus 5.264 at 4k. The final run has 8,192,000 target-token slot
exposures and 5,120,797 actually supervised masked positions. Final consecutive
Qwen agreement is 1.099/4 and full-block agreement 4.7%. Generated translation
remains malformed and prose/code are unreliable. Later-position training NLL
also remains high, so this is not solely a held-out overfitting problem.

**Decision:** freeze the 196,608-window mixed pool for the next decoder/conditioning
diagnostics. Use batch 128 and 4k updates as an initial comparison budget, with
validation-selected checkpoints; selectively extend changes that improve both
loss and generation. Do not spend on another corpus expansion yet. This is a
workable diagnostic budget, not a minimum sufficient dataset or a quality result.
No larger Qwen or J-space/all-layer hypothesis is validated by these runs.
