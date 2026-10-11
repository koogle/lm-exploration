# Small-data convergence and layer controls

A window is 128 prefix tokens plus four future target tokens. An optimizer update
samples a batch of windows with replacement. Neither update counts nor reporting
intervals change the four-token generation block. Learning rate remains 0.0003;
the denser reporting schedule is not a learning-rate sweep.

Corpus extraction is a separate stride: prose/code windows advance by 132 tokens
(nonoverlapping chunks); translation advances four German target tokens, reusing
source/context. At most four windows per source record are retained. A count such
as 3072 or 196608 means distinct training windows, not context length. The latter
contains 786432 distinct target-token slots; prefix tokens are conditioning, not
additional supervised labels. Adjacent translation windows are correlated, hence
document/pair/repository separation matters for evaluation.

The exploratory follow-up `small-layer-controls-01` uses nested 96/768/3072-window
subsets of the existing clean, balanced 196608-window training pool. Batch 32,
checkpoints 25/50/100/200/400/800/1600/3200 updates, all training windows measured
at every checkpoint. At 3200 updates the expected sampled passes are
1066.7/133.3/33.3 respectively; replacement sampling is not a literal epoch.
The same existing 192 validation windows are reused. The previously evaluated
test set is not evaluated again. This is an adaptive diagnostic, not confirmation.

Four otherwise matched diffusion heads attend all 24 residual layers: baseline,
25% whole-layer dropout, a null attention option, and both. Dropout is independent
per example and layer during training only. If every layer is dropped without a
null option, retain the final layer. The null option appends exactly zero K/V
vectors after attention projections (`MultiheadAttention.add_zero_attn`); its
logit is zero, so attention may allocate mass to no memory content. There is no
learned null embedding or routing regularizer. Workspace slot residual paths and
the common final-state output shortcut remain in every arm. Thus dropout does
not remove all final-state information.

Initial weights, ordered subsets, sampling and corruption RNG are matched.
Layer dropout uses a separate step-indexed RNG, reproducible under study resume.
Only selected training-prefix features are cached; the full frozen corpus and
ordered subset indices remain archived for auditing. All arms share one bounded
3300-second L40S job; incomplete arms must be reported as incomplete, not compared
as if they received equal training. Best checkpoints use validation NLL.

Interpretation: low training loss alone demonstrates fitting, not useful future
information. Train/validation gaps diagnose overfitting. Falling training loss at
the final checkpoint is evidence against claiming optimization convergence.
Small subsets cannot establish the final training-data requirement.

## Executed findings

[All 12 fits completed](../results/small-layer-controls.md), each at 3200 updates.
The baseline fits 96/768/3072 windows to 100%/98.9%/94.3% all-masked training-token
accuracy. Validation bottoms out earlier and then worsens. This rules out failure
to fit tiny data as the main explanation here; it does not establish that the
larger problem has enough training data or that optimization has fully converged.

On 3072 windows, best recorded validation NLL is 7.2641 baseline, 7.3332 dropout,
7.2625 null and 7.3135 both, all at update 400. The null option is essentially tied;
dropout does not show a clear benefit. This tests random 25% whole-layer dropout
and a fixed zero-key/value attention option, not learned layer selection or every
possible residual/gating design. One seed and an adaptively reused validation
slice do not justify ruling out those broader ideas.

The full suite used 1173 seconds across its four arms, plus orchestration, within
one bounded L40S job. 55 local tests pass, including exact resume with dropout.
The audit confirms matching ordered subsets, split hashes and supervision counts,
and full training-set metrics at every checkpoint. No new test metrics were read.
