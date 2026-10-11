# Experiment protocol

Read before modifying representations, datasets, training or evaluation.

## Scope (2026-10-11)

Frozen `Qwen/Qwen2.5-0.5B`, four target tokens, configurable 8/16-token blocks.
No collaborative proposer or from-scratch backbone. The identical denoising head
conditions on: one post-transformer residual state, a selected set of residual
states, one self-attention output (after output projection, before residual add),
a selected set of attention outputs, or a sparse J-lens vocabulary readout.
A parallel all-masked head is the context-only reference.

Layer indices are **one-based transformer blocks**, captured with forward hooks.
Post-block residuals are before the model's final RMSNorm, including layer 24.
The selected layer list must end at the final model layer. Selecting an earlier
layer for conditioning still executes the whole backbone; it is not a truncated
backbone speed claim. Every representation uses only already committed tokens.

## J-space provenance and adaptation

[Anthropic paper](https://transformer-circuits.pub/2026/workspace/index.html),
[arXiv](https://arxiv.org/abs/2607.15495), and
[reference code](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e)
are retained for reproducibility. `jspace.py` independently implements the
reference's batched Jacobian estimator: sum cotangents over valid future output
positions, average gradients over source positions, then average training-prefix
Jacobians. Exclude the first 16 and final positions. Matrices map raw post-block
residuals to raw final-block residuals; final layer's map is identity.

For conditioning, apply the transport, Qwen's RMSNorm and frozen unembedding;
keep the top 25 vocabulary scores, softmax within that set, and average their
frozen token output vectors. This is **compressed J-lens readout conditioning**,
not a full sparse nonnegative reconstruction of the paper's J-space frame. The
smoke calibration uses very few prompts and is unsuitable for scientific claims;
use at least 100 and test convergence before interpreting J-space quality.

## Data and fairness

Use WikiText official train/validation splits, group lines into articles, remove
exact held-out article duplicates from training, and window within articles.
No target tokens enter context extraction or Jacobian calibration. Model and
corpus commit hashes, article hashes and actual window counts are saved.
WikiText continuation is a plumbing/prediction task, not an answer-reasoning test.

All variants use the same sampled training windows, batch size and step budget;
conditioning ablations share architecture and trainable parameter count. The
loss samples a nonempty mask count uniformly and averages cross-entropy on only
masked positions per example. This is a simple masked denoising objective, **not**
the [BD3-LM](https://arxiv.org/abs/2503.09573) ELBO or its official reproduction.

Report held-out all-masked NLL/accuracy by target position, exact block accuracy,
paired per-example losses, parameter counts, exposures, elapsed time and memory.
All-masked NLL is a comparable conditional prediction metric, **not** normalized
sequence likelihood for the iterative sampler. Teacher-forced Qwen sees earlier
true target tokens and is labeled a distinct reference. Validation is used for
exploration; a separate untouched test split is needed for confirmatory claims.

Warm latency includes prefix encoding and committed-block cache updates. Do not
infer efficiency without matched output quality. Cached features accelerate
training; those savings do not imply faster generation. Repeat promising runs
across seeds and report paired differences before expanding the budget.

`selected_layers` optionally restricts the `multi`, `attention_multi` and `jspace`
conditions to a subset of extracted `layers`. Empty means all extracted layers.
Generation metrics also report greedy iterative block accuracy separately from
all-masked one-shot accuracy. Generation latency computes J-lens readouts only
for the J-space condition, not for raw-state comparisons.

## Pilot provenance and statistical limits

The initial larger pilot uses 4096 train windows from 292 WikiText-103 articles
and 256 validation windows from 22 articles. Calibration samples 100 training
window indices with a fixed independent seed and uses up to the first 128 prefix
tokens per window; it does not use target tokens. The selected multi-layer and
J-lens conditions use layers 6/12/18, excluding the final state as a stronger test
of intermediate information. The final-only comparator uses layer 24.

Reported per-example standard errors treat windows as independent and therefore
do not account for within-article correlation. Treat them as descriptive only.
Confirmatory inference needs article-cluster resampling, multiple seeds, and an
untouched test set after layer/configuration selection. The pilot is exploratory.

## Mixed-domain data gate (supersedes WikiText-only scope)

`corpus=mixed` trains one decoder on balanced prose, code and translation windows.
Sources: [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext),
[CodeParrot clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean),
and [WMT14 de-en](https://huggingface.co/datasets/wmt/wmt14). Data revisions and
window/group hashes are recorded. Code repositories map deterministically to
80/10/10 train/validation/test buckets. Official prose/translation partitions
are retained; cross-partition normalized exact duplicates are quarantined.
Translation losses cover German continuation tokens only, with masked left
padding and a bounded English-to-German prompt. Long prompts are truncated to
the context budget; this is not a full translation benchmark. At most four
windows per source record are collected; repositories can contain multiple files.

Validation controls checkpoint/layer/budget choices. `evaluate_test=false` is the
default, and data studies require it. Test tensors may be prepared and deduplicated
without evaluating their losses. See [data-budget protocol](data-budget.md).

## Active trained conditioning study

See [fixed protocol](conditioning-study.md). All 24 layers are extracted. The
current decoder ablations share a final-state residual shortcut; the information
probes explicitly disable it. Poor absolute quality does not block measuring
these relative conditioning comparisons, but limits their interpretation.


The trained conditioning comparison and its frozen test suite are now complete.
Historical reports correctly describe test as sealed at their creation time;
current status is **evaluated once after checkpoint selection**. This exposed
split cannot serve as a new untouched holdout for future adaptive experiments.
