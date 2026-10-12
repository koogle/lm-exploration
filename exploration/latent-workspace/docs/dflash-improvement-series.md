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

The first small-target teacher audit (`dflash-small-teacher-audit-01`) failed before
model loading because it assumed a separate config.json for a historical training
run. Read config from report.json instead; rerun `dflash-small-teacher-audit-02`
completed. Retain both $0.50 reservations. This diagnostic fix does not change
training or cached labels.

## Supervision exposure and next controlled data study

The original 20000-update runs used batch 3 and 8 anchors/response: only **3.36M
supervised NEW-token positions**. Update counts alone obscure how small this is.
The paper's nominal 800k responses × 6 epochs × 512 anchors × 15 NEW positions is
36.9B attempted label positions before sequence-length/masking effects; this is
arithmetic from its recipe, not a measured training-token count. It is not a
compute-equivalent comparison with our smaller target, shorter blocks or corpus.

Profile batch 48 on the same five-layer, two-feature, QK-normalized architecture
with a 300-update/96-response bounded smoke. This changes training throughput and
supervision, not inference architecture. Do not interpret the smoke as validation
of generalization or compare it causally to the earlier two-layer smoke.

After the three matched 20000-update architecture runs complete, choose the highest
selected validation NEW score among them. Freeze that architecture and checkpoint
for a matched data/exposure study. Both arms load exactly those draft weights,
start fresh AdamW/sampler/schedule, and use batch 48, 8 anchors, 20000 updates,
6e-4 LR, 4% warmup/cosine: **53.76M additional NEW-label exposures per arm**.
One uses the original 6144 training responses; the other uses 24576 responses,
requiring every original training example to remain present with unchanged labels.
Both share the unchanged validation cache. If the chosen architecture has five
feature depths, recache only the added planes on the expanded pool first.

Bound each worker to 3600 seconds/$5 and restore optimizer/RNG with one global
20000-update schedule when resuming. Initially allow at most eight worker
reservations ($40) for this pair; review progress before extending. The global
$500 cap remains binding. Full-fitting and conditioning audits distinguish remaining
optimization failure from memorization/generalization failure. No winner or useful
quality is claimed until measured, and sealed test stays untouched until the gate.

Batch-48 smoke completed in 115 seconds, all three verified rollouts match. At
300 updates (806400 NEW-label exposures), the 12-response training diagnostic
reaches NLL 0.239 / 6.979 consecutive matches; validation NLL 9.481 / 0.041.
This is tiny-set memorization, not useful generalization. A full 96-response audit
is running; do not treat diagnostic training samples as the whole corpus.

Full tiny-set audit completed: final step 300 averages NLL **0.29685 / 6.86195 NEW
matches** across all 96 training responses, versus validation **9.48065 / 0.04109**.
The earlier 6.979 training figure was the 12-response diagnostic, not the full set.
The validation-selected step 100 scores train 1.75404 / validation 0.08608; zeroing
its context memory reduces validation to 0.05086. These are memorization and weak
held-out-conditioning results, not a useful drafter. Raw evidence:
`results/dflash-batch48-fit-audit-01.json`.

Expanded pool completed in 782.6 seconds: 26163 usable training responses
(8700 prose, 8764 code, 8699 translation), with 1,453,823 distinct response-token
positions across that full pool. This supports the planned balanced 24576 subset.
All 192 validation examples remain cached unchanged; retained validation token+
feature SHA-256 is `0336c72f43d0480a6c0ed867c691f5b7f67311ab5b6e0f56923bc32dd0971bc8`.
The preparation retained 6730 existing train+validation examples and did not
regenerate their labels. Raw report: `results/dflash-expanded-data-01.json`.

## Completed fidelity selection

All three 20000-update runs completed and pass 12 verified FP32 rollout checks.
Selected validation NEW scores: QK-normalized two-layer 0.484354; five draft layers
with two features 0.519650; five draft layers with five features 0.530051. Choose
`dflash-fidelity-features-01`, step 18000, 78,578,432 trainable draft parameters,
by the predeclared rule. The last increment is small: paired conditional bootstrap
95% interval for five versus two feature depths is [-0.02018, 0.04234] NEW tokens.
It does not establish a reliable benefit from more feature depths. All remain
below the useful-quality gate. See `results/dflash-fidelity-comparison.json` and
`results/dflash-feature-depth-comparison.json` for uncertainty limitations.

Expanded five-depth recache completed in 154.2 seconds with old feature planes
preserved, 26163 train and 192 validation responses, and no label regeneration.
The next two configs share this pool and the same selected warm-start checkpoint.
Both require the original 6144 training groups with exactly matching token labels.
A 20-update GPU warm-start smoke precedes their launch. Selection/config hashes are
archived in `results/dflash-exposure-selection.json` before the new fits.

## Conditional next-pool preparation

At the matched 4500-update exposure checkpoint, 24576 responses score 0.8594 NEW
versus 0.4830 for 6144. The higher-supervision runs remain incomplete. Prepare a
54000-prompt extension to support a balanced 49152-response next stage if needed,
without yet selecting or launching that training. Reuse the complete five-depth
pool's existing examples, labels, features and held-out prompts. New data uses the
same sources, split/group policy and target. This changes neither active fit.
A bounded preparation worker reserves $5; long-worker host RAM increases to 128 GiB
for expanded caches. GPU, source package and hard timeout remain unchanged.
No training-source changes are made while the matched pair is running.


Both high-supervision arms completed their declared 20000 updates, with the small
arm resuming its final 898 updates under the unchanged schedule/source. Selected
validation NEW matches are 0.57378 versus 1.02179; conditional paired interval for
the gain is [0.37974, 0.51880]. All twelve rollout checks pass for both. Full-fitting
audits are running; the conditional next pool is complete but no fit on it is
selected. Test remains sealed and the quality gate remains unmet.


## Next controlled expansion, 2026-10-12

The selected 24576-response head's full-training audit scores 3.19508 NEW / 7,
versus validation 1.02179; zero-memory validation is 0.03785. Its saved checkpoint
hash matches the local download, and validation metrics match the training report
exactly. The full smaller-corpus audit confirms final memorization (6.999959 train
versus 0.44514 validation). The larger final-checkpoint audit is still running;
both selected checkpoints have been audited before choosing the next initialization.

Compare **24576 versus 49152 responses** from the prepared five-depth 54k pool.
Both arms start from `dflash-exposure-24576-01/best.pt` (step 10000), use fresh
AdamW/sampler and the same batch48/eight anchors/LR6e-4/4% warmup/20000-update cosine
recipe: 53.76M additional NEW labels per arm. All prior 24576 training groups and
labels must be retained. A required-subset sort may reorder old responses; compare
canonical group/token identities, not just historical order. Both arms share the
same validation tensors and initial weights. No architecture, target, source-domain,
block-size or inference-speed changes. Test stays sealed.

This matched pair isolates added training responses within the new stage. Comparison
with the previous stage also changes warm-start/optimizer history and is not a
one-factor data claim. Initially allow up to four bounded $5 workers for this pair,
resuming only remaining updates with the same optimizer/RNG/global schedule.
Preserve the existing $500 cap and all reservations. Freeze configuration/checkpoint
provenance in `results/dflash-expansion-selection.json` before launch. Keep package
source unchanged while this pair runs. Continue to report failed quality gates.


## Read-only source-capacity audit

The pinned WikiText source contains 29444 assembled articles, 29002 normalized
unique articles, and 23090 groups assigned to training under the existing hash
split. The current one-prompt-per-group rule therefore permits at most **69270
balanced training responses** across three domains, before historical holdout,
length, near-duplicate, short-response and duplicate-response filters. A simple
98304-response extension of this same recipe cannot succeed.

This was a CPU-only streaming count using `data.documents`, `corpus.digest`, and
`repository_split('prose:' + digest(article))` over the pinned official train split.
No target generation, test labels, or test metrics were accessed. Evidence:
`results/dflash-prose-source-capacity.json`. Current fits remain unchanged.

If completed results justify further expansion, first choose and validate either
additional prose sources or multiple distinct prefixes from training-only articles.
Keep the original article/repository/source-sentence split groups separate from
individual example identity, quarantine all existing held-out groups, preserve old
teacher labels, and account for correlated examples. Do not relabel extra prefixes
as new independent documents. Larger feature-cache storage/loading may also need
attention; these are unimplemented next-stage options, not chosen experiments.

Prototype primitives now exist on the isolated, unmerged `drafter-data-coverage`
branch (commit `2ccf058`): deterministic training-only prefixes, opt-in registry
checks, and private mapped teacher-cache reads. 98 CPU tests pass; preparation
integration and GPU validation remain unexecuted. Active fits retain main's source.
See that branch's `docs/dflash-prefix-coverage.md`; this is not a new training claim.
