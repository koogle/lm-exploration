# Sub-1B coding reasoner: training plan

Updated November 10, 2026. Train a model from scratch through foundation
pretraining, midtraining, reasoning SFT and outcome-based RL. The working model
has 959,843,116 training parameters and a 927,589,920-parameter deployed trunk
once auxiliary multi-token prediction (MTP) is removed. Retain KDA, interleaved
with full attention at a 3:1 ratio: 24 layers, width 1,536, 4,096-token context
and the frozen GLM tokenizer with vocabulary 154,880.

The primary capability target is executable coding on held-out problems, with
LiveCodeBench evaluated on a frozen dated window. Sub-1B frontier performance
is an ambition, not a forecast from the small-model study. Agentic repository
work is a later goal: the earlier coding-trajectory experiments did not establish
successful transfer. See the repository's reasoning-training and negative-result
references before choosing SFT or RL curricula.

## Stages and advancement

1. **Foundation pretraining:** conditional 100B-token target using 50% ClimbMix,
   40% code and 10% Nemotron CC-Math score-3 data. The
   [mixture decision](KDA_DATA_STUDY.md) records the measured tradeoff between
   the three shares. Within the code share, markup, oversized and vendored files
   are filtered out, and curated projects and starred repositories are weighted
   above the rest of GitHub and repeated
   ([code sources](KDA_960M_PRETRAINING.md#quality-weighted-code-sources-recipe-v3)).
   The MTP loss is sampled at fraction 0.25.
2. **Midtraining:** introduce higher-quality coding explanations, verified
   solutions and longer-context tasks in bounded continuation comparisons.
   Code Concepts and STEM instructional data are candidates, not committed
   foundation shares. Keep a foundation-data replay component and measure
   retention; freeze weights after source and capacity audits. Compare
   continuing with and without short convolutions added at identity
   ([128M study](KDA_960M_PRETRAINING.md#architecture-options-from-the-marin-notes-128m-study)).
3. **Reasoning SFT:** train explicit reasoning followed by final answers, with
   assistant-token loss masks that preserve reasoning content. Compare against
   answer-only supervision from the same base, at matched training and inference
   budgets. A proposed starting distribution is 70% verified code, 20% tool-free
   math and 10% general/direct answers, conditional on usable trace capacity.
   Explicit reasoning tokens do not require a separate architectural reasoning
   head; MTP is an auxiliary prediction objective, not a reasoning head.
4. **Outcome-based RL:** start only after SFT solves held-out tasks often enough
   to supply useful rewards. Use executable tests and verified math outcomes,
   curriculum matched to actual success, and held-out transfer checks. Long
   agent trajectories remain secondary until shorter tasks work reliably.

Track per-source BPB, executable pass@1, completion validity and looping, using
fixed prompts, sampling and generation budgets. Keep held-out repositories and
problem families separate from training, and reserve the final benchmark window.
SFT or RL cannot be assumed to repair an undertrained foundation model.

## Budget and launch gate

The original approximately $10,000 remaining-credit envelope provisionally reserves
$1,000 for preparation/pilots/evaluation, $5,000 for foundation pretraining and
$4,000 for subsequent stages and contingency. Reconcile actual experiments and
billing before committing the remaining budget; these are allocations, not a
statement of the current account balance.

Use exactly eight 80GB H100s for target-scale qualification. Corpus capacity is
measured for every source of the selected recipe. Full pretraining is held
pending the code-mixture comparison, the precision choice and spending
reconciliation. MFU is not a launch gate: at the measured 437k tokens/sec a
100B-token run takes about 508 GPU-hours, and the configurations compared so far
rank differently by MFU and by tokens per second. Runtime implementation, hardware-counter work, kernels
and MFU findings are reviewed in the stacked implementation PR, separately from
this planning decision. See [foundation constraints](KDA_960M_PRETRAINING.md).
