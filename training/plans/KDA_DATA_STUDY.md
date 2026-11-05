# KDA coding model: data reassessment and bounded comparison

2026-11-05. Keep KDA as the preferred mixer and a final target just below 1B total
deployed parameters. This documents the user's requested study and reassesses the
proposal in [PR #143](https://github.com/koogle/lm-exploration/pull/143), which is
still open. It supersedes that proposal's mixture-study ordering and recommended
foundation weights; historical manifests and completed pilot provenance remain
unchanged. The authorized four-arm execution is specified at the end of this document;
its checked-in configurations are runnable after capacity-checked preparation.
Mixture improvements remain hypotheses until results arrive.

Read the [Ripple findings in PR #148](https://github.com/koogle/lm-exploration/pull/148) and
[reasoning training comparison](../references/reasoning-training-2026-11-04.md).

## Reevaluate the mixture

The old six-source proposal allocates 20% to Code Concepts and 5% to STEM-SFT.
That is 25% explicitly synthetic instructional exposure before demonstrating
transfer. Code Concepts' AST validity is not execution correctness; the earlier
512-row sample had 115 conservative benchmark-screen hits, requiring attribution
review rather than a claim of confirmed contamination. Source names also do not
establish exclusive content categories: audit ClimbMix's actual code/math overlap.

The revised starting hypothesis gives more room to real implementations and
coding prose, and introduces concentrated synthetic instruction later. This is
our hypothesis, not a reproduction of Ripple's undisclosed proportions.

All percentages below are post-filter, nonpadding training-token exposure, not
rows, bytes, or loss-weight percentages. Each column sums to 100%.

| Source | Original proposal | Recommended foundation candidate |
| --- | ---: | ---: |
| ClimbMix general text | 30% | 30% |
| Permissive GitHub code | 20% | 40% |
| Nemotron-CC-Code-v1 | 15% | 15% |
| FineMath-4+ | 10% | 5% |
| Nemotron-CC-Math-v1, released 3+ data | 0% | 10% |
| Nemotron Code Concepts | 20% | 0% |
| Nemotron STEM-SFT | 5% | 0% |

This proposes 55% explicitly code-oriented sources, 15% mathematical web text,
and 30% broad text. It tests code emphasis without assuming math automatically
transfers to coding. FineMath and CC-Math require cross-source near-deduplication;
15% math is provisional and should be reduced if it displaces better coding data.

CC-Code's 15% remains reserved and pending. Last recorded authenticated byte probe
(November 4) returned 403 awaiting author review; that is historical status, not a
fresh access check. Recheck before preparation. Do not substitute or silently
redistribute this slot. The separately named three-source control below can be
studied while access remains pending.

### Source choices and readiness

- Use the existing permissive GitHub source for a bounded comparison, with
  repository-level splits, deduplication, generated/minified/vendor filtering and
  a frozen language distribution. Audit useful implementation/test/doc coverage.
  Python is important for executable evaluation, but do not silently make the
  entire corpus Python-only. Publish the measured language proportions before launch.
- [Nemotron-Pretraining-Code-v3](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v3)
  is metadata (repository, path, commit, language), not ready-to-tokenize code.
  Treat it as a later source-quality replacement study after retrieval, license
  checks and capacity auditing. Do not credit its advertised tokens as ours.
- CC-Math access and a bounded sample were verified. Use the released 3+ source
  explicitly; a higher-quality subset requires actual score filtering and a new
  capacity measurement. FineMath-4+ is a separate source, not that filter.
- Code Concepts remains a midtraining candidate, with capped exposure and
  executable verification where feasible. Do not equate explanatory prose with
  verified problem solutions. STEM-SFT belongs in a later technical curriculum.
- Nemotron Math-v2 and SFT-Math-v4 are reasoning-stage candidates. Split shared
  seed problems across versions before sampling traces; favor complete tool-free
  traces that fit the current context. Preserve final answers and termination.

ClimbMix's research-only terms and record-level source licenses still apply.
A public metadata license does not override the licenses of retrieved code.

## Small study: separate data from architecture

Reuse the shared trainer, prepared shards, evaluation and supervised Modal
workflows. No new training loop or automatic launch is part of this document.

### 1. Data comparison, holding KDA fixed

Start with the approximately 128M hybrid configuration from PR #143, the same
frozen tokenizer and a qualified 2K context for all arms. The previous 512-token
plumbing check does not qualify this context. Freeze initialization, optimizer,
MTP setting, batch tokens, LR schedule and cooldown, data order rules and evaluators.

| Arm | ClimbMix | GitHub | CC-Math-3+ | Purpose |
| --- | ---: | ---: | ---: | --- |
| A | 70% | 20% | 10% | Fresh representative version of the pilot control |
| B | 40% | 50% | 10% | Isolate code quantity with the same sources |

Train each for 250M nonpadding tokens initially; compare at matched cooled
endpoints. The 2,048-row prefix pilot is insufficient: prepare representative
shards and verify per-source capacity first. One seed screens large effects only.
If results are close, spend the reserved repeat on a second paired seed instead
of declaring a winner. Extend both to 500M only within the remaining pilot cap,
with identical continuation/cooldown treatment.

Once all sources are qualified, compare the recommended foundation candidate to
B at the same budget. This is a combined source/mix comparison, not an isolated
math-quality or CC-Code effect. If access stays pending, keep the full candidate
pending and report only the three-source result.

To test synthetic timing later, branch the same foundation checkpoint for equal
continuation-token budgets: selected foundation mix versus 80% of that mix plus
15% Code Concepts and 5% STEM-SFT. This tests a late instructional curriculum;
it does not establish whether including synthetic data from the start is worse.
Do not add this arm before the basic data and architecture comparisons fit budget.

### 2. Mixer comparison on the selected accessible mixture

Compare 3:1 KDA/full attention against all-full attention using the existing
`HybridConfig.layer_pattern` support. Keep depth, width, tokenizer, FFN dimensions,
context, MTP and training settings fixed; report the resulting parameter and
FLOP differences. This is a mixer substitution, not an exact parameter match.
An all-full control is not an implementation of Ripple's sliding-window design.

Start both from scratch for 250M tokens. Reuse the selected KDA data-arm run if
its complete configuration matches, avoiding a redundant run. Record:

- Common held-out code, math and general-text BPB, plus short executable tasks.
- Actual nonpadding tokens/s, GPU-hours, dollars, peak memory and instability.
- Prefill and decode cost at 2K and 8K, clearly separating systems measurements
  from quality: a 2K-trained model has not demonstrated 8K competence.

Equal tokens diagnose learning efficiency; they do not imply equal compute.
Report quality-versus-dollar curves at shared cost points, labeling intermediate
uncool checkpoints. If a close decision depends on matched-cost quality, reserve
paired runs with predeclared equal-dollar budgets and their own cooldowns.
Do not compare one cooled final checkpoint against another's unfinished schedule.

KDA remains the default if evidence is inconclusive. A consistent disadvantage
requires a follow-up, not an automatic switch. Before freezing the main run,
measure both candidates near the intended 950–980M deployed size on actual
hardware; use a short throughput check and bounded learning confirmation if the
small-scale ranking drives an architecture change. A speed check alone cannot
establish target-scale quality.

## Evaluation and advancement

Freeze common held-out repositories, problem families and dated task windows
before preparation. Deduplicate across sources before splitting; audit near-duplicate
clusters and repository forks. Report source exposure/repetition and rejection
rates, and inspect screen hits rather than treating every 13-gram hit as leakage.
For target T and source weight w, require at least T*w usable source tokens for
single-pass training. Never fill a short source with unreported repeated epochs.

Use independent short executable tasks as the pilot capability signal; hard
LiveCodeBench may stay at zero at this scale. Report pass@1 at fixed prompts,
sampling and output budgets, paired task uncertainty, valid completion and loops.
Keep the final LCB window untouched. HumanEval cannot be the sole selection metric
for Code Concepts. BPB is reported by source on identical held-out documents.

Provisional advancement rule: prefer lower code BPB with no more than 2% relative
regression in general-text BPB and no clear executable-task regression. This 2%
threshold is a proposed retention guardrail, not a measured tolerance. Conflicting
or noisy metrics require a repeat or an explicit tradeoff, not a weighted score
chosen after seeing results. A pilot winner is a candidate for scale-up, not a
forecast of sub-1B frontier performance.

## Reasoning stages and spending

Keep pretraining, midtraining and reasoning SFT as separate measurable stages.
For the first reasoning SFT candidate, propose 70% verified code, 20% tool-free
math and 10% general/direct-answer assistant-token exposure. These are starting
weights conditional on data capacity and trace lengths, not an already prepared
recipe. Compare answer-only and reasoning supervision from the same base at
matched training budgets and fixed inference budgets. Implement the thinking
encoder/masks first: current non-thinking chat-SFT would discard the target behavior.

Start bounded RL/distillation only after SFT solves held-out tasks. Match task
curriculum to demonstrated success, reward executable outcomes and measure new-task
transfer. Long agent trajectories remain secondary after the recorded SFT failures.

Use the existing $1,000 preparation/pilot/evaluation envelope within the $10,000
plan, subtracting actual previous spend first. Provisionally allocate the remaining
pilot balance: 20% preparation/evaluation, 45% A/B, 20% attention control and 15%
repeat/confirmation. Additional full-mix or synthetic arms require remaining room;
reduce token targets equally or defer them rather than consume main-training funds.
Measure dollars/token before launch, persist explicit spending/token limits and
use existing independent remote deadlines under 24 hours with bounded calls and
zero retries. Training launches use the checked-in workflow configs and their bounded sessions.

## Authorized fast comparison — November 5

The user requested parallel tests. Use a four-arm factorial instead of waiting
for a mixture winner: KDA and all-full attention, each on A (70/20/10) and B
(40/50/10). All arms use the same sampled ClimbMix/GitHub/CC-Math corpus, tokenizer,
held-out documents and 2K context. Each targets 250,019,840 tokens (7,630 updates
of 32,768 tokens), with the same seed, 5M-token warmup and final 20% cooldown.
The first eight updates qualify checkpoint/resume, followed by up to two bounded
continuation sessions. Four separate workflow launches allow actual parallelism.

The KDA model has 127,631,028 training parameters; all-full has 127,558,452.
The small count difference is measured, not corrected by changing other dimensions.
Both retain the same MTP objective. Single-seed results can identify large effects;
a close result keeps KDA preferred and requires a paired repeat, not a claimed win.

The new sample manifest is `data/kda-study-sample-v1.json`: eight separated shards
for each of ClimbMix and code, four for CC-Math; seeded rows within shards (one
random row group per math file to bound I/O). Code splits by repository; all
sources use normalized exact-content deduplication and the benchmark screen.
This is broader than the prefix smoke sample, but repository forks, global
near-duplicates and generated/vendor-code quality remain limitations. Results
qualify a bounded directional comparison, not final-corpus readiness. Reusing
this sampler requires checksum-verified raw samples and the identical plan.

Each arm uses one H100, at most three calls of 8,100 seconds with at most 7,200
training seconds each, retries zero, and the independent workflow deadline under
24 hours. Total maximum GPU allocation is 27 H100-hours across the four arms;
qualification should use only a small fraction of its cap. Record actual CPU,
memory and GPU billing; reserve up to $300 of the existing pilot envelope for
this wave including evaluation, rather than treating GPU-only cost as total cost.
No automatic extension or additional matrix is authorized by these configs.

Configurations: `hybrid/configs/study-128m-{kda,full}-{control,codeheavy}.json`;
workflows have the same stems under `workflows/`. Prepared recipe hashes, capacity,
launch IDs and measurements are published in HF `jfrick/lm-exploration-results`
under `training/sub1b-pilot/kda-data-study-20261105/` once available.
CC-Code was rechecked on November 5 and still returned HTTP 403; its 15% proposed
foundation allocation remains pending, outside these explicitly named controls.

### Reproduction and interpretation

From `training/`, prepare the manifest through `training_data.audit_coding_mix`
with `--plan data/kda-study-sample-v1.json --out artifacts/kda-study-20261105-v2
--windows 100`. Use `--resume` only for an existing identical-plan sample whose
raw checksums pass. Each arm's prepared recipe uses that same volume folder.
Verify filtered source capacity against both recipes before uploading `prepared/`
to `/kda-study-20261105-v2` on `lm-exploration-hybrid-data`. Launch each of the four
`workflows/study-128m-*.json` files with a unique run ID using the existing
`workflows.modal_train::main` entry point. The managed environment's
`data/eu/modal_cli.py` wrapper preserves its TLS/proxy settings.

The shared tokenizer uses about 79.3M embedding parameters in these small models,
so most of the 128M count is vocabulary capacity. This proportion is much larger
than at the intended near-1B scale; do not treat the small mixer ranking as a
precise forecast. Hold tokenizer fixed here to isolate the requested comparisons.
