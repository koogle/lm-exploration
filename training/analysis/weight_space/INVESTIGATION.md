# Investigation before further weight changes

The expanded investigation does not qualify either edit as a reliable agent
repair. The first substantially reduces repeated tool calls but frequently moves
repetition inside a truncated reply. The second remains close to the original,
with no solved tasks and uncertain behavioral differences. All three checkpoints
were held fixed; this follow-up changes only diagnostic scripts and documentation.

## Completed live comparison

The planned 48 tasks × three seeds × three models produces 432 episode records:
431 reach hidden-test grading without unexpected environment errors, and one
baseline episode is operationally censored as described below. The matched
comparison excludes that task/seed from every model, leaving 143 pairs per model.
These are existing benchmark tasks, not a fresh external test set.

| Matched outcome | Original | First edit | Second edit |
| --- | ---: | ---: | ---: |
| Hidden-test successes | 0/143 | 1/143 | 0/143 |
| Episodes with a patch | 20 | 34 | 21 |
| Mean per-episode repeated-call fraction | 63.76% | 30.23% | 67.76% |
| Repeats pooled across executed calls | 76.18% | 50.21% | 77.86% |
| Ended on a truncated reply | 26 | 78 | 19 |
| Exhausted context | 19 | 15 | 21 |
| Reached 100 turns | 98 | 49 | 102 |
| Explicit submissions | 0 | 0 | 0 |

Paired bootstrap intervals resample whole tasks, retaining each task's available
seeds. The first edit changes the mean repeated-call fraction by −33.53 percentage
points (95% interval −40.92 to −26.14), but increases truncated endings by 36.36
points (+25.17 to +47.52). Of its 78 truncated generations, 67 have at least half
their token eight-grams repeated; their median repeated-eight-gram fraction is
93.43%. These are predominantly repetitive expansions, not merely long useful
patches. The first edit generates fewer tokens and runs fewer turns because so
many episodes end in this state; that is not successful completion.

The second edit changes the repeated-call fraction by +4.00 points (−3.40 to
+11.74) and truncated endings by −4.90 points (−13.29 to +3.47). Its larger-sample
behavioral differences from the original are uncertain. The earlier 16-task
regression does not establish a strict regression on every metric, and the larger
study still gives no evidence of a useful agent repair.

The first edit passes two episodes over all 144 runs, both on one attribution
task, with one success in the censored pair. Only the other success enters the
matched comparison. Its paired success-rate difference is +0.70 points, with a
task-bootstrap interval spanning zero (0 to +2.13). Rare outcomes make this
interval descriptive, not strong evidence of improvement. The second edit and
original have no successes in the matched sample; the resulting degenerate
empirical bootstrap is not evidence that their true capabilities are equal.

More patches do not translate into broader test progress: 3 / 2 / 1 matched
episodes pass any fail-to-pass test, and 18 / 17 / 11 regress at least one
pass-to-pass test, respectively. All such regressions have nonempty patches.
Of the first edit's 34 matched patches, 31 touch a file changed by the reference
solution. This is a post-hoc localization diagnostic, not a correctness criterion;
hidden reference information is never provided to the agent or used for selection.

The investigation also completes 1,134 frozen-context generations, 21 captured-
prefix continuations, 576 teacher-forced measurements on 192 shared training
spans, and a 4,364-trajectory stored-data audit. All 1,375 tracked live sandboxes,
including excluded duplicate work, are stopped. The remaining live supervisor
and all diagnostic supervisors complete; the initial live supervisor is cancelled
after explicit cleanup of its stopped workers and children. Checkpoint hashes
match before and after completed model evaluations.

## Likely problems and how to distinguish them

The evidence points to several interacting problems. It does not identify one
defective layer, prove a training-data cause, or establish that model size alone
is the limit. This checkpoint is full-parameter SFT of Qwen3-0.6B-Base; reward
hacking during RL is not an explanation for this run.

| Problem | Evidence and confidence | Discriminating next check |
| --- | --- | --- |
| Repeated history encourages more of the same action | Strong evidence of history sensitivity: the original repeats 107/144 next commands with full history, versus 32/144 with four recent exchanges. Deduplication helps; clipping tool output barely does. This does not isolate repeated content from context length or prove useful progress. | Execute a paired recovery/history trial on the frozen original and grade the resulting patches. Preserve useful observations and test both kinds of repetition. |
| The weight-edit objective rewards the wrong behavior | Demonstrated limitation: fewer repeated executed calls coexist with 78 truncated endings, 67 containing heavy internal repetition. Selection on fixed histories misses failures on the edited model's own histories. The activation contrast also mixes repetition with trajectory stage and context. | Require task success and successful completion, alongside healthy editing and useful repeated-test controls, before selecting another edit. |
| The harness turns a malformed reply into a terminal failure | Confirmed mechanism: a reply without a parsed tool call ends the episode, including a generation cut off at 1,536 tokens. A longer budget completed only 3/21 captured-prefix continuations, all from one patch attempt; it mostly prolonged repetition. | Measure a bounded format-error recovery policy separately from any weight intervention. Count failures and retries explicitly. |
| Correctness, planning and completion are weak beyond the looping symptom | Strong behavioral evidence: 31/34 matched first-edit patches touch a reference-solution file, but only one passes all hidden tests. The two successes across all seeds belong to one task and end through context exhaustion or truncation. Removing a loop cannot by itself supply missing task understanding. | Evaluate whether recovery produces correct patches, then compare model sizes under the same recipe and budget. The existing unfinished 1.7B control cannot settle capacity versus recipe. |
| The SFT mixture may teach weak action policies or insufficient recovery | Plausible contributor, not established cause: at least 78.03% of stored mini-coder trajectories are unverified, and the mixture combines converted Open-SWE and terminal formats. The mini-coder converter rejects malformed/timeout trajectories, so this source does not directly teach recovery from those states. The stronger published 1.7B mini-coder reference also uses unfiltered rollouts, which argues against treating verification status as the sole explanation. | Hold model size, token budget and optimizer fixed while testing data quality, format and recovery examples separately. |

The audit found no obvious mask, label-shift, tokenizer-vocabulary or stop-token
defect. Small teacher-forced loss changes also argue against broad corruption of
learned demonstrations. These negative checks narrow the diagnosis; they do not
rule out every training or inference defect.

The latest [PR #130 corpus audit](../../plans/HYBRID_V2.md) adds an important
counterweight to the data-quality hypothesis: across the full published mix,
mini-coder repeats 10.4% of calls, Open-SWE 2.5% and PR-GLM 0.7%, compared with
about 75% for the original model's archived evaluation. Our sampled corpus audit
agrees with those rates. This weakens a simple explanation of copying frequent
training loops and supports investigating errors that compound on the model's
own histories. It does not establish why those errors arise or whether a
particular quality filter would improve the resulting policy.

The investigation itself has limits: only 48 tasks from eight already examined
repositories, three seeds, one operational censor, and occasional inference
nondeterminism despite identical seeds and prompts. The only successful task has
one censored comparison. Some diagnostics and the stricter pipeline-repetition
metric were added after observing failures, and behavioral detectors are
heuristics. The context interventions are next-action probes, while retention
uses training examples. Raw outputs and launch records remain in the local and
Modal artifact stores described below; they have not been published to HF.
These constraints prevent treating this study as an independent benchmark or a
validated repair.

## Protocol fixed before new results

The live comparison uses 48 tasks: six from each of the eight repositories in the
existing 179-task evaluation, selected by a fixed task hash. Each existing model
runs all tasks with three sampling seeds, allowing 100 turns and 1,536 output
tokens per turn. This is 432 episodes and at most 43,200 turns. Tasks run in
batches of 16 to bound sandbox concurrency. Each request gets a seed derived from
the run seed, task ID and turn number, so changes in active batch size do not
reassign the random seeds. Hardware floating-point and batching effects can still
affect sampling. These are repeated diagnostics on an existing benchmark, not a
fresh external test set. The earlier experiments' calibration contexts are not
used to choose a new edit.

Primary outcomes are resolved tasks, hidden-test results, applied patches and
submission. Behavioral outcomes include exact and near repeated commands,
repetition within a command, first attempted modification, command failures,
truncated replies, context exhaustion and length of unproductive sequences.
Episode-level comparisons and paired task bootstrap intervals accompany pooled
command fractions; shorter failed episodes cannot count as successful anti-looping.
Raw model replies are preserved with task, turn, sampling seed and finish reason.

The accompanying investigation audits all available archived original episodes,
the training data converters, loss masking, stop tokens, context handling and
tokenization. Frozen-context diagnostic experiments test shorter histories,
shorter tool output, and larger output budgets. They measure sensitivity and
failure mechanisms, not deployment-ready fixes. Long patch-writing outputs are
distinguished from repetitive pipelines before attributing truncation to looping.

Every new live call has a three-hour execution timeout and retries disabled.
An independent supervisor enforces a three-hour-fifteen-minute launch deadline,
including startup, and explicitly terminates all recorded child sandboxes.
Partial results are committed after each 16-task batch. Checkpoint SHA256 values
are verified before and after evaluation. Existing runners and original model
files are preserved.

Artifacts are under ignored `artifacts/weight-space/investigation/` locally and
`weight-space/qwen06-investigate-20261103-01/` on the runs Modal volume. Live
trajectories also use the existing data volume's `pr-tasks-pilot/agentic/` folder.

### Scheduling amendment during the live study

The original checkpoint repeatedly runs `timeout 900 python -m pytest tests/unit
-q -p no:cacheprovider 2>&1 | tail -8` in seed 11, batch 1. A read-only process
inspection confirms the test process is running. Repeated three-minute tool
timeouts delay the entire lockstep batch; this is not an inference-engine stall.
The seven remaining original-model batches therefore get independent workers,
with exactly the same task groups, per-request seeds, settings and original
launch deadline. The original worker and candidate workers continue unchanged.
For these seven batches, the new launch is the canonical source regardless of
outcome; the initial launch supplies seed 11 batches 0 and 1. Any duplicate work
is excluded. `remaining_modal.py` records this allocation before results arrive.
Unfinished episodes will be reported as incomplete, not scored as failures.

One operational censor was then necessary: the original model's seed-11
`pyntpot` attribution episode generated the identical 900-second pytest command
11 times, repeatedly hitting the harness's 180-second tool limit. At generated
turn 19, its patch, raw replies and filesystem snapshot were preserved and its
execution sandbox was terminated. This allows the other 15 episodes to proceed.
That task/seed is excluded from **all three models' paired comparisons**; any
subsequent replies to the terminated sandbox are excluded too. The planned
cohort remains 432 attempts, with at most 431 graded episodes and 143 comparable
task/seed pairs per model. This is an operational protocol deviation, not a
model failure or a successful completion. It is especially material because the
first edit passes the hidden test on this excluded seed; its result is reported
descriptively but cannot count as a paired win.

One completed duplicate baseline batch is excluded under the predeclared
allocation. Comparing its raw replies with the designated run finds 307 requests
with identical prompt-token hashes and sampling seeds; 13 completions differ.
All 16 first-turn completions match. This directly confirms that per-request
seeding does not make these vLLM runs bitwise reproducible across scheduling and
cache histories. We retain the duplicate as a reproducibility diagnostic and
do not choose between runs by outcome. The original worker is stopped once its
two designated batches and both edited-model arms have completed, and all its
remaining child sandboxes are explicitly cleaned up.

## Completed mechanism checks

### Archived trajectories: repetition grows across turns

All 179 archived original-model episodes were audited, including 14,376 bash
calls. None resolved a task. Exact repeated commands account for 10,751 calls
(74.78%). The fraction rises from 1.84% in turns 1–10 to 29.36% in turns 11–20,
78.96% in turns 21–40, 96.60% in turns 41–60 and 99.19% in turns 61–100. Late
bins include only episodes still active, so this is not a within-episode causal
estimate. Still, extending the rollout primarily adds repeated commands.

There are 9,904 exact repeats with identical returned text and no intervening
detected write. That write detector is a shell-text heuristic and can miss edits
or mistake a search for an edit; final patches and hidden tests are the measures
of actual changes. The archive contains 28 nonempty patches. Five episodes pass
at least one fail-to-pass test; one passes all fail-to-pass tests but regresses a
pass-to-pass test. There is some useful partial work despite zero full resolutions.

Only one result reports a missing executable. Missing paths occur 669 times;
4,847 calls return empty successful output. Missing search utilities are therefore
not a general explanation of the loops. The duplicate-command detector also
checks repeated pipeline fragments, so appending another identical `grep -v`
does not count as a distinct action merely because the full string changed.

### Context interventions on frozen original weights

Forty-eight stuck histories are selected from the archive, each evaluated with
three seeds. The system prompt and initial task are preserved. Novelty is always
scored against commands in the full original history, including commands hidden
by a context intervention. These are next-action probes, not executed episodes.

| Original-model context | Valid calls / 144 | Exact repeats | Distinct calls | Truncated replies |
| --- | ---: | ---: | ---: | ---: |
| Full history | 142 | 107 | 13 | 2 |
| Latest four assistant/result groups | 144 | 32 | 52 | 0 |
| Long tool output clipped to about 1,000 characters | 144 | 103 | 12 | 0 |
| Identical assistant/result groups deduplicated | 141 | 71 | 26 | 3 |

Keeping four groups increases distinct calls by 27.08 percentage points (95%
task-cluster bootstrap interval 15.28–38.89). Deduplication adds 9.03 points
(2.08–16.67). Clipping tool output changes this by −0.69 points (−6.25–4.86).
History matters more in this sample than raw tool-output length. The shorter
histories still produce mostly search/read commands, and possible-write markers
fall from 20 to 11. This is evidence of history sensitivity, not a demonstrated
task-solving fix or a recommendation to discard useful context.

The distinct-call metric here also rejects duplicated single-pipe segments. This
post-generation diagnostic refinement catches repetitions the earlier separator
heuristic missed; the earlier scores remain in the artifacts. Exact-repeat
counts are unaffected. These remain exploratory behavioral metrics.

On the same full histories, the first edit produces 41 exact repeats, 30 distinct
calls and five truncated replies; the second produces 110 repeats, 11 distinct
calls and two truncated replies. All three models produce valid calls on all 48
first-turn controls. These probes reinforce the original warning: a score on
fixed histories does not measure behavior on a model's own future histories.

### Truncation comprises both repetitive expansion and unfinished patches

Seven previously captured terminal histories fit a 4,096-token output budget.
The 1,536-versus-4,096 comparisons use three seeds and all three frozen models.
Longer outputs do not recover any of the nine truncated cases whose initial token
prefixes match across budgets. Other prefixes differ because resampling in
different inference batches is not bitwise identical; they cannot support that
paired continuation claim.

A separate check conditions on the actual captured unfinished reply text, then
adds up to 2,560 tokens using its original checkpoint. Of 21 continuations, 18
again hit the limit. The three completed calls are three seeds of one long
patch-writing attempt; their shell and embedded Python syntax parse. They have
not been executed or graded, so closure does not establish a usable patch.
Repetitive search outputs continue expanding. This follow-up uses captured text,
not the exact original token stream, which the earlier live recorder did not save.
One further terminal history cannot fit this larger budget and is excluded.

Thus increasing the budget can help an unfinished patch, but does not generally
repair the repetitive outputs. Treating every truncated call as intentional
stopping, or every truncation as the same kind of loop, would misdiagnose it.

### Stored training data and retention

The checkpoint is full-parameter SFT, not an RL checkpoint. Its actual stored
corpus was audited using memory-mapped tokens, masks and offsets: 4,364 sampled
trajectories, 96,091 assistant spans, with up to 1,000 examples per source. Masks
are binary; every sampled assistant span ends in the expected end-of-message
token; no sampled supervision includes a message-start token or an unparseable
tool-call span. The training and inference tokenizer vocabularies match. The
source code's causal label shift, target normalization and saved stop tokens
also agree. These checks find no obvious masking or EOS configuration defect;
they do not exhaustively validate every corpus row.

The mini-coder recipe sorts verified trajectories first but includes unverified
ones when its count is unrestricted. All 60 cached metadata shards were read:
68,313 of 316,306 eligible raw candidates are verified (21.6%). The stored source
contains 310,878 trajectories, so at least 242,565, or 78.03%, are unverified.
This is a bound because post-conversion stored membership has not been mapped
back to raw metadata. “Unverified” does not establish that a trajectory failed.
It is a substantial data-quality hypothesis that needs a controlled training
comparison; the audit alone does not prove that these rows caused looping.
The existing PR already reports a stronger 1.7B mini-coder reference trained on
unfiltered rollouts (11/179 tasks solved at 40 turns), alongside weaker results
for the converted mixture. Its different model size, format and training recipe
prevent attributing our result to verification status alone. The unfinished
same-mixture 1.7B control remains relevant to separating capacity from recipe.

Exact repeated bash calls occur in 2.54% of sampled OpenSWE commands and 10.80%
of sampled mini-coder commands. Extremely long repeated `grep -v` pipelines are
rare in this sample (one mini-coder command, none in OpenSWE), despite being
prominent generated failures. Assistant spans longer than 1,536 tokens also
occur in training, including legitimate edits. The inference output cap can
therefore truncate behavior the model was trained to produce.

A separate teacher-forced retention diagnostic uses one fixed eligible assistant
span from each of 192 training trajectories: 64 each from the resolved OpenSWE,
resolved PR-GLM and reward-one Terminal-Lego sources. Full preceding context is
retained up to 16,384 tokens; target spans contain 32–1,536 tokens. The three
models predict exactly the same 40,160 target tokens on one GPU with the same
implementation, and their checkpoint hashes match before and after.

| Frozen model | Token-weighted loss, nats/token | Mean per-example loss change, 95% bootstrap interval |
| --- | ---: | ---: |
| Original | 0.39574 | reference |
| First edit | 0.39758 | +0.00191 [0.00075, 0.00303] |
| Second edit | 0.39612 | +0.00032 [−0.00024, 0.00086] |

The first edit has a small measurable retention cost; this is not catastrophic
corruption. The second is close to the original on this diagnostic. These are
training examples selected for successful source trajectories, not an estimate
of held-out generalization. Small teacher-forced loss changes can coexist with
large long-rollout behavior changes, so this diagnostic cannot qualify an edit.

### Correct patches and completion are separate outcomes

The first edit's seed-11 attribution task in `findlaywebb/pyntpot` passes all
hidden tests. It makes the one-line fix on turn 4, removing the label-display
condition from attribution drawing, then continues until context exhaustion on
turn 84. It does not submit. On seed 29 it instead adds an explicit early return
when labels are disabled, preserving the reported bug; existing visible tests
pass, but the hidden regression test does not. This contrast demonstrates both
some underlying patch capability and unreliable task comprehension/completion.
It does not establish a statistically reliable improvement from one solved case.
Seed 47 also passes this task's hidden tests, but then ends on turn 5 in a
truncated Python script repeating the same method definition. Thus the positive
result replicates on two seeds of one task; neither run finishes cleanly.

## Relation to the motivating article

[Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
studies Qwen3.5-0.8B, a different hybrid architecture, with long reasoning outputs
and different sampling settings. Our dominant failure is repeated shell actions
across tool turns, plus some repetitive expansion inside a single call. The
article's layer numbers and successful math intervention do not establish a
transferable agent repair. Its negative small-Gemma result also matters: removing
repetition cannot supply reasoning ability that the underlying model lacks.

## Decision and next experiment

Keep the original checkpoint as the reference. Neither sparse edit meets a
task-completion gate, and the first edit's large truncation increase disqualifies
its repeated-call score as a repair criterion. The measured distinction is
between cross-turn command repetition, repetitive expansion inside replies,
patch correctness and recognizing completion; no single novelty score covers
all four.

The next justified experiment is a bounded live comparison on the frozen original
of explicit recovery/history handling after unchanged repeated commands or a
truncated reply. Context probes motivate testing it; they do not justify deploying
recent-four truncation. Use hidden-test success as the primary outcome and record
both kinds of repetition, malformed replies and completion. The actual harness
currently ends on a reply without a parsed call; a format-error recovery trial
should be separately measured, not silently folded into a weight comparison.
No recovery policy has been implemented in the default harness here.

Further weight work needs controls for healthy edits, useful repeated tests and
nonlooping long histories. The original activation contrast confounds trajectory
stage and context with repetition, and successful single-next-action probes did
not predict the edited model's own long histories. The unverified-data share and
mixed formats warrant controlled same-model training experiments, but neither
is an established cause from these measurements.

## Reproduce and inspect

Run from `training/`:

```sh
python analysis/weight_space/investigate_modal.py \
  --config artifacts/weight-space/investigation/live-config.json \
  --output artifacts/weight-space/investigation
python analysis/weight_space/fetch_investigation.py \
  --directory artifacts/weight-space/investigation
python -m analysis.weight_space.summarize_investigation \
  --directory artifacts/weight-space/investigation --complete
python -m analysis.weight_space.summarize_probes \
  --directory artifacts/weight-space/investigation
python analysis/weight_space/audit_truncation.py \
  --directory artifacts/weight-space/investigation \
  --tokenizer artifacts/weight-space/checkpoint/model
```

Launch configurations, source hashes, selected tasks, raw generations, grading
results, censor records and cleanup records are preserved in the ignored artifact
directory and on the existing Modal volumes. Use a fresh experiment identifier
for new work. Generated artifacts have not been published to HF. Local validation
covers repeated pipeline detection, history deduplication that retains changed
observations, and paired estimates with a censored seed; formatting and lint
checks pass. The completed supervisors and before/after model hashes are recorded
with the run artifacts.
