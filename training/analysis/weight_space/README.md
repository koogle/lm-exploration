# Sparse weight edits for the Qwen agent's repeated commands

**Result after two rounds: no reliable repair.** The first edit reduces repetition
but increases truncated tool-call endings. The second protects call validity but
repeats more than the original in the second round's 16-task check. Every model
resolves zero tasks in those initial small live checks.
Keep the original as the reference model; both edited checkpoints are experimental.

The subsequent [longer investigation](INVESTIGATION.md) holds all three checkpoints
fixed: 431 graded live episodes, 1,155 generation probes and a stored-data audit.
On 143 matched task/seed pairs, original / first edit / second edit resolve
0 / 1 / 0 and end on truncated replies 26 / 78 / 19 times. The first edit often
moves repetition inside an unfinished reply; the second has no demonstrated
task-completion benefit. One censored baseline case is excluded from every paired
arm. Full outcomes, uncertainty and the retained positive and negative cases are
reported separately from these first two edit rounds.

This study investigates `qwen3-0.6b-base-scale-500k-02` from
[PR #130](https://github.com/koogle/lm-exploration/pull/130). It is full-parameter
SFT of Qwen3-0.6B-Base on 493,711 agent trajectories, rather than an RL checkpoint.
The archived 179-task evaluation resolved no tasks and repeated about 75% of its
tool calls. The original model is preserved.

The inspiration is [Weight-Space Programming, experiment 8](https://rikiyat.github.io/blog/weight-space-programming/).
That experiment edits MLP output directions to reduce reasoning loops in a
different architecture, Qwen3.5-0.8B. Its layer numbers are not a prescription for
this model. Our problem is mostly repeated shell searches across turns, which
requires separate evidence from repeated text within one answer.

## Protocol

Run the scripts from `training/`. `prepare.py` takes the downloaded evaluation
trajectories and tokenizer as explicit inputs. It selects the first repeated
command after at least three assistant turns, with a context of at most 16,000
tokens and a command length between 6 and 149 tokens. Each task contributes one
pair: the context of that repeat and the context of the command's first occurrence.
Both are rendered with the same Open-SWE/Qwen chat code as the original evaluation.
Later contexts are longer and occur at a different trajectory stage. The activation
contrast therefore does not establish a uniquely identified repetition circuit;
the behavioral comparison tests the intervention, not that interpretation.

There are 118 eligible tasks. Sorting task IDs by SHA256 fixes 24 calibration tasks
and 94 test tasks. These are disjoint tasks, not disjoint repositories. The source
is the existing benchmark; this is a diagnostic split, not a new external benchmark.
Repeated commands are a behavioral proxy: rerunning a test after an edit can be useful.

The first 16 calibration pairs provide a mean activation contrast at matched command
positions for each MLP. For each layer, `d` is the normalized MLP output contrast,
and `J` contains the 64 columns contributing most to that contrast. A trial edits:

```text
W[:, J] := W[:, J] - alpha * d[:, None] * (d @ W[:, J])[None, :]
```

At `alpha=1` this removes the projection; larger values reverse and amplify the
component. No gradients, optimizer steps or retraining are used. The remaining
eight calibration pairs measure changes in the repeated command's likelihood
against its first-use likelihood. Every trial restores the original weight bytes.

The scan covers all 28 MLPs. Layers 14 and 17 have the strongest selective effects,
though the likelihood differences are small. Generation trials cover these two
layers, the four leading layers, and strengths 1, 3 and 4. An additional probe ranks
attention heads by direct target-logit contribution and causally ablates the top
12 candidates; the best two-head generation trial does not materially help.
Layer and head indices are zero-based.

Generation uses bf16 Transformers/SDPA on one H100, temperature 0.7, top-p 0.8,
top-k 20, repetition penalty 1, and the original stop tokens 151645 and 151643.
Calibration uses seed 2026; the frozen test uses seed 2027. Both comparison arms
have a 512-token output cap. This cap is shorter than the original live agent's
1,536 tokens. These tests generate one next action from frozen histories; they do
not execute the command or measure completed bug fixes.

## Metrics and selection

An exact repeated command is byte-identical to an earlier command in its history.
Initial inspection showed that this metric alone rewards slightly altered searches
and multiple copies of one search inside a single call. Before opening the test
outputs, we added two conservative heuristics:

- Near repeat: maximum character-sequence similarity to an earlier command is at
  least 0.85 (`SequenceMatcher`, `autojunk=False`).
- Internal repeat: duplicate shell segments separated by `;`, `&&` or `||`, using
  shell tokenization and requiring at least three tokens per segment.

`valid_distinct` requires one valid bash tool call, neither a near repeat nor an
internal repeat. This is still not semantic correctness or proof of useful progress.
The metrics also record invalid calls, token-limit truncation and premature submission.

On the 24 calibration tasks, the original makes 12 exact repeats and 6 valid,
distinct commands. Editing layers 14 and 17 at strength 4 makes 2 exact repeats
and 12 valid, distinct commands, with one invalid truncated response. This candidate
was frozen before test generation. Its local export changes 130,750 of 596,049,920
parameters (0.02194%); only the two selected MLP tensors differ. Configuration,
tokenizer and generation files in the CPU reconstruction are byte-identical to the
original. The canonical evaluated checkpoint is downloaded separately as
`artifacts/weight-space/validated-candidate/`. GPU-versus-CPU float32 arithmetic
changes the bf16 rounding of 23 edited scalars, so the CPU reconstruction is not
byte-identical to the evaluated weights. The evaluated export records the newer
Transformers version and left padding; vocabulary and diagnostic token encodings
are checked separately.

## Held-out next-action results

The frozen two-layer, strength-4 edit was tested once on all 94 reserved tasks,
with a new sampling seed and the same prompts and decoding parameters in both arms.

| Measure | Original | Edited |
| --- | ---: | ---: |
| Exact repeated command | 37/94 (39.4%) | 7/94 (7.4%) |
| Near-repeated command | 62/94 (66.0%) | 43/94 (45.7%) |
| Repetition inside one command | 11/94 (11.7%) | 6/94 (6.4%) |
| Valid, distinct command | 31/94 (33.0%) | 48/94 (51.1%) |
| Valid bash call | 94/94 | 92/94 |
| Hit 512-token limit | 0/94 | 2/94 |
| Premature submission | 0/94 | 0/94 |

The gain in valid, distinct commands is 18.1 percentage points; a paired task
bootstrap gives a 95% interval of +6.4 to +29.8 points. Exact repeats fall by 31.9
points (interval −41.5 to −22.3). The internal-repeat reduction is less certain
(interval −11.7 to +1.1). These intervals describe this task set and one sampling
seed, not independent replications or uncertainty from candidate selection.

The edit reduces repetition on unseen task histories, but nearly half the actions
are still close to earlier commands. Two outputs truncate, and command novelty is
not a measure of bug-fixing success. Commands containing `grep` or `rg` remain
82/94 before and 81/94 after the edit. A live check uses the existing repository
sandboxes and hidden tests on 16 evenly spaced tasks from the reserved partition,
with 40 turns and the original 1,536-token per-turn cap in both arms.

## Live task check

The existing vLLM 0.31.0 evaluator ran both models on the same 16 tasks, with
temperature 0.7, top-p 0.8, top-k 20, seed 0 and 40 turns. These tasks come from the
reserved partition. Repository commands run in network-blocked Modal sandboxes;
the harness grades patches with its existing hidden tests. Both arms complete
all 16 environments without setup failures.

| Measure | Original | Edited |
| --- | ---: | ---: |
| Tasks resolved | 0/16 | 0/16 |
| Episodes with a patch | 1/16 | 1/16 |
| Submitted | 0/16 | 0/16 |
| Exact repeated calls | 54.04% (328/607) | 28.05% (131/467) |
| Ended without a tool call | 2/16 | 8/16 |
| Reached 40-turn limit | 14/16 | 8/16 |

The live check confirms reduced exact command repetition, but shows no improvement
in task completion and more early episode endings. Fewer turns cannot be counted
as an improvement when they end without a solution. The candidate is not promoted
as a fixed agent. These are small, single-seed diagnostics, not a replacement for
the original 179-task benchmark or evidence that weight edits cannot work.

## Second round: protecting tool-call structure

The second round targets the first edit's early endings. It uses the same 24
calibration loop contexts, eight first-use contexts as controls, and the eight
terminal contexts from the first candidate's live check. These 40 examples are
calibration data; the terminal contexts are not held out. Sixteen live tasks are
reserved from the 61 original benchmark tasks excluded from all 118 first-round
diagnostic examples. Selection is round-robin across repositories, ordered by a
fixed task hash within each repository, before evaluating any new candidate.

`round2.py` collects prediction-aligned activations for the command string alone,
separating the opening and closing tool-call syntax. It uses MLP layers 14 and 17
and the original 16 calibration pairs. The output direction is projected away
from eight dominant directions of the format-token outputs. Two variants also
project their input direction away from 16 dominant directions of format and
first-use command activations. This protects a fitted subspace approximately;
it does not guarantee valid tool calls in new contexts. The selected-column update
norm is capped at 25% of the original selected-column weight norm.

The candidate set includes a 64-column output-protected edit, 64- and 128-column
edits with input protection, and a half-strength interpolation of the first-round
candidate toward the original. Two nominal strengths of the 128-column edit hit
the same cap and produce byte-identical patches; they count as one candidate.
Calibration generation uses vLLM 0.31.0, bf16, temperature 0.7, top-p 0.8, top-k 20,
a 1,536-token cap and fixed per-request seeds. Prefix caching is disabled while
patches are loaded, and the original is restored for a repeated generation check.
The live harness records raw outputs and termination reasons for diagnosis.

| Calibration candidate | Valid loop calls /24 | Distinct loop calls /24 | Valid failure-context calls /8 |
| --- | ---: | ---: | ---: |
| Original | 24 | 6 | 6 |
| First-round edit | 23 | 10 | 7 |
| Half of first-round edit | 23 | 9 | 6 |
| Output-protected, 64 columns | 24 | 7 | 6 |
| Input/output-protected, 128 columns | 24 | 6 | 6 |
| Input/output-protected, 64 columns | 24 | 6 | 4 |

Every arm makes valid calls on all eight first-use controls. All invalid outputs
in this calibration sweep hit the 1,536-token limit. Raw outputs include repeated
`grep -v` pipelines inside one unfinished tool call, as well as long patch-writing
commands. An empty parsed answer therefore does not establish intentional early
stopping. Restoring the original weights reproduces all 40 original token sequences.

The output-protected 64-column candidate is frozen for the live check. It changes
129,702 scalar weights (0.02176% of the model), confined to the same two MLP tensors.
It is a conservative tradeoff: it preserves all 24 calibration loop calls, but its
novelty gain is small and weaker than the first edit. The calibration results do
not establish that it fixes either repetition or early endings.

The three models then run on the same 16 reserved live tasks, with the existing
40-turn harness, 1,536-token output cap, sampling settings and seed 0. The candidate
is frozen before this evaluation; no edit is selected using these live outputs.

| Fresh live tasks | Original | First-round edit | Second-round edit |
| --- | ---: | ---: | ---: |
| Tasks resolved | 0/16 | 0/16 | 0/16 |
| Exact repeated calls | 41.53% (233/561) | 17.05% (90/528) | 54.55% (348/638) |
| Ended with a truncated tool call | 3/16 | 5/16 | 0/16 |
| Exhausted context | 0/16 | 1/16 | 1/16 |
| Reached 40-turn limit | 12/16 | 10/16 | 15/16 |
| Episodes with a patch | 2/16 | 1/16 | 2/16 |
| Submitted | 1/16 | 0/16 | 0/16 |

The raw-output recording identifies every parsed-answer ending in the original
and first-round arms as a generation cut off at 1,536 tokens. Thus the first edit's
shorter episodes do not demonstrate better stopping behavior. The second edit
avoids those endings in this sample, but returns to more repetitive searches and
does not improve patch correctness. It is not promoted as a repaired model.
These are 16 tasks and one sampling seed; the result is a diagnostic tradeoff,
not a stable estimate across seeds or repositories. The original, both candidate
checkpoints and all negative calibration trials are preserved.
All 48 episodes reach hidden-test grading without environment errors. The remote
supervisor completes cleanup of the 144 tracked setup, execution and grading
sandboxes. Local checks cover edit-subspace math, sparse-edit restoration and the
repetition metric; formatting and lint checks pass.

## Artifacts and commands

Downloaded and generated artifacts are under ignored `training/artifacts/weight-space/`.
Remote outputs are on the `lm-exploration-hybrid-runs` Modal volume under
`weight-space/qwen06-loop-20261103-01/`. The original source run remains under
`agentic-reference/qwen3-0.6b-base-scale-500k-02/`. The archive of the original
evaluation is the HF dataset `jfrick/lm-exploration-results`, folder
`training/results/agent-mix-sft/`.

Second-round outputs are in local `artifacts/weight-space/round2/` and remote
`weight-space/qwen06-loop-20261103-02/`. The downloaded candidate is
`round2/candidate-content64-s2/`; its full tensor comparison confirms only the two
intended matrices differ, and all tokenizer, generation and model configuration
files are byte-identical to the original. The directory retains calibration
outputs for every candidate, failed launcher logs, patch hashes, the frozen live
task list and candidate selection, and launch/deadline records.

`modal_run.py` has `scan`, `attention-scan`, `candidates` and `test` phases. Supply
an experiment run ID and a JSON selection file for generation phases. Each candidate
specifies a name, MLP layer indices and strength; optional attention indices refer
to the saved attention proposals. Use fresh names to preserve individual trials.
The test selection, source hashes, checkpoint hashes and launch IDs are recorded
with the artifacts. Every launch has a two-hour deadline, a 6,600-second worker
timeout, retries disabled, and a separately running remote cancellation supervisor.
`live.py` reuses the existing agent evaluator, with a one-hour launch deadline,
3,300-second worker timeouts, and a remote registry/supervisor that explicitly
tracks and terminates every sandbox. Its outputs are also retained on the
`lm-exploration-hybrid-data` volume under `pr-tasks-pilot/agentic/`, and downloaded
under `artifacts/weight-space/live/`.

```sh
python analysis/weight_space/prepare.py \
  --trajectories /path/to/trajectories.jsonl \
  --tokenizer /path/to/original/model \
  --output artifacts/weight-space/prompts.json
python analysis/weight_space/modal_run.py --run-id NEW_ID --phase scan \
  --model-path /runs/agentic-reference/SOURCE_RUN/model
python analysis/weight_space/modal_run.py --run-id NEW_ID --phase candidates \
  --selection /path/to/candidates.json --model-path /runs/agentic-reference/SOURCE_RUN/model
python analysis/weight_space/modal_run.py --run-id NEW_ID --phase test \
  --selection /path/to/frozen-test-selection.json --model-path /runs/agentic-reference/SOURCE_RUN/model
python analysis/weight_space/export.py --model /path/to/original/model \
  --proposals /path/to/proposals.pt --layers 14 17 --strength 4 \
  --output /path/to/fresh/candidate
python analysis/weight_space/test_study.py
python -m unittest analysis.weight_space.test_round2
```

For the second-round dataset, run `prepare_round2.py` with `--tokenizer`,
`--previous-prompts`, `--failure-trajectories`, `--source-trajectories` and `--output`.
`round2_modal.py` provides `build`, `generate` and `export` phases for this checkpoint;
generation/export take a `--selection` JSON file. `live.py --config CONFIG --output
DIRECTORY` takes explicit task IDs, model variants and the turn cap. All round-two
launches have a one-hour deadline, a 3,300-second worker timeout and retries disabled.
`python -m analysis.weight_space.analyze_live --trajectories FILE --raw FILE
--tokenizer DIRECTORY --output FILE` joins terminal turns to recorded raw generations
and emits behavioral counts and termination diagnostics.

`summarize.py` rescores saved outputs using the command heuristics, emits JSON
summaries, and computes paired task-bootstrap confidence
intervals for a supplied original/candidate pair. Tests check exact restoration
and sparse edit scope, and ensure a duplicated search cannot pass as distinct merely
because the full command string changed.
