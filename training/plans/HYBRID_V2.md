# Hybrid v2: a sub-2B coding and agentic model

Goal: the best open model below 2B parameters for coding, including agentic coding
(issue → repository edit → tests pass), starting from the 720M hybrid
([HYBRID_720M.md](HYBRID_720M.md)) and growing only where measurements show size is
the limit.

## Where v1 stands (measured)

| | 720M v1 (40B tokens, SFT, RL) | Best reference ≤ 2B |
| --- | ---: | ---: |
| LiveCodeBench v6 pass@1 | 1.7% (SFT) | 6.9% (Qwen3-1.7B-Base); 6.3% at 0.6B |
| HumanEval / MBPP | 13.4 / 12.0 | 29.9 / 38.8 (Qwen3-0.6B-Base) |
| Context | 4,096 | 32K+ |
| Reasoning traces, tool use in repositories | none | — |

Qwen gains almost nothing from 0.6B to 2B on LiveCodeBench (6.3% → 6.9%) and jumps
at 4B (17%), so below 2B the gap is data and training, not parameters.

## Stages

Each stage has a gate; a failed gate stops the plan for a decision.

1. **Long context, 32K** (implemented on this branch; data preparation and run
   next). Continued pretraining of the 40B export on `hybrid-long32k-v1`: 2B tokens,
   60% long documents (repository-level permissive code, files of one repository
   joined in path order, ≥ 4,096 tokens; PG19 books) and 40% replay of
   `hybrid-40b-v1` read at 32K. Learning rate a fifth of pretraining, linear decay.
   *Gate:* held-out loss on long documents keeps falling with in-document position
   up to 32K (`loss_by_position`), and the short-context benchmarks stay within one
   point of the 40B export.
   *Measured:* 32K training fits one H100 at microbatch 1: 62,700 tokens/s, 31.9% MFU
   (synthetic 30K-token documents, 2026-10-28).
2. **Code-heavy base.** Continue pretraining with code at about 40% (permissive
   GitHub code plus the repository collections), math 15%, web 45%, for 40–60B more
   tokens at 32K. Before scaling, an ablation at 2B tokens compares 720M with a
   ~1.5B variant (24 layers, width 2048, same layer pattern).
   *Gate:* HumanEval/MBPP at or above Qwen3-0.6B-Base's 30/39, measured, before any
   larger run.
3. **Reasoning SFT** at 16–32K: thinking traces for code and math (smoltalk2's think
   splits, OpenCodeReasoning-style data, GLM-5.3-written traces kept only when their
   code passes tests), then the existing RL loop with thinking on.
   *Gate:* LiveCodeBench v6 above 6.9% (Qwen3-1.7B-Base).
4. **Agentic SFT.** Multi-turn trajectories on PR→task environments
   ([pilot](HYBRID_720M.md#pr-to-task-pilot-2026-10-28), 506 verified tasks):
   GLM-5.3 acts as the agent with read/search/edit/run-tests tools inside the
   verification sandboxes; only trajectories whose final patch makes the
   fail-to-pass tests pass are kept, rendered in GLM's tool-call format.
5. **Agentic RL.** The same sandboxes give the reward (fail-to-pass tests pass, no
   regressions), with a curriculum from single-file to multi-file tasks.

## Evaluation

- LiveCodeBench v6, HumanEval/MBPP (and EvalPlus) every stage.
- Held-out PR tasks from repositories never used for training (split by
  repository), resolved rate.
- SWE-bench Verified (easy subset) and Terminal-Bench easy tasks once stage 4 runs.
- Long context: `loss_by_position`, plus retrieval of a value planted early in a
  32K prompt.

## Data needs

- More repositories with tests (Python and Go first): the MIT collector runs again
  with batched uploads; a fresh GitHub token triples its search rate.
- Licenses: permissive code only for pretraining (github-code-clean allowlist, MIT
  collection); PG19 is public domain; GLM-5.3 outputs (MIT-style weights license)
  may be used to train.

## Mid-training run (`hybrid-mid32k-v1`)

One run addresses the three v1 gaps together: context (long documents at 32K), code
(a code-heavy share of unseen permissive GitHub files) and reasoning (thinking traces,
OpenThoughts3 math, code and science traces from smoltalk2, rendered in GLM's chat
format with every turn's `<think>` kept). Sources prepared by `hybrid-long32k-v1` are
reused through `prepared_in`. 3B tokens from the 40B export with 30% replay of
`hybrid-40b-v1`; config `hybrid/configs/hybrid-720m-mid32k.json`.

| Non-replay source | Share |
| --- | ---: |
| Repository-level code (≥ 4,096 tokens) | 35% |
| PG19 books | 10% |
| Unseen permissive GitHub files | 30% |
| Thinking traces (OpenThoughts3) | 25% |

### Result: `hybrid-720m-mid32k-01` (2026-10-28)

8×H100, 2,861 steps, 3.0B tokens in 1.7 hours (median 513K tokens/s, 31.4% MFU), from
the 40B export. Long-document loss falls with in-document position, so the model uses
context up to 32K; the replayed short-document sources improved too.

| Held-out source | Loss, step 200 → end | Loss by position, 0–1K → 16–32K (end) |
| --- | --- | --- |
| Repository code | 1.044 → 0.901 | 1.183 → 0.610 |
| Unseen GitHub files | 1.002 → 0.886 | 1.160 → 0.341 |
| PG19 books | 3.178 → 2.868 | 2.945 → 2.838 |
| Thinking traces | 1.949 → 1.647 | 1.281 → 1.819 (rises: later trace text is harder) |
| ClimbMix / code / math (replay) | 2.636 / 1.019 / 1.794 → 2.575 / 0.894 / 1.724 | — |

Benchmarks (same harness and shots as the 40B record), 40B export → mid-training:

| HellaSwag | ARC-E | ARC-C | PIQA | WinoG | OBQA | SciQ | LAMBADA | MMLU | GSM8K | HumanEval | MBPP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 51.4 → 48.7 | 63.4 → 60.7 | 35.4 → 32.8 | 72.7 → 73.0 | 57.4 → 55.6 | 38.8 → 38.0 | 87.8 → 85.9 | 41.2 → 43.9 | 26.4 → 24.8 | 3.5 → 2.4 | 4.3 → 6.1 | 5.0 → 8.0 |

LiveCodeBench v6 (base prompt): 0/175, as for the 40B export.

**Gate not met.** Code improved (HumanEval +1.8, MBPP +3.0) and long context works,
but commonsense benchmarks fell 2–3 points (HellaSwag, ARC), beyond the one-point
limit. Web text was 21% of training tokens here (ClimbMix inside the 30% replay)
against 70% in the 40B run. Next, before more stages: raise replay to 50% and keep
the code share by lowering books, compared on the same 3B tokens.

A second, accidental launch of the same run trained independently for about 500
steps, then failed out of memory before its first checkpoint; it wrote no
checkpoint and the recorded metrics are the completed launch's only.

Outputs (metrics, configs, capacity, benchmarks, LiveCodeBench):
`training/results/hybrid-720m-mid32k-01/` in `jfrick/lm-exploration-results`.

## Agentic SFT pilot (`agentic-sft-v1`, 2026-10-29)

Agent episodes on the 506 verified PR tasks (`data/pr_task_agent.py`, `modal_scripts/modal_agentic.py`):
the agent starts from the parent revision with the commit's tests and git history
removed, has bash, read/edit/write file and submit tools (40 turns), and is graded by the
commit's hidden tests in a separate sandbox (resolved: every fail-to-pass and
pass-to-pass test passes). Repositories are split by owner: 179 tasks from 8
repositories are held out.

### Teacher: GLM-5.3 (medium reasoning effort)

| | Episodes | Resolved |
| --- | ---: | ---: |
| Held-out repositories (one attempt) | 179 | 113 (63.1%) |
| Training repositories (two attempts) | 654 | 370 (56.6%), 211 of 327 tasks |

$45.66 for 833 episodes (265M prompt tokens), 85 minutes. A 12-task pilot at low effort
and 30 turns resolved 3.

### Student: `hybrid-720m-agentic-sft-02`

From the 32K mid-training export: the 356 resolved training-repository trajectories
that fit 32K (4.6M tokens, read 4 times) with a seeded 30% of chat-sft-v1 and 25%
replay of `hybrid-40b-v1`, 400M tokens, 381 steps in 21 minutes on 8×H100. The first
launch (`-01`) failed out of memory on one rank at step 45 with 8.4 GB reserved but
unallocated; the image now uses expandable allocator segments.

**Held-out resolved: 0 of 179.** The format is learned: 99.8% of 6,507 assistant turns
are well-formed tool calls. The behavior is not: 98.6% of calls are bash exploration
(grep, sed), 70% of calls repeat an earlier call in the same episode, 28 edits were
attempted (about half with text that does not occur in the file), 5 episodes produced
a patch and none submitted. 145 episodes ended at the turn limit, 20 at the context
limit and 14 by answering without a tool call.

Outputs (trajectories, summaries, run metrics, the failed launch's log):
`training/results/agentic-sft-pilot/` in `jfrick/lm-exploration-results`.

## Qwen reference runs (2026-11-01)

Whether broad pretraining is what the 720M student lacks: Qwen3 models, pretrained on
36T tokens, through the same 179 held-out tasks (`modal_agentic.py::reference_train`,
`::reference_evaluate`; vLLM, greedy, 40 turns, 1,536 new tokens per turn, as the
hybrid evaluation).

- **Fine-tuned:** Qwen3-0.6B-Base and Qwen3-1.7B-Base, full fine-tune on the resolved
  training-repository trajectories of `glm-5.3-1793298716` in Qwen3's chat format
  (365 fit 32K with Qwen's tokenizer; 5.11M tokens, 1.26M of them assistant tokens per
  epoch), 4 epochs, 183 steps of 8 trajectories, learning rate 2e-5 with cosine decay,
  loss on assistant turns only. Unlike `hybrid-720m-agentic-sft-02`, no chat-SFT mix
  and no pretraining replay. One H100 each: 17 and 24 minutes, final loss 0.173 and
  0.108.
- **Zero-shot:** Qwen3-0.6B and Qwen3-1.7B (post-trained chat models), thinking off.

**Held-out resolved: 0 of 179 for all four.** Behavior from
`evaluation.agentic_behavior` (edits pair with their results by position):

| Agent | Resolved | Turn limit | Submitted | Bash calls | Repeated calls | Edits applied | Episodes with patch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-5.3 teacher (held-out) | 113 | 66 | 82 | 74% | 0.3% | 857 / 973 | 174 |
| Hybrid 720M, agentic SFT | 0 | 145 | 0 | 99% | 75% | 8 / 28 | 5 |
| Qwen3-0.6B-Base, agentic SFT | 0 | 158 | 0 | 95% | 64% | 18 / 185 | 12 |
| Qwen3-1.7B-Base, agentic SFT | 0 | 147 | 0 | 96% | 52% | 26 / 85 | 10 |
| Qwen3-0.6B, zero-shot | 0 | 1 | 131 | 2% | 27% | 9 / 22 | 9 |
| Qwen3-1.7B, zero-shot | 0 | 46 | 71 | 38% | 76% | 349 / 1,191 | 65 |

The fine-tuned Qwen models fail like the 720M student: well-formed tool calls (99.8%
of turns) that mostly repeat bash exploration until the turn limit, edits that mostly
fail to apply, no submission. Broad pretraining does not change this with
365 trajectories, so the 720M result is not explained by its pretraining alone. The
zero-shot chat models behave differently but no better: Qwen3-0.6B submits at once in
131 episodes; Qwen3-1.7B edits in 65 episodes (and calls tools that do not exist,
such as `grep`), without a passing patch.

Published small agents use far more trajectories: mini-coder-1.7b, about 400k
(18.6% on SWE-bench Verified); SWE-smith's 7B model went from 2.2% at 100 trajectories
to about 17% at 2.4k (see the
[open-model frontier reference](../references/open-model-frontier-2026-11-01.md)). The
next test is trajectory count, and sampling instead of greedy decoding, which every
run here used and which favors repeated calls.

The fine-tuned weights are on the `lm-exploration-hybrid-runs` volume
(`agentic-reference/<run>/model`). Outputs (evaluation summaries and trajectories,
including the two 4-task smoke runs, the first of which failed before the sandbox
image fix; SFT metrics and summaries; `qwen-reference-behavior.jsonl`):
`training/results/agentic-sft-pilot/` in `jfrick/lm-exploration-results`.

### Sampling, more trajectories and a working 1.7B agent (2026-11-01)

The three follow-ups, on the same 179 held-out tasks:

- **Sampling.** The two 365-trajectory fine-tunes, re-evaluated at temperature 0.7
  (Qwen3's top-p 0.8, top-k 20) instead of greedy.
- **A working small agent as control.** `ricdomolm/mini-coder-1.7b`, Qwen3-1.7B trained
  on 396,515 trajectories of Qwen3-Coder-30B-A3B in mini-swe-agent on SWE-smith's Python
  tasks (18.6% on SWE-bench Verified, bash only), run greedily in its own bash-only format
  (`training_data.mini_swe_format`; the dataset's system prompt and instructions, one
  command per turn, `submit` on its submission command).
- **More trajectories.** Qwen3-0.6B-Base and Qwen3-1.7B-Base trained one epoch on 10,000
  trajectories sampled from that dataset (`ricdomolm/mini-coder-trajs-400k`, pinned;
  9,923 fit 32K: 132.7M tokens, 46.4M of them assistant tokens; 1,241 steps; 2.2 and 2.8
  hours on one H100, final loss 0.353 and 0.306), and Qwen3-0.6B-Base on 20,000 (19,879:
  264.3M tokens, 92.1M assistant; 2,485 steps; 4.3 hours; final loss 0.427). Same
  optimizer settings as above; evaluated greedily in the bash-only format.

| Agent | Trajectories | Resolved | Repeated calls | Episodes with patch | Submitted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B-Base, GLM-5.3 SFT, greedy | 365 | 0 | 64% | 12 | 0 |
| same, sampled at 0.7 | 365 | 0 | 37% | 28 | 1 |
| Qwen3-1.7B-Base, GLM-5.3 SFT, greedy | 365 | 0 | 52% | 10 | 0 |
| same, sampled at 0.7 | 365 | 0 | 22% | 46 | 1 |
| Qwen3-0.6B-Base, mini-coder SFT | 10,000 | 0 | 42% | 76 | 5 |
| Qwen3-0.6B-Base, mini-coder SFT | 20,000 | 0 | 37% | 69 | 8 |
| Qwen3-1.7B-Base, mini-coder SFT | 10,000 | 1 (Go) | 20% | 109 | 24 |
| mini-coder-1.7b | 396,515 | 11 (6 Go, 5 Python) | 6% | 140 | 51 |

- **The tasks are solvable at 1.7B.** mini-coder-1.7b resolves 11 of 179 (6.1%), all from
  three repositories, despite Python-only training data and a 40-turn limit that cuts
  its episodes short (its own trajectories have a median of 29 turns and a 90th
  percentile of 61).
- **Trajectory count moves behavior first, then success.** At 1.7B, 10,000 trajectories
  give the first resolved task and move every behavior measure toward mini-coder. At
  0.6B, 10,000 and 20,000 improve behavior over 365 but resolve nothing; doubling the
  data changed little.
- **Sampling reduces loops but does not substitute for data.** It halves repeated calls
  and multiplies patches for the 365-trajectory models without a resolved task.
- **Episodes that end as "answered"** (28–66 per model) are mostly replies that
  degenerate into repeated text until the 1,536-token limit, leaving no closed bash
  block; mini-coder-1.7b ends 48 episodes this way. mini-swe-agent would answer with a
  format error and continue; this harness ends the episode.

The mini-coder data is SWE-smith's 59,136 synthetic bug-fix tasks (all sampled instance
ids are in SWE-smith), about 6.7 rollouts per task, unfiltered: 16% of trajectories
pass their tests (22–36% for single procedural bugs, 3–6% for combined bugs). Its scale
comes from synthetic tasks with executable environments and a cheap open teacher,
which this repository's sandboxes and harness can reproduce on our own repositories.

Outputs (evaluations with trajectories, SFT metrics and summaries,
`mini-coder-behavior.jsonl`): `training/results/agentic-sft-pilot/` in
`jfrick/lm-exploration-results`. Fine-tuned weights are on the `lm-exploration-hybrid-runs`
volume (`agentic-reference/<run>/model`).
