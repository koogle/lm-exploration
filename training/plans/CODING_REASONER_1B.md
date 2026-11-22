> **Two paths.** The 960M foundation model is trained:
> [Malama Base](https://huggingface.co/jfrick/malama-base), 100B tokens. Two paths
> build on it. **Coding reasoner** (this plan) is the primary path, with the
> LiveCodeBench target below. **Autocomplete** ([plan](MALAMA_AUTOCOMPLETE.md))
> continues from its fill-in-the-middle branch; the reasoner's midtraining keeps FIM
> data so that one model can serve both.

Updated November 15, 2026. Train a model from scratch through foundation
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
references before choosing SFT or RL curricula. Measured results of the code
mid-training, reasoning SFT and contest RL stages are in the
[findings](CODING_REASONER_1B_FINDINGS.md).

## LiveCodeBench target

Set the best LiveCodeBench v6 pass@1 among models under 1B parameters. The sub-1B
models to beat, from the [frontier survey](../references/open-model-frontier-2026-11-01.md#november-4-refresh):

| Model | Parameters | LCB v6 | Training |
| --- | ---: | ---: | --- |
| MobileLLM-R1.5-950M | 0.95B | 29.1 (19.9 after SFT) | About 4T pretraining tokens, reasoning SFT, on-policy distillation |
| MobileLLM-R1.5-360M | 0.36B | 8.6 | Same recipe |
| K2 Horizon "0.9B" | 1.08B stored | 37.4 | 5T pretraining tokens, RL experts, merging, multi-teacher distillation; above the 1B cutoff, reported as the stretch bar |

These are vendor numbers with long reasoning and pass@1 averaged over samples. The
claim counts only with our own measurement of the reference models under the same
protocol:

- **Protocol.** Thinking mode, temperature 0.6, top-p 0.95, 16 samples per problem,
  pass@1 averaged over samples, up to a fixed reasoning budget (32,768 tokens to start),
  graded on all public and private tests by the existing
  [LiveCodeBench harness](../analysis/evaluation/README.md#livecodebench).
- **Windows.** v6 (2025-01-04 to 2025-04-06) for comparison with published numbers,
  plus the newest release window available, which postdates every teacher used for
  our traces. Training problems are cut by date before both windows and screened by
  13-gram overlap against both.
- **Size.** Malama's deployed trunk is 928M parameters, under the cutoff.

The gap is large. Malama Base saw 100B tokens, 40 times fewer than MobileLLM-R1.5;
the stages below make up for it with targeted data and dense teacher supervision
rather than more pretraining, and each one is measured on this protocol before the
next starts.

### Stage plan

1. **Reasoning midtraining (20–40B tokens).** From the foundation run's last
   constant-learning-rate checkpoint, as the FIM branch did, with its own decay.
   Extend the context from 4,096 to 32,768 tokens (RoPE base of the full-attention
   layers raised; KDA layers need no position change) and validate long-context
   retrieval before any long traces. Mix: competitive-programming problems with
   verified solutions, high-quality code and code explanations, math (CC-Math 4+),
   reasoning traces as plain text, FIM at a reduced rate, and foundation replay.
2. **Reasoning SFT.** Code-contest reasoning traces from strong open teachers,
   verified by tests where they ship with them, several epochs, thinking spans
   supervised and the final fenced program graded. Candidates: OpenCodeReasoning-2,
   the Nemotron post-training code and math sets, OpenThoughts3; all filtered by
   license, problem date and the screen above. Compare against answer-only SFT at
   matched tokens, as planned below.
3. **On-policy distillation.** Malama uses the GLM tokenizer, so a GLM teacher that
   shares it (GLM-4.7-Flash or GLM-5.3-Flash, to verify token for token) can score
   Malama's own samples with full-vocabulary KL and no cross-tokenizer alignment.
   MobileLLM-R1.5's step from 19.9 to 29.1 came from this stage.
4. **RL on executable rewards.** GRPO-style RL on test-graded contest problems held
   out from both windows, with curriculum matched to the model's success rate, after
   SFT solves enough problems to give mixed groups. KDA's constant-size state keeps
   long rollouts cheap.

### Work before the first GPU stage

- Score Malama Base and the FIM branch on the current no-reasoning protocol
  (running), then measure MobileLLM-R1.5-950M, MobileLLM-R1.5-360M and Qwen3-0.6B
  on the thinking protocol, so the target is our own number.
- Thinking mode end to end: `data/chat_sft.py` rejects thinking spans and
  `hybrid/post_train.py` builds non-thinking prompts (see
  [reasoning training](../references/reasoning-training-2026-11-04.md#required-changes-in-our-pipeline)).
- Long-context training and generation for the hybrid family at 32,768 tokens.
- Data audit: sources, licenses, token counts with the GLM tokenizer, date cuts and
  the contamination screen for both windows.

### Budget

Estimates on 8 × H100 at the foundation run's throughput, before long-context
slowdowns: midtraining 30B tokens about 220 H100-hours; SFT about 10B tokens
(three epochs) about 75 H100-hours; on-policy distillation and RL about 400–600
H100-hours together, dominated by long rollouts and teacher scoring; evaluation
about 10 H100-hours per model on the thinking protocol. About 700–900 H100-hours in
all. Reconcile with actual spending before launching the midtraining stage.

## Stages and advancement

1. **Foundation pretraining (complete):** 100B tokens of 50% ClimbMix,
   40% code and 10% Nemotron CC-Math score-3 data, finished on November 14, 2026 as
   [Malama Base](https://huggingface.co/jfrick/malama-base). The
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

## Budget

The original approximately $10,000 remaining-credit envelope reserved $1,000 for
preparation, pilots and evaluation, $5,000 for foundation pretraining and $4,000 for
later stages and contingency. Reconcile actual experiments and billing before
committing the remaining budget; these are allocations, not a statement of the
current account balance.

Foundation pretraining is complete: 100B tokens on eight 80GB H100s, BF16, after the
code-mixture comparison (recipe v3). It took 75.3 hours of training (602 H100-hours)
at a median 380k tokens per second, over seven 24-hour launches. Runtime implementation, hardware-counter work, kernels
and MFU findings are reviewed in the stacked implementation PR, separately from
this planning decision. See [foundation constraints](KDA_960M_PRETRAINING.md).
