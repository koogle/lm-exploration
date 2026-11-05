# Ricursive Ripple: lessons for our sub-1B coding model

Reviewed 2026-11-05. Ricursive Intelligence's Ripple is distinct from Recursive's
Flint. These are author-reported results, not measurements by this project.

## Primary-source findings

The [launch post](https://www.ricursive.com/ripple) describes iterative model
ladders, substantially upsampled code, and a combined recipe improvement of up to
16× compute efficiency at 4B on dolma_stack cross-entropy. This is neither an
isolated data ablation nor a 16× coding pass-rate improvement. Some architectural
changes reversed their ranking with scale: two-stream mHC helped small models but
not 4B, while pre+post normalization started helping at 4B. They report $32M total
R&D, including experiments and synthetic data, not just the final training run.

Their RL curriculum starts from strong SFT, matches tasks to current ability,
and increases difficulty. The reported 1.4M rollouts and 80.52 Terminal-Bench 2.1
score do not establish comparable efficiency for a weak sub-1B base.

The [model card](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) describes a
65B dense decoder, 80 layers and width 5120, local/global attention, gating and
stability changes. Natural data dominates pretraining; substantially more synthetic
data enters midtraining. Reported stages include 12T pretraining, 200B midtraining
and 183B SFT tokens. Exact source weights are not disclosed. Benchmark results
are harness-sensitive and the headline comparison concerns US open-weight models,
not an exhaustive global frontier. There is no separate reasoning head described.

## Our interpretation and decisions

- Keep the user's preferred KDA/full-attention hybrid and strict sub-1B deployed
  target. Ripple's attention design is motivation for a control, not evidence
  against KDA. Measure quality and hardware cost together.
- Borrow the experimental method: isolate data and mixer changes, retain negative
  results, and check transfer to target scale. Do not add normalization, NoPE,
  value residuals and head-count changes simultaneously.
- Treat natural-code-heavy pretraining and later synthetic reasoning as a testable
  curriculum. Neither Ripple nor Nemotron establishes the optimal mixture for us.
- Reasoning remains explicit midtraining/SFT plus a bounded RL/distillation
  comparison. Correct thinking delimiters and low training loss are insufficient.
- Start RL where groups contain successes and failures. Final executable correctness
  and independent transfer are the outcomes; longer traces are not the objective.

Our 128M, 131K-token control established preparation and checkpoint continuity,
including a fixed optional-sample resume comparison. It did not compare mixtures.
The earlier agent-trajectory SFT failures in [HYBRID_V2](../plans/HYBRID_V2.md)
show why imitation loss cannot replace execution evaluation. They were not a
controlled negative test of policy-gradient RL or LiveCodeBench.

The [KDA and data study](../plans/KDA_DATA_STUDY.md) specifies revised weights,
small experiments and advancement criteria. It is a proposal, not a launched run.
