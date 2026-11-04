# Training reasoning below 1B — 2026-11-04

Reasoning is a training objective and learned output behavior. An autoregressive
model can emit a reasoning span and then an answer through its existing language
model head. Delimiters alone do not teach problem solving. Our next model should
explicitly learn reasoning; architecture changes remain an empirical choice.

## What the open models actually train

| Reference | Reasoning training | Implication for us |
| --- | --- | --- |
| [DeepSeek-R1](https://arxiv.org/html/2501.12948v1), §2.3–2.4 | Flagship: cold-start SFT, reasoning RL, rejection-sampled SFT, general RL. Released small distills: SFT on 800k teacher examples, without their own RL stage. | Teacher-trace SFT is a meaningful reasoning stage. R1-Zero started from a powerful pretrained base, not random weights. |
| [Qwen3](https://arxiv.org/html/2505.09388v1), §4.5 | Small models including 0.6B learn teacher responses in thinking/non-thinking modes, then match teacher logits on student-generated sequences. | Strong-to-weak distillation is more relevant than copying the flagship's four-stage RL recipe. |
| [MobileLLM-R1.5](https://huggingface.co/facebook/MobileLLM-R1.5-950M) | Reasoning SFT precedes semi on-policy forward-KL distillation with Nemotron Nano 4B. Fixed student-generated trajectories are reused for four epochs. | Dense token supervision is a serious candidate after SFT. Its 950M LCB-v6 improves from 19.9 to 29.1; reproducing its full compute is outside our budget. |
| [K2 Horizon](https://huggingface.co/IFM/K2-Horizon-0.9B) | Math/code/instruction/STEM RL experts, merging, then multi-teacher on-policy distillation. Pretraining is 5T tokens before context extension. | Evidence that RL can work near our scale with a strong base; its 1.078B stored parameters exceed our strict cutoff. |
| [Nemotron 3 Nano](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf), §3 | SFT teaches reasoning, on/off and budget control; multi-environment verifiable-reward RL follows. Hybrid Mamba/attention with MoE. | Hybrid recurrence can support reasoning. This does not establish that its architecture or mixture is optimal below 1B. |

These releases have different levels of openness: available weights do not imply
fully reproducible training data, unrestricted licenses, or identical evaluation.
The comparison is a recipe study, not a claim of an exhaustive latest leaderboard.

## Required changes in our pipeline

At the current implementation, `data/chat.py` can render thinking for packed
midtraining, but its SFT encoder renders non-thinking turns.
`data/chat_sft.py` rejects thinking-enabled conversations and explicit thinking
spans. `hybrid/post_train.py` also constructs non-thinking RL prompts. Consequently,
adding reasoning data to a manifest does not establish end-to-end reasoning SFT/RL.

Before training, implement and verify:

1. Explicit thinking mode throughout rendering, SFT masks, generation and evaluation.
   Supervise assistant reasoning, final answer and termination; mask user/tool input.
2. Preserve complete traces. Measure lengths with the chosen tokenizer, bucket by
   length and reject overflow rather than cutting off answers. Extend and validate
   context before introducing long traces.
3. Grade final answers/code separately from exploratory reasoning. Apply outcome
   reward to the generated trajectory, including reasoning tokens. Track valid
   termination, correct final extraction, truncation, loops and actual token cost.
4. Add teacher-logit distillation only after choosing teacher/tokenizer and measuring
   scoring cost. Direct vocabulary-wise KL requires compatible token identities;
   our GLM tokenizer cannot directly match Llama/Qwen logits. Teacher-text SFT works
   across tokenizers. Cross-tokenizer distribution alignment would be additional work.
5. Compare reasoning on/off at fixed output budgets on held-out problems. A longer
   response or a correctly emitted thinking delimiter is not evidence of better reasoning.

The existing MTP auxiliary head predicts future tokens; it is not a reasoning head.
A full-attention control against the KDA/attention hybrid would test the architecture
question at matched training compute. Keep the sub-1B total-parameter target, but
choose tokenizer and teacher before freezing dimensions: vocabulary embeddings use
a substantial fraction of a small model's capacity.

## NVIDIA data candidates and verified access

Authenticated data-byte probes on November 4, not just model-card access:

| Dataset | Role | Access / preparation implication |
| --- | --- | --- |
| [Nemotron-CC-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1) | Extracted web code text for pretraining | HTTP 403: awaiting author review after terms acceptance. Keep its planned slot pending. |
| [Nemotron-Pretraining-Code-v3](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v3) | Incremental GitHub corpus metadata | HTTP 206. Contains commit IDs, paths and language, not source text. Requires retrieval plus license/provenance validation; intended alongside v1/v2. |
| [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | Math web text for pretraining | User reports access granted; a fresh data request with the workspace HF token still returns HTTP 403 (not in authorized list). Its 4+ subset is a candidate comparison against FineMath once workspace access resolves. |
| [Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) | Reasoning trajectories at multiple budgets, with/without tools | HTTP 206. Useful for a length/budget curriculum and tool-free initial SFT. |
| [Nemotron-SFT-Math-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v3) | Reasoning SFT with and without Python tools | HTTP 206 verified against `data/train.jsonl`. Shares seed problems with v2/v4. |
| [Nemotron-SFT-Math-v4](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v4) | Newer reasoning SFT candidate | HTTP 206. DeepSeek-V4-Pro high-mode solutions on Math-v2 problems, filtered by final-answer agreement. Compare length and learnability against v2 before selecting. |

Math-v2, SFT-Math-v3 and v4 reuse seed problems: split by problem identity across
versions, not by trajectory. Preserve row-level attribution/license fields. Select
non-tool traces initially; tool observations must not become ordinary assistant
text. Correct final answers do not verify every intermediate reasoning step.
Apply our own benchmark screen and retain an untouched coding evaluation set.

## Recommended experiment sequence

First validate a bounded sample through the actual thinking SFT encoder, loss mask,
generation and final-answer grader on the small pilot. Then compare answer-only
SFT with verified reasoning SFT from the same base, matching training tokens and
reporting inference budgets. Retain both code and math, with code the majority for
our LiveCodeBench objective. Measure transfer, rather than assuming math improvement
implies coding improvement.

After this works, compare teacher-guided distillation and verifiable-reward RL from
the same reasoning checkpoint. Start RL on tasks with some successes and failures
per sample group; all-fail groups give no useful relative reward signal. Reserve
budget for student rollouts, teacher scoring and evaluation before allocating the
large pretraining run. No GPU run or reasoning implementation is part of this reference update.
