# Inkling 2 Mini: a fixed three-pass looped MoE — 2026-11-21

Reviewed 2026-11-21 against Thinking Machines' November 20 release. Numbers in the
first four sections are the authors'; nothing there is measured by us. Our own looped
work is in [plans/LOOPED_LM.md](../plans/LOOPED_LM.md), and the earlier looped
releases (Dory1.0-4B, ELR / Koi-4B) are in
[looped-lms-2026-11-03.md](looped-lms-2026-11-03.md).

## Sources

1. [Introducing Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/):
   architecture, loop-placement and looped-versus-plain comparisons, training stages,
   the monitorability study.
2. [Inkling 2 Mini model card](https://thinkingmachines.ai/model-card/inkling-2-mini/):
   parameter counts, expert routing, attention layout, limitations.
3. [Artificial Analysis on Inkling 2 Mini](https://artificialanalysis.ai/articles/inkling-2-mini):
   independent cost, time and output-token measurements.

Linked from the post and not read for this note: the
[TML-d24-varloop-research model card](https://thinkingmachines.ai/model-card/tml-d24-varloop-research/)
(full per-pass evaluations), the
[4-bit training article](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/)
and [Goodfire's interpretability write-up](https://www.goodfire.com/research/interpreting-inkling-2-mini/).
Weights: [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and
[TML-d24-varloop-research](https://huggingface.co/thinkingmachines/TML-d24-varloop-research)
(Apache 2.0). No paper or technical report accompanies the release.

## Structure

| | |
| --- | --- |
| Parameters | 26.4B total, 3.4B active per token, 5.4B weight activations per token with looping (each extra pass adds about 1.0B) |
| Layers | 24 stored layers in four blocks of six |
| Split | block 1 (prelude) once; blocks 2–3 (layers 7–18) three times with shared weights; block 4 once: 6 + 3 × 12 + 6 = 48 layer applications |
| Loop count | fixed at three; the released model has no other setting |
| Experts | 8 of 128 per token (trained as 8 of 384, then pruned) |
| Attention | "a hybrid of local and global layers"; ratio and window not stated |
| Inputs | text and images (ViT encoder), up to 1M tokens of context |
| Precision | 20 layers NVFP4, the final 4 MXFP8, in training and in deployment; about 20 GB |

**Loop update.** The model uses attention residuals
([arXiv:2603.15031](https://arxiv.org/abs/2603.15031)): each layer reads the outputs of
earlier blocks and learns its own mix of them instead of one running residual sum.
Every pass starts from the prelude output. Passes two and three also receive the
previous pass's output of the middle blocks (the "message"), merged with the prelude
output through a learned gate. The gate starts closed, so each pass begins training as
a copy of the non-looped computation, and the message's gain grows during training.
The post gives this as the way looping was introduced stably.

Not stated: hidden size, heads, vocabulary; any loop-index conditioning; whether MoE
routing is shared across passes; KV-cache handling across passes; the step at which
looping was introduced; pretraining token count and compute.

**Training stages.** Derived from the lab's "mainline predictive scaling ladder";
the two changes from the mainline recipe are looping and expert pruning.

1. Pretraining with NVFP4 expert layers on GB300.
2. Midtraining: at one point, keep the 128 most-used experts per layer (two-thirds
   pruned).
3. Healing: 1T tokens distilled from a larger internal Inkling model.
4. SFT on a mix similar to Inkling 2 Small's.
5. RL: 2.2T tokens over 2,000 steps.
6. On-policy distillation for alignment.

**TML-d24-varloop-research** is the variable-loop companion: one to five passes,
trained mostly at three with less exposure to the others, and RL at three only. The
per-pass results below come from it.

## Reported learnings

- **Loop the middle.** At equal training compute and three passes, looping the middle
  12 layers beat looping the last 12 on 27 of 28 evaluations. At step 2,000 the
  non-looped baseline needed 1.51–1.91× the compute to match the middle-loop loss
  and 1.24–1.36× to match the last-loop loss.
- **Looping trades a little training compute for a lot of inference compute.** On the
  unpruned scaling ladder, the non-looped model cost 7.5% less to train and needed
  140% more inference compute to match the looped model's loss on 512K-token agentic
  traces. The post does not describe the non-looped model's size.
- **Gains are in reasoning and long context, not knowledge.** More passes help MRCR
  most, then HMMT, GPQA Diamond, HLE, SWE-bench Pro and BrowseComp; closed-book QA
  (AA-Omniscience, SimpleQA Verified, SimpleVQA, WorldVQA) is nearly invariant.
  "Looping improves the model's reasoning, it doesn't enable its weights to store
  more facts."
- **Returns level off at three passes.** The no-chain-of-thought time horizon (Think
  Fast) is 16 s at one pass, 26 s at two, 37 s at three, 42 s at five and 40 s at ten.
  N-Hop Lookup keeps improving to ten passes. Ten passes, twice the trained maximum,
  did not degrade the model.
- **Shorter reasoning, same legibility.** On GPQA Diamond questions answered correctly
  at every pass count, mean reasoning tokens are 3,703 / 1,731 / 1,661 at one / three /
  five passes. Words per sentence (8.9 / 8.7 / 8.7) and the dictionary-word fraction
  (94.6 / 94.5 / 94.4%) are unchanged; backtracking markers ("wait", "actually", "hmm")
  fall from 9.7 to 3.9 to 3.6.
- **No monitorability loss at this scale.** On LinuxArena with a weaker monitor
  (Qwen3.6-35B-A3B, 1% FPR), stealth rates at one to five passes show no trend
  (1.0, 2.5, 4.0, 2.8, 5.0% on tool calls alone; 0.0–0.6% when the monitor also reads
  text and thinking). The caveat is the authors': RL ran at three passes only, so a
  model trained with RL at more passes might compress its reasoning further.
- **Train in the deployment precision.** The checkpoint ships in the NVFP4 / MXFP8
  mix it was trained in, with no conversion step.
- **Stated limitations.** The model "sometimes became stuck in long reasoning loops",
  mostly while debugging code; minimum reasoning effort does not fully disable
  reasoning; multi-turn chat and many-constraint instructions trail Qwen3.8-27B
  (MultiChallenge 50.0%, ComplexConstraints 33.4%); it states wrong facts more
  confidently than larger models (AA-Omniscience −8.6, non-hallucination 63.4%
  against Qwen3.8-27B's 71.4%).

Headline results: Artificial Analysis Intelligence Index v5.0 20.1 (tied with
Qwen3.8-27B and Nanbeige4.3-33B), Vals Index v2.2 46.5%, DeepSWE v1.1 48.2%
(Qwen3.8-27B 42.2%), SWE-bench Verified 79.2%, Terminal-Bench 2.1 71.6% (Qwen3.8-27B
73.0%). The full tables are in the post and model card.

**Independent measurement.** Artificial Analysis reports $0.59 and 9.0 minutes per
Intelligence Index task against $3.53 and 38.2 minutes for Qwen3.8-27B, and about
152K output tokens per task against Qwen's 110K. "Fewer tokens" therefore holds
within the model across pass counts, not against other models at the same score.
The post gives no measured tokens per second, only a bandwidth ceiling for batch-one
decode that counts the repeated reads of the looped weights.

## Compared with our looped hybrid

| | Inkling 2 Mini | Our looped 720M hybrid ([LOOPED_LM.md](../plans/LOOPED_LM.md)) |
| --- | --- | --- |
| Backbone | MoE Transformer, local + global attention, attention residuals | 12 KDA + 4 full-attention blocks, dense, pre-norm residual stream |
| Split | 6 / 12 / 6, middle half shared | 4 / 8 / 4, middle half shared |
| Loop count | fixed 3 | `k` per request, trained on `k ~ Uniform{1..4}` with one-loop rank overrides |
| Loop entry | prelude output + gate × previous pass's message | `state + prelude output` (ELR re-injection) |
| Loop update | next pass reads the message; no interpolation described | adaptive delta: `state += beta * (proposal - state)` |
| Start of looping | gate closed: every pass equals the non-looped pass | from scratch `beta ≈ 0.35`; `loop_retrofit` starts at step 1 from a zero state |
| One-loop parity | not a goal (one pass is not served) | gate 1: within 0.01 nats of the plain model |
| Loop conditioning | not stated | stage 5 (ELR progress/budget modulation), proposed |
| Mixed-depth data | varloop model: mostly 3, some 1, 2, 4, 5 | uniform over 1..4, `p_1 = 0.34`, `k̄ = 2.31` |
| Training budget with loops | trillions of tokens plus 2.2T of RL at three passes | 1B tokens (conversion test); 10B proposed for stage 3 |
| Decoding | cache handling not stated | per-loop and last-loop caches, proposed (stage 4) |
| Evidence | benchmark gains, per-pass curves on the varloop model | held-out loss: one loop within the gate, no gain at `k = 2..4` |

What the release supports in our design:

- **Middle placement.** Their 27-of-28 result for middle over last is the first
  published placement comparison we have; ELR and Dory assert the split. Our 4 / 8 / 4
  is the same proportion.
- **Where to look for gains.** Their per-pass gains are multi-hop and long-context
  reasoning, and knowledge is flat, the same pattern as Dory and Koi. Stage 3's gate 3
  (GSM8K, HumanEval, MBPP moving; MMLU still) asks the right question.
- **Three passes is enough.** Their curves flatten at three, ELR's 4B model gets 96% of
  its four-loop gain at three, and our conversion test's fourth loop was the worst of
  the extra loops (+0.002 to +0.004 nats).
- **Bandwidth-bound decode.** Their decode ceiling counts each pass's re-read of the
  looped weights, as our [phone inference](../../inference/phone/README.md) note
  assumed.

Where it differs, and what that says about our null result:

- **How extra passes start.** Our converted model's untrained extra loops were 0.15–1.4
  nats worse than one loop, because the second loop's entry is the recurrent blocks'
  own output plus the prelude output, an input the parent never saw. The loop update's
  step then stayed at 1.0 for the whole run. Inkling's closed gate makes pass two start
  from the prelude output alone, so every pass count begins equal to the one-pass
  model and the model opens the gate only as far as the message helps. Their gate sits
  on the loop's input; ours sits on the state update.
- **Budget.** Their looping is trained through pretraining, 1T tokens of distillation
  and 2.2T of RL. The conversion test gave mixed depth 1B tokens. Their result does
  not contradict ours; it leaves stage 3 (from scratch, 10B) as the first test at a
  budget where a gain could show.
- **The metric.** Their evidence for extra passes is task accuracy, the
  no-chain-of-thought horizon and N-hop chain length. Next-token loss on our held-out
  sources is a mix dominated by tokens that need no multi-step computation, so a
  synthetic N-hop lookup at `k = 1..4` would be a more sensitive measure of whether
  extra loops compute anything.
- **Elasticity.** They ship a fixed count and keep the variable model as research,
  with uneven exposure. Our plan needs elastic `k` for the per-request latency setting
  on phone and for the `k = 1` model drafting for `k = 4`; a fixed-count model can do
  neither. That keeps ELR's one-loop machinery in our plan even though Inkling does
  without it.
- **Repetition.** Their model card reports long reasoning loops in agentic debugging.
  Our converted model's greedy repeated 4-grams rose from 0.47 at one loop to 0.56 at
  four. Neither is evidence that looping causes repetition (theirs has no per-pass
  figure, ours is a model that gained nothing from loops), but stage 4's repetition
  measurement per `k` stays in.
- **Not transferable.** MoE expert pruning and healing, NVFP4 on GB300 (our H100 FP8
  comparison is in [the Inkling 2 Small note](thinking-machines-inkling2-2026-11-07.md)),
  and attention residuals as a whole: our hybrid has one residual stream.

## Proposals

None of these is run or agreed.

1. **Closed-gate retrofit.** Add a zero-initialized gate on the state's contribution to
   the loop entry (`entry = h + g * state`) as a second `loop_retrofit` initialization
   in `HybridLM.trunk`, and repeat the 1B-token conversion test's looped branch with
   it. It is the cheapest test of whether our null result came from the starting point.
   Before any run, check on CPU that every `k` then equals the parent at step 0.
2. **N-hop probe.** A synthetic lookup-chain evaluation run at each `k` on both
   conversion-test exports and on the stage 3 runs, next to `heldout_by_loops`.
3. **`K = 3` for stage 3.** Uniform over 1..3 has `k̄ = 2` before overrides against 2.5
   at `K = 4`, about 15% less mixed-depth compute, for a fourth loop that neither
   source finds worth much.
