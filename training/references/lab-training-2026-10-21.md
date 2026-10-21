# Learning from Mirendil, Recursive and Flapping Airplanes

Reviewed 2026-10-21. “Mendra” is interpreted as **Mirendil**, whose announcement
matches the requested date and topic. All external results below are author-reported,
not reproduced here. These are references and experiment proposals, not a new recipe.

## What the announcements actually establish

### Mirendil: human-directed, agent-assisted pretraining

[Pretraining at scale without a pretraining team](https://mirendil.com/news/pretraining-at-scale-without-a-pretraining-team/)
(2026-10-21) reports one researcher building Mirendil-Base in 3.5 months for under
$10M, including research runs, agent tokens and data processing. Agents wrote over
99% of the code, ran experiments and diagnosed failures; the researcher selected
directions and decided which results to trust. Training used public organic data,
without synthetic data or distillation. The reported efficiency improvements span
architecture, data mixtures and kernels, but the post does not disclose a complete
reproduction recipe or the model's parameter count.

Its external-repository evaluation measures next-token prediction in bits per byte
(BPB): 0.222 average versus Kimi-K3's 0.193, with lower being better. This is not
an executable coding-task success rate; some comparisons mix base and post-trained
models. “Smallest in the family” does not establish that it is a small model at
our scale. Mirendil explicitly says fully autonomous recursive self-improvement
has not yet been reached.

### Recursive: automate the experiment loop; co-design the stack

[Introducing Flint](https://recursive.com/updates/introducing-recursive-flint)
(announced 2026-10-21; the page itself displays no publication date) describes a
40B-total/2.5B-active MoE coding model. Its Eureka Machine proposes, implements and
tests ideas, retains earlier discoveries and combines useful branches. Flint's
reported development took under four months. This is research-process improvement,
not evidence that deployed Flint continually updates its weights from conversations.

Concrete examples include searching expert configurations, quantizing the output
head to four bits, recovering quantization accuracy through offline distillation
from the 16-bit model, and generating hardware-specific kernels, including a Metal
decode megakernel. The reported NVFP4 result is within 0.7 benchmark points of the
16-bit version. Its reported short-context M4 Pro speed exceeds 125 tokens/second;
that is not an iPhone result or an end-to-end agent completion rate.

The earlier [First Steps Toward Automated AI Research](https://recursive.com/updates/first-steps-toward-automated-ai-research)
(2026-06-11, not today's release) is more relevant to small-model training. It
reports 0.9109 versus 0.9372 BPB in five-minute NanoChat trials evaluated on B200,
and 77.5 versus 79.7 seconds to the NanoGPT target loss on eight H100s. Research
search cost is separate from each candidate's training budget. The work combines
architecture, optimizer and systems changes; it does not isolate one universal trick.

One candidate worth studying is gated hashed bigram/trigram memory in attention's
value path. The post notes that such tables can add 1–2B sparse parameters to a
roughly 50M backbone: cheap lookup does not mean a small memory footprint.
The system also audits reward hacks and repeats evaluations. That validation is
part of the method, not optional bookkeeping.

### Flapping Airplanes: data efficiency and hardware flexibility

[Index Ventures' announcement](https://www.indexventures.com/perspectives/toward-more-efficient-ai-leading-flapping-airplanes-1b-series-a/)
(2026-10-21) describes work across kernels, pretraining, post-training and inference,
with megakernels enabling more flexible GPU use. This is an investor's account of
the research direction, not a disclosed training algorithm or controlled proof of
better sample efficiency. The company website did not expose readable research
text during this review; no reproducible new learning rule was verified.

For technical grounding, cofounder Benjamin Spector's earlier, public
[Hazy Research megakernel work](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
(2025-09-28; [code](https://github.com/HazyResearch/Megakernels)) explains overlapping
memory, compute and communication instead of executing coarse operations serially.
It reports over 22% higher end-to-end throughput than SGLang for its H100 Llama-70B
workload. This predates the current announcement and is inference research, not
evidence for Flapping Airplanes' current training or continual-learning claims.
See [the older megakernel results](megakernels-2025.md) for the March, May and
September releases, workload details, scheduling regressions and code references.

## Relevant experiments for this repository

Our [current plan](../plans/SIMPLE_LLM_PRETRAINING.md) uses a frozen 32K tokenizer,
a 124,389,632-parameter dense model and an initial 20B-exposure target, conditional
on corpus readiness. It excludes teacher distillation. Our [phone roadmap](../../inference/phone/README.md)
starts with correctness against a standard runtime before custom Metal optimization.
The following are our inferences and proposed follow-ups, not lab claims:

| Priority | Proposal | What would count as useful evidence? |
| --- | --- | --- |
| First | Use bounded, human-reviewed experiment loops on the existing trainer: one hypothesis, pinned inputs, fixed budget, saved result and next decision. | Repeat baseline and candidate across at least three seeds; retain failures; confirm the winner on an untouched test. Account for all search GPU time and agent costs. |
| First | Measure data efficiency separately from training speed. Keep the agreed mixture as the control; propose any mixture change explicitly. | Plot held-out quality against unique accepted tokens, total exposures, GPU-hours and cost. Preserve parent-level splits and contamination checks; faster tokens/second alone is insufficient. |
| Next | Evaluate the behavior we want, alongside language-model loss. | Report domain BPB using summed token NLL divided by evaluated UTF-8 bytes and ln(2), with consistent boundaries. Add held-out code execution and JSON/tool-format checks; do not infer instruction ability from base-model BPB. |
| Later | Test a strictly memory-capped n-gram lookup ablation. | Compare quality at equal time and equal memory, counting tables and optimizer state. Verify benefits beyond repeated local patterns and across our code/text domains. |
| Later | Profile complete Apple-device decode before selecting fusion or quantization work. | Measure correctness, prefill, decode latency, peak memory and sustained performance at matched contexts on the actual target device. Kernel microbenchmarks alone do not establish a user-visible gain. |

The transferable lesson is to improve what we learn from each experiment and each
unit of data while measuring the whole system. These sources do not justify
replacing the current dense baseline with MoE, adopting distillation, or launching
new training in this documentation change.
