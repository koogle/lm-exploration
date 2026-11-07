# Inkling 2 Small: lower-precision training and lessons for our KDA model

Reviewed 2026-11-07 against Thinking Machines' November 5 release and model card.
External results below are author-reported. Our comparison uses H100 FP8 and is
not a reproduction of their Blackwell NVFP4 recipe.

## Sources and what they establish

1. [Introducing Inkling 2 Small](https://thinkingmachines.ai/news/inkling-2-small/),
   especially **Training Inkling 2 Small**: hardware, native NVFP4 pretraining,
   curriculum, RL scale and systems changes.
2. [Inkling 2 Small model card](https://thinkingmachines.ai/model-card/inkling-2-small/):
   594B total / 17B active parameters, 8 of 384 experts per token, hybrid
   local/global attention, image inputs, up to 1M context, and MXFP8/NVFP4 support.
3. [torchao v0.18.0 float8 configurations](https://github.com/pytorch/ao/blob/v0.18.0/torchao/float8/config.py)
   and [linear conversion](https://github.com/pytorch/ao/blob/v0.18.0/torchao/float8/float8_linear.py):
   the public, inspectable implementation selected for our separate FP8 experiment.
   This is not Thinking Machines' implementation.

At review time the announcement describes initial closed-weight availability
through Tinker and an intended Apache-2.0 weight release the following week,
conditional on external testing. Do not label the intended release as completed.

## Reported training recipe

| Stage | Reported scale and method | Transferable lesson |
| --- | --- | --- |
| H1 | 8.4T tokens of natural language, code and images; 68 hours on GB300 NVL72 racks; native NVFP4 | Low precision can be part of pretraining, not only an inference export |
| H2 | 3.8T tokens of higher-quality, newer, unseen data at 512K context | Change data quality and context deliberately after foundation training |
| H3 | ~320B tokens of reasoning/agent trajectories at 1M context, alongside replayed pretraining data | Introduce deployment-format behavior with replay |
| H4 | ~34B tokens of highest-quality agent trajectories | Filter for correctness and alignment before final specialization |
| Joint RL | 1,075 steps, 17M+ rollouts, 3T+ tokens, 2,048 GPUs over nine days | Judge learner and rollout costs together; task-specific graders matter |

H3/H4 trajectories came mostly from earlier models, including Kimi K3 and their
own models. These are reported training-stage totals, not an available dataset
or an exact natural-language/code/math mixture we can reproduce. In particular,
the reported **65% coding share is RL tokens**, not a pretraining mixture weight;
coding is 38.5% of rollouts because lengths differ across task families.

They report effort conditioning and a length penalty to learn a speed/thoroughness
tradeoff, and StableReinforce score centering for asynchronous RL. The release
describes training reasoning behavior; it does not establish a separate dedicated
reasoning head. For us, retain the earlier distinction between formatting reasoning
traces and improving executable problem-solving ability.

## What the low-precision claim does—and does not—tell us

The central evidence is **native NVFP4 pretraining on Blackwell GB300**, together
with NVFP4 MoE training and sampling in the RL infrastructure. The post explicitly
says the NVFP4 recipe will be detailed in a follow-up. The release does not supply
a matched BF16/NVFP4 training ablation, isolated precision speedup, or sufficient
per-operation scaling/accumulation details to reproduce it. “Native NVFP4” does
not establish that optimizer states, reductions, recurrent states, or every
operation use four bits.

Our eight H100s have native FP8 tensor-core support, not Blackwell's native NVFP4
matrix throughput. FP4 storage or emulation on H100 would not reproduce that
hardware benefit. MXFP8 support in the model card also does not make its
Blackwell microscaling path interchangeable with Hopper FP8. The first practical
comparison on our hardware is BF16 versus selectively applied FP8.

The 68-hour figure is not a transferable estimate for eight H100s: the release
identifies GB300 NVL72 racks without specifying an H1 GPU count there, and the
model is a large MoE with high expert parallelism. Nor does a faster kernel
benchmark demonstrate an equal end-to-end training gain. The release itself
makes this distinction for its sparse-attention autoresearch demonstration.

## Proposed precision boundary for the H100 experiment

| Component | First comparison policy | Why |
| --- | --- | --- |
| Large MLP, attention projection and MTP fusion linears | torchao FP8 rowwise versus existing BF16 autocast | Largest plausible dense-GEMM opportunity using an established implementation |
| Small projection/gating maps | Existing precision | Conversion overhead and sensitivity may outweigh savings |
| KDA recurrent update/state and convolutions | Existing BF16/FP32 behavior | Long chains of updates need separate numerical evidence |
| Tied vocabulary output and MTP losses | Existing precision | Functional output path is not converted by replacing `nn.Linear`; report its cost as uncovered |
| Parameters, optimizer, gradient communication, normalization and loss reductions | Existing precision | Keep this an isolated compute-precision experiment |

Use FP32 parameters with transient FP8 operands in eligible linears, preserving
parameter identities and optimizer roles. The selected torchao recipe uses
rowwise dynamic scales, E4M3 operands and power-of-two scale rounding; verify
actual installed code/version in the run. FP8 operand formats do not imply FP8
accumulation or FP8 optimizer state. Keep compilation, microbatch, global batch,
initialization, data order and MTP policy fixed.

## Comparison and advancement criteria

The implementation belongs in the existing MFU work, [PR #150](https://github.com/koogle/lm-exploration/pull/150).
This reference PR captures the rationale independently of that implementation.

1. On the actual H100, compare outputs, input gradients and weight gradients of
   representative production linear shapes against BF16. Record relative L2
   errors and reject nonfinite results. A 10% relative-L2 ceiling is a coarse
   smoke-test rejection threshold, not a convergence or quality acceptance rule.
2. Use one allocation for **BF16 / FP8 / BF16**, each 30 steps on the same real
   50/40/10 mixture, 960M KDA, eight H100s, context 4096, microbatch six and global
   batch 256. Discard five startup steps per arm. Record all step times, loss,
   gradient norms, memory, per-rank phases, converted modules and software versions.
   Preserve failed checks and failed runs. Enforce the existing 30-minute worker
   timeout, no retries, and independent overall deadline supervision.
3. Report median step time and tokens/sec against both controls; inspect drift,
   spikes and full loss/gradient trajectories. A short run checks executability
   and gross numerical divergence, not final held-out quality or long-run stability.
4. If the speed gain exceeds control drift and numerical behavior remains sound,
   run a separately bounded matched-token BF16/FP8 learning-curve comparison with
   held-out code/math/text losses before promoting FP8 for pretraining. Evaluate
   a trained checkpoint too, since fresh initialization can hide outlier issues.

Use **milliseconds saved and cost per useful token** as the primary systems
outcomes. Our existing MFU estimator uses H100 dense BF16 peak; with FP8 enabled
it must be labeled **BF16-equivalent MFU**, not native FP8 utilization. A single
FP8 peak denominator for this mixed-precision model is also incomplete: substantial
BF16 recurrence, loss and optimizer work remains. Quantization/scaling overhead,
extra saved tensors and tensor-core kernels must all be included in end-to-end
measurements.

The previous KDA control needs roughly 530 ms/step saved to reach 40% under its
current BF16-equivalent estimator. FP8 can address eligible matrix work, but does
not automatically accelerate recurrence, elementwise work, synchronization or
CPU dispatch. Earlier hypothetical speedup arithmetic is a planning scenario,
not an Inkling result or a measurement of our implementation.

## Decisions for our model

Keep KDA and the 50/40/10 foundation mixture while testing precision in isolation.
Inkling supplies no source proportions that justify changing those weights.
Borrow its broad-to-specialized curriculum, correctness-filtered trajectories,
and explicit speed/quality evaluation rather than its MoE scale or context
length. Its DRAM KV-cache offload is a rollout-serving technique for prefix reuse;
it is not evidence that CPU offload will speed up our training-step activations.
No production precision change or large pretraining launch follows from this
reading alone. Comparison results should be appended with their run IDs and
provenance, including negative outcomes.

## Our completed eight-H100 comparison

Run `kda-fp8-linears-h100x8-20261107-02`, source `e1c9a4b`, used the protocol
above and converted 183 large linear layers (707,002,368 weight elements).
[Raw results, numerical checks, per-step logs and provenance](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/sub1b-pilot/real-data-mfu-20261105/deep-dive/kda-fp8-linears-h100x8-20261107-02)
are archived together. Each arm completed 30 steps / 31.46M window tokens;
25 post-warmup steps contribute to these medians.

| Arm | Median step | Tokens/sec | BF16-equivalent MFU | Peak allocation/GPU |
| --- | ---: | ---: | ---: | ---: |
| BF16 first | 2.918 s | 359,347 | 33.05% | 65.50 GiB |
| FP8 rowwise linears | 3.612 s | 290,298 | 26.79% | 59.88 GiB |
| BF16 repeat | 2.918 s | 359,408 | 33.12% | 65.50 GiB |

FP8 reduced throughput **19.22–19.23%**, while the controls differed by only
**0.017%**. It saved **5.62 GiB/GPU** but added approximately **694 ms/step**.
All 90 steps had finite main/MTP losses and gradient norms. Representative
linear output/input-gradient/weight-gradient errors were 3.57–3.78% relative L2.
Final main training losses were 11.33825 / 11.35398 / 11.33803 respectively; the
maximum matched-step FP8-versus-first-control main-loss difference was 0.01573.
These are early training losses during the 500M-token learning-rate warmup,
not held-out quality or evidence of equivalent long-run convergence.

The phase medians locate the regression: local backward increased from **1,301
to 1,900 ms**, synchronizing backward from **153 to 202 ms**, and forward/loss
from **1,346 to 1,391 ms**. Optimizer time stayed approximately 105 ms. The
synchronizing-backward phase includes computation and is not pure communication;
phase medians are independently summarized and not an exact additive ledger.

**Decision: retain BF16.** This measured rowwise FP8 implementation fails the
speed criterion, despite passing the coarse numerical smoke check and saving
memory. Do not spend the longer convergence-test budget on this candidate yet.
The source performs dynamic rescaling/casting for backward operands along
different axes; that is a plausible investigation target, but phase timings alone
do not separate its cost from GEMM kernel choice. A targeted backward trace or
shape benchmark is needed before claiming a cause or implementing a fix.
A larger microbatch enabled by the saved memory is another untested possibility,
not a measured gain. This result rejects this configuration, not the general
possibility of efficient FP8 training or Thinking Machines' different NVFP4 stack.
