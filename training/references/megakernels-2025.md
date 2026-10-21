# Older megakernel results: what transfers to our models?

Reviewed 2026-10-21. These are Hazy Research author-reported **inference** results,
not reproduced measurements here. They precede the current Flapping Airplanes
announcement; none establishes better sample efficiency or faster training.

## Three stages of the work

| Release and scope | Workload and comparison | Reported result |
| --- | --- | --- |
| [ThunderMLA, 2025-03-04](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla): attention decode only | SXM H100; four prompts of lengths 4,641 / 45,118 / 1,730 / 1,696; four query tokens; 16 heads/GPU in an eight-way tensor-parallel configuration | 41 µs versus FlashMLA's 52 µs: 1.27× speed, or 21% less latency (calculated). This is a kernel measurement, not whole-model throughput. |
| [Look Ma, No Bubbles!, 2025-05-27](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles): whole Llama-3.2-1B forward pass | BF16, batch one, 32-token prompt and 128 generated tokens, no speculative decoding; baselines use CUDA graphs and compilation | H100: under 1 ms, 78% bandwidth utilization, nearly 2.5× vLLM and over 1.5× SGLang. B200: under 680 µs, over 3.5× vLLM and over 1.5× SGLang. |
| [Whole GPU, 2025-09-28](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main): Llama-70B serving | Eight H100s, integrated into Tokasaurus; 65,536 ShareGPT prompts with sampled completion lengths | Total throughput 23,468 versus SGLang's 19,170 tokens/s; output throughput 9,043 versus 7,387. Approximately 22.4% higher throughput. |

### March: scheduling overhead is part of the result

ThunderMLA fuses partial attention and reduction through a GPU interpreter.
Its CPU schedulers take 1–2 ms, so reuse across layers/steps and overlap with GPU
execution are essential. The additional workload table spans roughly 7.6–36.3%
speedups; the headline is not a guarantee for every shape. For us, this motivates
counting host scheduling and cache reuse in any fused-attention experiment.

### May: the closest match to small-model local inference

The implementation overlaps the next instruction's loads with current work,
pages scarce shared memory, and uses explicit dependency counters. Removing
kernel boundaries also removes implicit synchronization, increasing correctness
work. The B200 runtime analysis still attributes substantial time to activation
movement and synchronization. This supports profiling stalls beyond raw weight
bandwidth. H100/B200 results cannot predict Metal performance or long-context speed.

### September: useful ablations, including a regression

On the separate short-cookie-joke workload at batch 8,192, removing the global
work queue drops decode throughput from 31,516 to 27,033 tokens/s. At batch 1,024,
the queue instead loses: 18,320 versus 18,676 without it. This is evidence that
dynamic scheduling can be the wrong choice for small workloads. These ablations
are not the ShareGPT serving benchmark. The authors describe training megakernels
as future work and warn that the released research code is sensitive to toolchains
and GPU configuration.

## Code references and reproducibility gaps

The two Llama releases use different branches. Inspected branch heads below are
pinned as of this review, **not verified commits used for the published numbers**:

- [Low-latency source, `main` at `7309cec`](https://github.com/HazyResearch/Megakernels/tree/7309cec801537b61fea3b50d7dfe454a6cde578e).
  Its [README](https://github.com/HazyResearch/Megakernels/blob/7309cec801537b61fea3b50d7dfe454a6cde578e/README.md)
  documents CUDA 12.8 PyTorch wheels, recursive submodules, H100/B200 compilation,
  and `megakernels/scripts/generate.py`. The example uses 100 generated tokens;
  it does not reproduce the article's 32/128 protocol unchanged.
- [Throughput source at `91eaff2`](https://github.com/HazyResearch/Megakernels/tree/91eaff262c2b473cfdcb135f5f2abefbe2835fe9).
  Its [README](https://github.com/HazyResearch/Megakernels/blob/91eaff262c2b473cfdcb135f5f2abefbe2835fe9/README.md)
  offers research-reference code without a supported reproduction walkthrough.
- ThunderMLA's [original code link](https://github.com/HazyResearch/ThunderKittens/blob/mla/kernels/attn/demo/mla_decode/template_mla_decode.cu)
  currently fails: GitHub reports no `mla` ref. The article remains readable,
  but that release's code would need recovery from history before reproduction.

Before claiming reproduction, pin model weights, submodules, compiler/CUDA and
baseline versions, GPU topology/clocks, prompt lengths, batching, warm-up and timing
boundaries. Check outputs independently and account for CPU preparation. Source
inspection and published timing tables are the evidence collected here; no GPU
benchmarks or third-party code were executed.

## Proposed application here

For the [Apple inference work](../../inference/phone/README.md), start with batch-one
profiling, then compare separate kernels, modest fusion and a larger fused block.
Use our model's actual shapes and matched precision/context. Keep full-model
latency, peak memory, correctness and sustained device performance as acceptance
criteria; a fast inner kernel is insufficient. Only attempt a full-model persistent
kernel after smaller experiments show launch or dependency stalls dominate.

For [training](../README.md), retain standard kernels initially. A separate future
experiment would need forward **and backward**, optimizer updates and equal-quality
training comparisons; none of these inference speedups supplies that evidence.
