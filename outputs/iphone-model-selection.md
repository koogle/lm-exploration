# Small models for a hand-written iPhone Metal runtime

Research date: 2026-10-08. Target: iPhone 17 Pro, native execution, user-written
Metal compute kernels. This is a source-based selection, not an on-device test.
No model weights were downloaded and no inference experiments were run.

## Recommendation

**Start with `HuggingFaceTB/SmolLM2-135M-Instruct` in FP16.** It is the smallest
instruction-tuned model I would choose for this particular kernel-learning
project, not a claim that it is the smallest language model in existence.
Its conventional Llama-style decoder is a better first implementation target
than a hybrid architecture. Its weakness is output quality: do not expect a
reliable coding assistant, factual QA, or complex reasoning.

Use **SmolLM2-360M-Instruct** as the next checkpoint for the same architecture.
For a small, more practically task-oriented candidate, investigate
**LFM2.5-230M**, not just the older 350M option. Liquid's current catalog and
release post confirm that 230M is a generative, post-trained text model, distinct
from its similarly named encoder models. The catch is extra convolution/state
logic, and I found no iPhone 17 Pro Metal benchmark for it in the reviewed sources.

**Qwen3-0.6B** is the strongest directly benchmarked iPhone 17 Pro speed reference
in this shortlist. It is still small enough to be a plausible later target,
but has additional Q/K normalization and a substantially larger vocabulary.
Disable thinking through its actual chat template when measuring visible-answer
latency; do not count hidden reasoning as useful answer throughput.

Treat **30 decode tokens/sec** as an initial interactive performance goal and
**60 tokens/sec** as a stretch goal, not predictions for an unimplemented engine.
A correct but naive implementation can be much slower than mature runtimes.

## Comparison matrix

Weight figures are decimal MB estimates from nominal parameter count: FP16 =
2 bytes/parameter; packed Q4 = 0.5 bytes/parameter. They are NOT checkpoint file
sizes or peak runtime RAM. Q4 scales, mixed-precision tensors, tokenizer,
KV cache, activations, scratch buffers, and copies add overhead.

| Model | Nominal parameters | FP16 weights | Packed Q4 weights | Fit for this project |
| --- | ---: | ---: | ---: | --- |
| SmolLM2-135M-Instruct | 135M | 270 MB | 68 MB | Best tiny first target; conventional architecture; weak general capability |
| LFM2.5-230M | 230M | 460 MB | 115 MB | Small task-oriented candidate; hybrid convolution/attention; no reviewed iPhone speed measurement |
| Gemma 3 270M IT | 270M | 540 MB | 135 MB | Compact task-specific option; 256K vocabulary is relatively large for a first runtime |
| LFM2.5-350M | 350M | 700 MB | 175 MB | Task-oriented candidate with Apple-phone GPU evidence; still requires hybrid kernels |
| SmolLM2-360M-Instruct | 360M | 720 MB | 180 MB | Natural upgrade using the same SmolLM2 kernel family |
| Qwen3-0.6B | 600M | 1,200 MB | 300 MB | Direct iPhone 17 Pro benchmark evidence; larger vocabulary and Q/K normalization |

The existing GPT-2 file can remain a reference artifact. Its base-model behavior
and mixed F32/Q4_0/Q6_K tensors make it a less attractive first instruction-model
implementation than a single-precision SmolLM2 checkpoint.

### Architecture and memory facts

Official configs show:

- SmolLM2-135M: 30 layers, hidden width 576, 9 query heads, 3 KV heads,
  64-dimensional heads, 49,152-token vocabulary, tied embeddings.
- SmolLM2-360M: 32 layers, hidden width 960, 15 query heads, 5 KV heads,
  64-dimensional heads, the same vocabulary size and Llama model type.
- LFM2.5-230M: 14 layers, 8 convolution and 6 full-attention layers, width 1,024,
  65,536-token vocabulary, and convolution cache length 3.
- LFM2.5-350M: 16 layers, 10 convolution and 6 full-attention layers.
- Qwen3-0.6B: 28 layers, 16 query heads, 8 KV heads, 128-dimensional heads,
  151,936-token vocabulary.

A standard unquantized FP16 KV cache is estimated by
`2 × attention_layers × KV_heads × head_dim × context_tokens × 2 bytes`.
At 1,024 cached tokens this gives approximately 23.6 MB for SmolLM2-135M,
41.9 MB for SmolLM2-360M, 12.6 MB for either shortlisted LFM model's attention
layers (excluding convolution state), and 117.4 MB for Qwen3-0.6B.
This is cache storage only, not measured total memory.

Small parameter count is not a speed guarantee. SmolLM2-135M still has 30
sequential layers, so dispatch/synchronization overhead can matter even when
weights are small. Tokenizers also differ: equal tokens/sec is not necessarily
equal words/sec.

## Published speed evidence

These are author-reported measurements, checked against the linked source, not
independently reproduced results. Different devices, runtimes, quantization,
contexts, and cooling protocols prevent a fair ranking across rows.

| Model | Device/runtime | Reported decode | Scope and caveats |
| --- | --- | ---: | --- |
| SmolLM2-135M | iPhone 15 Pro, Optimum ExecuTorch | 29.64 tok/s | Official project README; custom SDPA/KV-cache and 8da4w quantization. Not iPhone 17 Pro or proof of custom Metal performance. |
| Qwen3-0.6B, Q4 | iPhone 17 Pro, Russet/MLX | 69.7 short; 62.2 medium; 43.4 long tok/s | Fixed outputs of 128/256/512 tokens; roughly 47/279/690 input tokens; 2 warmups, 10 measured runs; fresh cache; GPU decode excludes text/UI overhead; device cooled between model runs. |
| Qwen3-0.6B, Q4 | iPhone 17 Pro, apple-silicon-llm-bench / mlx-swift | 178.8 warm tok/s | Leaderboard capture 2026-08-26; short-chat protocol; warm median of 2 runs for this capture. A separate protocol, not evidence that the longer Russet test should reach this rate. |
| LFM2.5-350M, BF16 | Apple A18 Pro GPU, Mirai | 73 tok/s | Liquid's publisher benchmark: 1K prompt, 100 decoded tokens. Not A19 Pro, not our kernels. |
| LFM2.5-230M, Q4 | Galaxy S25 Ultra CPU | 213 tok/s | Liquid's release; 2K input context. Useful evidence of model efficiency, not an Apple GPU estimate. |

No credible exact iPhone 17 Pro / hand-written Metal speed measurement for
SmolLM2-135M or LFM2.5-230M was established in this review. Do not fill that gap
with extrapolated numbers. Parameter count and nominal memory are sufficient to
justify trying them, but actual tokens/sec must be measured on the phone.

Liquid reports IFEval 71.71 for 230M and 76.96 for 350M in its comparison.
Hugging Face reports IFEval (average prompt/instance) 29.9 for SmolLM2-135M and
41.0 for SmolLM2-360M. These are separate evaluation reports, not a controlled
cross-family comparison. They support task evaluation, not a blanket claim that
one model is a reliable general assistant. Google explicitly describes Gemma
270M as not designed for complex conversational use.

## Implementation direction

1. Implement one SmolLM2 architecture, not a general multi-model framework.
   Keep dimensions explicit/config-driven so 360M can reuse the kernels later.
2. Export/convert the pinned checkpoint to FP16 for the first GPU implementation.
   BF16 storage and FP16 storage are not interchangeable bit patterns.
   Defer packed-weight formats until the full forward pass is correct.
3. Validate matrix-vector projection, RMSNorm, RoPE, grouped-query causal
   attention/softmax, SwiGLU, residuals, and output projection against a CPU
   reference using the same weights. Verify tokenizer and chat template too.
4. Implement incremental decode with persistent weights and a KV cache.
   Avoid rebuilding pipelines or re-uploading weights per token.
5. Start with greedy decoding. Keep intermediate buffers on the GPU; reduce the
   final logits to a token ID there rather than inserting CPU synchronization
   between every operation. Batch dispatches into sensible command buffers.
6. Only after correctness: fuse suitable operations, reduce dispatch overhead,
   and implement one well-defined packed Q4 matrix-vector path if beneficial.

This preserves the user's goal: our inference kernels are Metal, not a Core ML
conversion. MLX Swift or llama.cpp can be separate reference baselines without
becoming the production implementation.

## Phone benchmark plan

- Record device, iOS version, app/build mode, model revision, weight format,
  context length, and sampling settings.
- Measure load time, prompt prefill, time to first token, and decode separately.
  Include end-to-end displayed-text throughput as a separate number.
- Use fixed 128/512/1,024-token prompts and fixed 256-token output budgets;
  for throughput-only runs, ignore EOS but mark this explicitly.
- Use 2 warmups and 10 measured runs, resetting KV/convolution state each run.
  Report median and spread, plus a sustained multi-minute test without artificial
  cooling. Log thermal state and exclude unrelated background GPU work.
- Measure peak process memory, not just weight file size. Test both FP16 and Q4
  once both are available; use identical prompt token counts for each model.
- Score a small task set alongside speed: straightforward instructions,
  extraction, rewriting, and continuation. Show actual outputs so fast nonsense
  does not count as success.

## Sources and provenance

All sources accessed 2026-10-08. Model configs/cards were read at the then-current
`main` revision. Pin these revisions before exporting weights:

| Official Hugging Face model ID | Observed repository revision |
| --- | --- |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | `a10cc1512eabd3dde888204e902eca88bddb4951` |
| `LiquidAI/LFM2.5-230M` | `40cb2ad3b3044d5a41eee083a6103c8b523afa45` |
| `LiquidAI/LFM2.5-350M` | `9e6c6ccf47cd318696e137d381a7ded8fe4df09f` |
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |

- [SmolLM2-135M official card/config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct): architecture, capability limitations, Apache-2.0 license, evaluation.
- [SmolLM2-360M official card/config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct): same architecture family and evaluation.
- [Qwen3-0.6B official card/config](https://huggingface.co/Qwen/Qwen3-0.6B): configuration, Apache-2.0 license, and `enable_thinking=False` chat-template usage. [Transformers Qwen3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) confirms Q/K normalization.
- [LFM2.5-230M official card/config](https://huggingface.co/LiquidAI/LFM2.5-230M) and [release](https://www.liquid.ai/blog/lfm2-5-230m): hybrid architecture, post-training, publisher evaluation and CPU speed measurements.
- [LFM2.5-350M release](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind) and [official config](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/config.json): Apple A18 Pro GPU benchmark and architecture. Both Liquid model cards identify a custom LFM license; review its terms before redistribution/commercial use.
- [Liquid's current model catalog](https://www.liquid.ai/models): distinguishes generative 230M from encoder/embedding products.
- [Google's Gemma 3 270M announcement](https://developers.googleblog.com/introducing-gemma-3-270m/), 2025-08-14: 170M embedding parameters, 100M transformer parameters, 256K vocabulary, intended task-specific use.
- [Optimum ExecuTorch README](https://github.com/huggingface/optimum-executorch/blob/main/README.md): SmolLM2-135M iPhone 15 Pro decode table and quantization caveats.
- [Ricky Takkar's Russet iPhone 17 Pro benchmark](https://rickytakkar.com/blog_russet_mlx_benchmark.html), 2026-02-08: Qwen3-0.6B decode speeds and detailed test methodology. Its LFM measurements concern **1.2B**, not 230M/350M.
- [apple-silicon-llm-bench leaderboard](https://github.com/john-rocky/apple-silicon-llm-bench/blob/main/LEADERBOARD.md) and [fairness rules](https://github.com/john-rocky/apple-silicon-llm-bench/blob/main/methodology/fairness-rules.md): 178.8 tok/s Qwen3 capture, runtime build `60bd0d7880c82980f9481f8be78862e9b63c58a3`, captured 2026-08-26. This is a mutable community leaderboard, not a vendor guarantee.
