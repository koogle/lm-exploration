# Quantization for LLM Inference Speed (PTQ, QAT, KV cache): state as of Nov 2026

Scope notes: I made about 15 search/fetch calls. Where I only saw a search snippet or a secondary write-up, the item says so. "Background" marks work from before 2025. Dates come from the sources themselves; I flag any I could not confirm.

## 1. Low-precision formats in practice, and which open models ship natively in them

### Takeaway
By late 2026 the default deployment formats are FP8 (W8A8) and 4-bit block-scaled floats. MXFP4 has 32-element blocks with E8M0 power-of-two scales, about 4.25 bits/weight. NVFP4 has 16-element blocks with E4M3 scales, about 4.5 bits/weight. NVFP4 is the more accurate of the two. Several flagship open models now ship in low precision from the lab itself: gpt-oss uses MXFP4 for its MoE weights, Kimi K2 Thinking uses INT4 weights via QAT, and DeepSeek V3 uses block FP8. The main benefits are memory capacity (fitting on one GPU) and decode bandwidth. Real FP4 compute speedups need Blackwell-class hardware.

### Cited Findings
- **gpt-oss (Aug 2025):** the MoE weights are post-trained with MXFP4 quantization, which lets gpt-oss-120b run on a single 80 GB GPU (H100/MI300X) and gpt-oss-20b fit in 16 GB — [HF model card](https://huggingface.co/openai/gpt-oss-120b)
- MoE weights are over 90% of gpt-oss parameters and are stored at about 4.25 bits/parameter in MXFP4 — [Ollama blog](https://registry.ollama.ai/blog/gpt-oss)
- A proposed edit to the card would call the precision "native MXFP4" for the MoE layers. I could not confirm it was merged, so "post-trained with MXFP4" is the official wording — [HF discussion](https://huggingface.co/openai/gpt-oss-20b/discussions/96/files)
- **Kimi K2 Thinking (Nov 2025):** "native INT4", which Moonshot describes as giving lossless reductions in inference latency and GPU memory. All reported benchmark scores were measured at INT4 precision — [HF model card](https://huggingface.co/moonshotai/Kimi-K2-Thinking)
- K2 Thinking applies INT4 weight-only QAT to the MoE components during post-training. Moonshot claims about a 2x generation speedup in low-latency mode — [DeepInfra listing](https://deepinfra.com/moonshotai/Kimi-K2-Thinking) (vendor restating Moonshot's figure; I found no independent check)
- The K2 Thinking HF release is 594 GB, versus 1.03 TB for the original Kimi K2 — [Simon Willison](https://simonwillison.net/2025/Nov/6/kimi-k2-thinking)
- **DeepSeek-V3 FP8 (background, Dec 2024):** weights are quantized in 128x128 blocks and activations in 1x128 tiles (fine-grained FP8) — [DeepSeek-V3 hardware reflections, arXiv 2505.09343](https://arxiv.org/pdf/2505.09343). The checkpoint stores float32 `weight_scale_inv` with block size [128,128] — [DeepSeek-V3 repo docs](https://docsearch.algolia.com/mcp/docs/repo/deepseek-ai/deepseek-v3)
- UE8M0 (power-of-two) FP8 scales are attributed to DeepSeek-V3.1, aimed at domestic Chinese accelerators, not to the original V3 — [cnyes article (Chinese)](https://hao.cnyes.com/post/189652) (secondary)
- **NVFP4 vs MXFP4 format:** NVFP4 uses block size 16 with FP8 E4M3 scales. MXFP4 uses block size 32 with E8M0 power-of-two scales — [Unsloth NVFP4 docs](https://unsloth.ai/docs/basics/nvfp4); confirmed by the MLX format table (mxfp4 group 32/E8M0, nvfp4 group 16/E4M3, mxfp8 group 32/E8M0) — [MLX quantize docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- Under W4A4 both formats are lossy, but MXFP4 loses much more accuracy. MR-GPTQ, a format-specialized GPTQ variant, brings MXFP4 to within 1-2% of NVFP4. The gap is partly the bits per element (4.5 for NVFP4 vs 4.25 for MXFP4) — [MR-GPTQ, arXiv 2509.23202](https://arxiv.org/pdf/2509.23202v1)
- MR-GPTQ's own FP4 kernels give 3.6x layer-wise and 2.2x end-to-end speedups on B200, and 6x layer-wise and 4x end-to-end on RTX 5090 — [arXiv 2509.23202](https://arxiv.org/pdf/2509.23202v1)
- **NVFP4 for pretraining (Sep 2025):** MXFP4 has about 2.5% relative error vs about 1.5% for NVFP4. MXFP4 matches NVFP4 loss only with about 36% more training tokens — [NVIDIA, Pretraining LLMs with NVFP4, arXiv 2509.25149](https://arxiv.org/pdf/2509.25149)
- Hopper (H100) has no FP4 tensor cores, so FP4 there is emulated or weight-only and gives no FP4 compute speedup — [Medium blog, secondary](https://medium.com/ai-actually/nvfp4-doubles-your-inference-on-blackwell-for-under-1-accuracy-loss-a74f7f023ea4)
- **Measured NVFP4 W4A4 serving (Qwen3.6-35B-A3B, vLLM, 4x GB200, AIPerf):**

  | Shape | BF16 tok/s | W4A16 tok/s | W4A4 tok/s | W4A4 vs BF16 |
  |---|---|---|---|---|
  | decode 128/2048, conc 128 | 17,636 | 15,222 | 20,076 | 1.14x |
  | chat 8000/1000, conc 32 | 3,649 | 2,373 | 4,079 | 1.12x |
  | prefill 32000/400, conc 128 | 854 | 1,092 | 1,107 | 1.30x |

  - W4A4 beats BF16 in 9 of 12 shapes.
  - W4A16 is slower than BF16 in 10 of 12 shapes.
  - At concurrency 1, W4A4 runs at 0.87-0.88x of BF16.
  - The checkpoint shrinks from 67 GiB to 22 GiB.

  Source: [NVIDIA ModelOpt Qwen3.6 QAD post, 2026-09-16, updated 2026-10-21](https://nvidia.github.io/Model-Optimizer/announcements/20260916-qwen36-w4a4-qad.html)
- **Unsloth "Dynamic NVFP4":** keeps sensitive layers in FP8 or BF16 and quantizes the rest to W4A4. It covers Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B, Qwen3.8-27B, GLM-4.7-Flash and the Gemma 4 family, and needs a Blackwell GPU. Claimed speedups vs BF16:

  | Model | Speedup |
  |---|---|
  | Gemma 4 E2B | 1.12x |
  | Gemma 4 E4B | 1.22x |
  | Gemma 4 12B | 1.26x |
  | Gemma 4 26B-A4B | 1.41x |
  | Gemma 4 31B | 1.45x |
  | Qwen3.5-122B-A10B (B200, conc 128) | 1.38x |

  Decode-only gains are smaller: 1.03x for 27B and 1.17-1.22x for 35B — [Unsloth NVFP4 docs](https://unsloth.ai/docs/basics/nvfp4) (vendor numbers)
- **Unsloth accuracy table** (MMLU-Pro / GPQA / AIME25). NVFP4 is within noise of BF16:

  | Model | Unsloth NVFP4 | FP8 | BF16 |
  |---|---|---|---|
  | Qwen3.6-27B | 86.25 / 86.34 / 93.12 | 86.11 / 86.87 / 93.75 | 85.96 / 88.13 / 93.33 |
  | Qwen3.6-35B-A3B | 85.85 / 86.74 / 92.29 | 85.75 / 86.74 / 93.12 | 85.75 / 86.36 / 92.50 |

  Unsloth reports output lengths comparable across quants — [Unsloth NVFP4 docs](https://unsloth.ai/docs/basics/nvfp4)
- vLLM kernel choice matters a lot for FP4. Decode tok/s by backend:
  - Qwen3.6-27B: cute-DSL (auto) 125.9, CUTLASS 113.5, FlashInfer-TRTLLM 112.6, Marlin 105.6.
  - Qwen3.6-35B-A3B: cute-DSL 295.2 vs CUTLASS 158.3.
  - Forcing Marlin makes W4A4 MoE about 2.5x slower. DGX Spark needs `--moe-backend flashinfer_b12x`.

  Source: [Unsloth NVFP4 docs](https://unsloth.ai/docs/basics/nvfp4)

### Inferences
- **Weight-only 4-bit (W4A16) helps decode at low batch.** It cuts memory traffic, but at high concurrency it can be slower than BF16, as the ModelOpt table shows. Gains at high batch and in prefill need activation quantization (W8A8 FP8 or W4A4 NVFP4) on hardware with matching tensor cores.
- **Native low-precision releases change what you serve.** With gpt-oss MXFP4 and Kimi INT4, the low-precision checkpoint is the reference: benchmarks are reported on it, and upcasting gains nothing.
- **For Apple Metal:** FP4 tensor-core arguments do not transfer. On a phone the benefit is purely bandwidth and memory, so weight-only formats (4-bit affine, mxfp4) are the relevant ones.

### Gaps
- I did not verify specific Qwen FP8 or Nemotron NVFP4 official releases, for example which Qwen3.x and Nemotron 3 checkpoints are published in FP8 or NVFP4 by the labs themselves. Unsloth's table references an "NVIDIA" NVFP4 Qwen3.6 quant, but I did not open that card.
- I found no independent benchmark of Kimi K2 Thinking's claimed 2x INT4 speedup.
- I did not check whether DeepSeek V3.2 or later changed formats.

## 2. QAT vs PTQ: recent recipes, GPTQ/AWQ successors, 2-bit/ternary

### Takeaway
For 4-bit, the 2025-2026 trend is to distill against the full-precision model rather than run plain QAT on task labels:
- Gemma QAT uses the BF16 model's output probabilities as targets.
- NVIDIA's QAD uses KL divergence to a BF16 teacher.
- Kimi uses weight-only fake-quant QAT with a straight-through estimator.

Plain PTQ is already near-lossless at W8A8 and W4A16. QAT/QAD earns its cost mainly at W4A4 and for small models. Below 4 bits, trellis/vector codebooks (QTIP/EXL3) and natively ternary training (BitNet) are the serious options.

### Cited Findings
- **NVIDIA QAD (arXiv 2601.20088, Jan 2026, v3 Mar 2026):**
  - Distills a BF16 teacher into an NVFP4 student with a KL loss, and recovers near-BF16 accuracy on AceReason Nemotron, Nemotron 3 Nano, Nemotron Nano V2 / V2 VL and Llama Nemotron Super v1 — [arXiv abstract](https://arxiv.org/abs/2601.20088)
  - The authors say QAD is robust to data quality and coverage, so the full training data is not needed. It is more stable than QAT for models trained through multi-stage SFT, RL and model merging — [arXiv abstract](https://arxiv.org/abs/2601.20088)
  - QAD significantly outperforms QAT on hard reasoning benchmarks (AIME25, GPQA-D) for SFT-heavy models like Nemotron Nano V2. For the VLM, QAT is sometimes comparable. KL beats MSE as the loss, and the original BF16 model is a better teacher than a larger model from the same family — [NVFP4 QAD report (search summary)](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)
- **QAD case study, Qwen3.6-35B-A3B (W4A4 NVFP4):**
  - PTQ alone took 9 minutes. It lost 2.6 pp on IFBench and 1.2 pp on MMMU-Pro, with no deficit beyond noise on GPQA-D, SciCode, AA-LCR or tau2-bench.
  - 500 QAD iterations (32K sequence length, 128 GB200, 5.7 h, about 735 GPU-hours) recovered IFBench to -0.3 pp. MMMU-Pro was only about 40% recovered (-0.7 pp).
  - One benchmark showed a 3 pp swing at 3 repeats that disappeared at 8 repeats.

  Source: [ModelOpt post](https://nvidia.github.io/Model-Optimizer/announcements/20260916-qwen36-w4a4-qad.html)
- **The same study found QAD side effects (useful negative results):**
  - SciCode completions got 91% longer with an unchanged score.
  - Runs hitting the 131K token cap rose from 0.7% (BF16) to 3.6% (QAD).
  - On AA-Omniscience, the hallucination rate rose from 41% (BF16) to 49% (PTQ) to 61% (QAD), because the model abstained less with the same number of correct answers.

  Source: [ModelOpt post](https://nvidia.github.io/Model-Optimizer/announcements/20260916-qwen36-w4a4-qad.html)
- **Kimi K2 Thinking:** weight-only INT4 QAT with fake quantization and a straight-through estimator. The team chose QAT over calibration-based PTQ for long-context reasoning stability — [Zhihu engineer thread via Threadreader](https://threadreaderapp.com/thread/1987125624599970218.html) (secondary)
- **Gemma 3 QAT (Apr 2025):** about 5,000 QAT steps using the non-quantized checkpoint's probabilities as targets. Google says this "reduced the perplexity drop by 54%" vs PTQ at llama.cpp Q4_0. This is a relative cut in degradation, not a perplexity improvement — [Simon Willison link post](https://simonwillison.net/2025/Apr/19/gemma-3-qat-models/); quoted wording on [HN thread](https://remix-clone-hacker-news.flameddd1.workers.dev/item/48417089). I did not fetch Google's original post.
- The official Gemma 3 QAT Q4_0 GGUF keeps embeddings in fp16. A community requant to 15.6 GB, vs 17.2 GB official, has essentially the same perplexity (8.2335 vs 8.2323) — [stduhpf HF card](https://huggingface.co/stduhpf/google-gemma-3-27b-it-qat-q4_0-gguf-small)
- **Gemma 4 QAT (released 2026-06-05 per blog):** includes an official Q4_0 GGUF. In one user's Apple Silicon / Ollama test of Gemma 4 12B:
  - QAT Q4_0 ran at 42.9 tok/s, a non-QAT unsloth Q4_0 at 38.4, and Q4_K_M at 33.2. The speed comes from the Q4_0 format, not from QAT.
  - Task quality was tied (4.5/6 each). The author did not reproduce Google's perplexity claim.

  Source: [3h4x blog](https://3h4x.github.io/tech/2026/06/08/gemma4-qat-vs-non-qat) (single-user measurement)
- **Rotation methods (background, 2024):**
  - QuaRot rotates hidden states to remove outliers while leaving outputs unchanged — [Microsoft Research](https://www.microsoft.com/en-us/research/?p=1054284)
  - SpinQuant learns the rotations. At W4A4KV4 it cuts the gap to full precision to 2.9 points for LLaMA-2 7B (zero-shot), and shrinks QuaRot's gap by 30.2% (LLaMA-2 7B) and 34.1% (LLaMA-3 8B) — [SpinQuant HF paper page](https://huggingface.co/papers/2405.16406); [repo](https://github.com/recogni/SpinQuant)
  - llm-compressor ships a `SpinQuantModifier` — [llm-compressor docs](https://aireilly-llm-compressor.readthedocs.io/en/latest/api/llmcompressor/modifiers/transform/spinquant/base/)
- **MR-GPTQ (2025):** a GPTQ variant specialized for FP4 block formats, combining rotations with format-aware rounding (see Section 1 for numbers) — [arXiv 2509.23202](https://arxiv.org/pdf/2509.23202v1)
- **QTIP / EXL3:**
  - QTIP (background, 2024) uses trellis-coded quantization plus incoherence processing. At 2 bits on Llama-2-70B it reaches Wikitext2 perplexity 3.87 vs 3.12 for FP16, and 4.16 for QuIP# — [QTIP arXiv 2406.11235](https://arxiv.org/html/2406.11235v4); [summary](https://www.themoonlight.io/review/qtip-quantization-with-trellises-and-incoherence-processing)
  - ExLlamaV3's EXL3 is a streamlined QTIP derivative. On an RTX 5090 it gives better quality at matched size and much faster prompt processing than GGUF, while GGUF loads and decodes faster and runs on more hardware — [Atomic Chat comparison](https://atomic.chat/blog/guides/exl3-vs-gguf) (secondary; perplexity numbers not in the excerpt)
  - Methods that hold up at 4-bit can degrade significantly at 2-bit — [Bielik 2-bit study, arXiv 2603.04162](https://arxiv.org/html/2603.04162)
- **BitNet:**
  - BitNet b1.58 2B4T is an open native ternary-weight, 8-bit-activation model trained from scratch on 4T tokens — [DEV Community summary](https://dev.to/bspann/bitnet-microsofts-1-bit-llms-that-run-on-your-cpu-20h8)
  - bitnet.cpp (background, Oct 2024) is 1.37-5.07x faster on ARM CPUs with 55.4-70.0% less energy, and 2.37-6.17x faster on x86 with 71.9-82.2% less energy, across 125M-100B models — [bitnet.cpp, arXiv 2410.16144](https://export.arxiv.org/pdf/2410.16144)
  - Through HF transformers there are no speed benefits, because the specialized kernels are missing — [DEV Community](https://dev.to/bspann/bitnet-microsofts-1-bit-llms-that-run-on-your-cpu-20h8)

### Inferences
- QAD is the practical "QAT" for already post-trained models: it needs no original SFT/RL data and only hundreds of steps. The ModelOpt side effects show a gap in quality evaluation: score parity alone misses longer outputs and lost calibration. Track output length and abstention as well as accuracy.
- **For from-scratch training (user context):** BitNet and NVFP4-pretraining results suggest that quantization designed into training (ternary or FP4 from the start) is the route to sub-4-bit inference. PTQ at 2 bits stays fragile.
- Kimi's choice of weight-only INT4 QAT for MoE experts shows a cheap pattern when decode bandwidth is the bottleneck: quantize the experts, keep attention and activations in higher precision.

### Gaps
- I did not research HQQ, the "GPTQ v2"/GPTAQ papers, AWQ updates, or the Unsloth Dynamic 2.0 GGUF methodology in detail (only the NVFP4 page). No 2026 primary sources were gathered for them.
- I found no head-to-head EXL3 vs GGUF IQ2 perplexity numbers at 2 bits.
- I did not verify BitNet 2B4T-specific latency/energy numbers from Microsoft. A secondary source quotes 29 ms/token vs 65 ms/token for Qwen2.5-1.5B and a Jan 2026 CPU update giving 1.15-2.1x more speed, but neither was confirmed.
- I found no data on whether ternary models beyond 2B were released in 2026.

## 3. KV cache quantization and FP8 attention

### Takeaway
FP8 KV cache (`--kv-cache-dtype fp8`) is the vLLM default recommendation. It doubles KV capacity and roughly halves the per-token attention decode cost, with 94-99% accuracy recovery even uncalibrated. In 2026 vLLM had to fix a Hopper FA3 FP8 accumulation bug that had collapsed 128K-context needle accuracy. Gains only appear beyond about 4-7K tokens of context.

### Cited Findings
- **Hopper FA3 accumulation bug:** FP8 lost accumulation precision at long context. 128K needle-in-a-haystack accuracy was 91% in BF16 and 13% in FP8 before the fix, and 89% after a two-level FP32 accumulation fix — [vLLM blog, 2026-04-22](https://vllm.ai/blog/fp8-kvcache)
- **Accuracy recovery with FP8 KV plus FP8 attention, using uncalibrated per-tensor scales (scale 1.0, described as worst case):**
  - Llama-3.3-70B MRCR at 128K: 97-98% of baseline AUC.
  - Qwen3-30B-A3B up to 256K: about 94% (BF16 weights) and 98% (FP8 weights).
  - Qwen3.5-27B up to 1M: matches baseline.
  - Reasoning: Qwen3-30B-A3B-Thinking loses at most 1-2 points (worst on GPQA-D, 97% recovery). Qwen3.5-27B loses at most 0.7 points (AIME25, 99%).
  - B200/FlashInfer: 93-96% recovery.
  - Kimi-K2.5 with FlashMLA shows a consistent drop, so calibration is recommended there.

  Source: [vLLM blog](https://vllm.ai/blog/fp8-kvcache)
- **Decode speed:**
  - Llama-3.1-8B on H100: the per-token ITL slope falls to 54% of BF16, with break-even at about 7K tokens.
  - B200: slope also about 54%, with break-even at about 4K tokens.
  - gpt-oss-20b on H100: slope at 71% of BF16 when skipping sliding-window layers.
  - Serving at about 20K input / 2K output, concurrency 8: Llama-3.1-8B output throughput +14.9%, gpt-oss-20b +4.8%.

  Source: [vLLM blog](https://vllm.ai/blog/fp8-kvcache)
- **Costs and recommendations:**
  - For head_dim 256 models (gemma-4-E2B), long-context TTFT is about 1.6x worse with FP8.
  - Stay on BF16 KV below about 7K-token contexts.
  - Use `--kv-cache-dtype-skip-layers sliding_window` for hybrid models.
  - Calibrate (llm-compressor) if recovery falls below 95%.

  Source: [vLLM blog](https://vllm.ai/blog/fp8-kvcache); [vLLM quantized KV docs](https://docs.vllm.ai/en/v0.9.2/features/quantization/quantized_kvcache.html)
- vLLM's TurboQuant study (May 2026) still found FP8 the best KV default, with 2.6x higher burst throughput than BF16 and 2x KV capacity — [vLLM TurboQuant blog](https://vllm.ai/blog/turboquant) (via search snippet)
- Unsloth reports that FP8 KV calibration allows about 2x longer context with its NVFP4 quants — [Unsloth NVFP4 docs](https://unsloth.ai/docs/basics/nvfp4)
- **Reasoning models (2025):** 3-bit KV cache, or 4-bit weights plus activations plus KV (W4A4KV4), causes 3% or larger accuracy drops, especially for 1.5B and 7B models — [Liu et al., arXiv 2504.04823](https://arxiv.org/abs/2504.04823)

### Inferences
- FP8 KV is close to free for long-context and agentic serving. Below 4-bit KV, reasoning accuracy degrades, so 4-bit or 2-bit KV schemes (TurboQuant, KIVI-style) need per-workload validation.
- **For phone inference:** KV memory dominates at long context on small devices, so 8-bit KV with per-head or per-token scales is the safe first step. The "break-even context length" finding (fixed overhead vs per-token savings) likely applies to Metal kernels too, but I found no measurement of it.

### Gaps
- I did not read TurboQuant's full results (accuracy of sub-8-bit KV).
- I did not check SGLang's FP8 / FP4 KV support or MLX KV-cache quantization in detail.
- I found no data on FP8 attention on Apple GPUs. Apple GPUs have no FP8 tensor units, as far as any source I saw says.

## 4. Tooling: llm-compressor, ModelOpt, torchao, llama.cpp, MLX, vLLM/SGLang

### Takeaway
The NVIDIA path is ModelOpt (PTQ/QAD into NVFP4/FP8) plus Megatron-Bridge for QAD. The vLLM path is llm-compressor (GPTQ/AWQ, SpinQuant transforms, KV and attention quantization). The local and Apple path is GGUF via llama.cpp, Ollama and Unsloth, or MLX. MLX's quantizer supports affine 1-8 bit plus mxfp4, mxfp8 and nvfp4.

### Cited Findings
- **ModelOpt:** NVFP4 W4A4 PTQ of Qwen3.6-35B-A3B takes about 9 minutes plus HF export. QAD runs through Megatron-Bridge with the BF16 teacher — [ModelOpt post](https://nvidia.github.io/Model-Optimizer/announcements/20260916-qwen36-w4a4-qad.html)
- **llm-compressor:** includes SpinQuant-style transforms, KV cache quantization and attention quantization, and is the recommended calibration tool for vLLM FP8 KV scales — [vLLM llm-compressor docs](https://docs.vllm.ai/en/v0.29.0/features/quantization/llm_compressor/); [vLLM quantized KV docs](https://docs.vllm.ai/en/v0.9.2/features/quantization/quantized_kvcache.html)
- **MLX `quantize` modes:**

  | Mode | Bits | Group sizes | Scales | Bias |
  |---|---|---|---|---|
  | affine | 1, 2, 3, 4, 5, 6, 8 (default 4) | 32, 64, 128 (default 64) | input dtype | yes |
  | mxfp4 | 4 | 32 | E8M0 | no |
  | mxfp8 | 8 | 32 | E8M0 | no |
  | nvfp4 | 4 | 16 | E4M3, optional float32 `global_scale` | no |

  Source: [MLX docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- **MLX caveats:**
  - A GitHub issue reported that Metal supported only mxfp4 for quantized matmul at the time, with mxfp8 and nvfp4 planned — [MLX issue #2887](https://upd.dev/ml-explore/mlx/issues/2887). Current status is unverified.
  - Another issue says MLX nvfp4 uses signed E4M3 scales instead of unsigned UE4M3, which gives less dynamic range — [MLX issue #2962](https://upd.dev/ml-explore/mlx/issues/2962)
  - On Apple Silicon, mxfp8 and nvfp4 models run without activation quantization. MLX CUDA requires it — [mlx-audio guide](https://blaizzy.github.io/mlx-audio/guides/quantization/)
  - Community MLX nvfp4 and mxfp4 checkpoints exist on HF — [NuExtract3-mlx-nvfp4](https://huggingface.co/numind/NuExtract3-mlx-nvfp4); [LIMI-Air mxfp4](https://hface.pages.dev/nightmedia/LIMI-Air-mxfp4-mlx)
- **llama.cpp / GGUF:**
  - Official Gemma 3 and Gemma 4 QAT checkpoints ship as Q4_0 GGUF.
  - On Apple Silicon, Q4_0 decodes about 16-29% faster than Q4_K_M at similar size (42.9 vs 33.2 tok/s on Gemma 4 12B).

  Source: [3h4x blog](https://3h4x.github.io/tech/2026/06/08/gemma4-qat-vs-non-qat)
- **vLLM:** auto-selects FP4 MoE backends (cute-DSL, CUTLASS, FlashInfer-TRTLLM, Marlin). The choice changes decode speed by up to 1.9x — [Unsloth NVFP4 docs](https://unsloth.ai/docs/basics/nvfp4). SGLang launch recipes for NVFP4 with NEXTN speculative decoding also exist — [same](https://unsloth.ai/docs/basics/nvfp4)

### Inferences
- For a handwritten-Metal phone stack, the most transferable formats are:
  - Q4_0 / affine 4-bit with group 32-64: simple dequant, fastest decode on Apple per the Gemma 4 test.
  - mxfp4 (E8M0 scales): cheap shift-based dequant, supported in MLX Metal matmul.

  NVFP4's E4M3 scales cost more to decode in software and give no hardware benefit off Blackwell.
- A QAT-trained Q4_0 checkpoint (Gemma style: about 5K steps distilling against the BF16 model's probabilities) is a cheap recipe the user could apply to their own from-scratch models before export.

### Gaps
- I did not research torchao in 2026: its QAT APIs, int4/fp8 and MX support, ExecuTorch mobile paths. No sources were gathered.
- I did not verify llama.cpp's 2026 additions (MXFP4 GGUF type for gpt-oss, NVFP4 support, KV quant types).
- I did not verify Apple Core ML / ANE quantization options.
- SGLang FP8/FP4 support matrix: not researched.

## 5. Measured speedups and accuracy loss, especially for reasoning models

### Takeaway
Across 2025-2026 measurements, W8A8 (FP8/INT8) and W4A16 are near-lossless for reasoning models, and NVFP4 W4A4 on 27-35B models is within noise on most benchmarks. Instruction-following and some multimodal tasks lose 1-3 pp, and QAD can mostly recover that. The real-world speedups are modest:
- About 1.1-1.45x throughput from NVFP4 vs BF16 on Blackwell.
- About 2x claimed by Moonshot for INT4 weight-only at low latency.
- Larger gains on bandwidth-bound decode.

Small models, 3-bit, and W4A4KV4 are where accuracy breaks.

### Cited Findings
- **Liu et al. 2025** (DeepSeek-R1-Distill Qwen/LLaMA 1.5B-70B, QwQ-32B):
  - W8A8 or W4A16 is lossless.
  - 3-bit weights or KV, and W4A4KV4, cause 3% or larger drops, especially at 1.5B and 7B.
  - Quantized models do not produce longer outputs.
  - Larger quantized models beat smaller BF16 models on the accuracy-latency trade-off.

  Source: [arXiv 2504.04823](https://arxiv.org/abs/2504.04823)
- An earlier 2025 study found that AWQ/GPTQ caused up to 32.39% (average 11.31%) degradation on math reasoning for Llama-3 models under aggressive settings — [arXiv 2501.03035](https://arxiv.org/abs/2501.03035) (from search snippet; settings not checked)
- **Contradiction with Liu et al.:** the ModelOpt QAD run on Qwen3.6-35B-A3B found output length did change, with +91% SciCode completion length after QAD — [ModelOpt post](https://nvidia.github.io/Model-Optimizer/announcements/20260916-qwen36-w4a4-qad.html). Unsloth reports comparable output lengths for its PTQ NVFP4 quants — [Unsloth](https://unsloth.ai/docs/basics/nvfp4). The likely reconciliation: PTQ preserves length, while extra training (QAD) can shift it.
- **NVFP4 W4A4 reasoning accuracy (Qwen3.6-35B-A3B, NVIDIA's quant):** GPQA-D, SciCode, AA-LCR and tau2-bench are at BF16 parity. IFBench drops 2.6 pp and MMMU-Pro 1.2 pp before QAD — [ModelOpt post](https://nvidia.github.io/Model-Optimizer/announcements/20260916-qwen36-w4a4-qad.html)
- **FP8 KV on reasoning models:** 97-99% recovery on GPQA-D and AIME25 — [vLLM blog](https://vllm.ai/blog/fp8-kvcache)
- **Speedup summary:**

  | Setup | Speedup | Source |
  |---|---|---|
  | NVFP4 W4A4 vs BF16, Qwen3.6-35B-A3B on GB200 | 1.12-1.30x at batch; 0.87x at concurrency 1 | [ModelOpt](https://nvidia.github.io/Model-Optimizer/announcements/20260916-qwen36-w4a4-qad.html) |
  | Unsloth NVFP4, Gemma 4 | 1.12-1.45x | [Unsloth](https://unsloth.ai/docs/basics/nvfp4) |
  | MR-GPTQ kernels, B200 | 2.2x end-to-end | [arXiv 2509.23202](https://arxiv.org/pdf/2509.23202v1) |
  | MR-GPTQ kernels, RTX 5090 | 4x end-to-end | [arXiv 2509.23202](https://arxiv.org/pdf/2509.23202v1) |
  | Kimi INT4, low-latency mode | 2x (claimed) | [DeepInfra](https://deepinfra.com/moonshotai/Kimi-K2-Thinking) |
  | FP8 KV, Llama-3.1-8B, 20K context | +14.9% serving throughput | [vLLM](https://vllm.ai/blog/fp8-kvcache) |

- Evaluation noise matters. A 3 pp swing at 3 repeats vanished at 8 repeats, and the IFBench QAD gain only showed at 8 repeats. The full 6-benchmark evaluation cost about 100 GPU-hours — [ModelOpt post](https://nvidia.github.io/Model-Optimizer/announcements/20260916-qwen36-w4a4-qad.html)

### Inferences
- **Vendor "2-4x" claims vs end-to-end numbers:** the bigger figures are kernel-level or low-batch decode. Measured end-to-end serving gains for W4A4 on MoE models are 1.1-1.45x. Weight-only 4-bit can be slower than BF16 at high concurrency.
- **For reasoning-model evaluation:** use 8 or more repeats and track output length, truncation rate and hallucination/abstention alongside accuracy. Quantization or QAD can move these while scores stay flat.
- **For small phone-scale models (1-7B):** the Liu et al. sensitivity results argue for keeping weights at 4 bits or more, KV at 8 bits, and activations at 16 bits (or W8A8) unless QAT is used.

### Gaps
- I found no measured results for DeepSeek-R1/V3 at NVFP4, or for gpt-oss MXFP4 vs a BF16 upcast, on reasoning benchmarks.
- I found no Apple-device (iPhone/iPad Metal) benchmark comparing quantization formats for decode speed. The only Apple data point is one Mac user's Ollama test.
- I did not gather independent (non-vendor) replications of the Unsloth NVFP4 speed numbers.
