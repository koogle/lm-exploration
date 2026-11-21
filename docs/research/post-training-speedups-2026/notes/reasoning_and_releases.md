# Token-efficient reasoning post-training and speed features in 2025–2026 open-weight releases

Research date: 2026-11-10. Verification legend:
- **[fetched]**: primary source (model card, tech report, arXiv) read in this session.
- **[search]**: from search-result summaries or secondary write-ups. Not read at the primary source.
- **[prior]**: researcher's knowledge from before mid-2026, with the canonical primary URL. Not re-fetched in this session. These are stable, well-documented 2025 facts, but the report writer should treat them as one notch below [fetched].

Anything dated after about June 2026 comes from sources found in this session. Many are secondary and disagree with each other, so they carry more uncertainty, and each one is flagged.

## Reasoning length reduction: post-training methods and measured token savings vs accuracy

### Takeaway
The main post-training levers that cut reasoning tokens are:
- RL length rewards and budget-truncated RL (Kimi k1.5, L1/LCPO, ThinkPrune, LASER, ALP). Published results typically cut tokens 40–60% on math at roughly −2 to +6 points of accuracy, mostly on 1.5B–32B R1-distill models.
- Inference-time budget control (s1 budget forcing, Qwen3 thinking budgets).
- Trained discrete effort levels or think/no-think modes, now standard in releases (gpt-oss low/medium/high, DeepSeek V4 Non-think/High/Max, DeepSeek V4.1-Flash's continuous 1–100 effort scale).

Shipped models mostly expose *controls* (modes, effort levels, budgets) rather than claiming a fixed "shorter CoT" result.

### Cited Findings
**Surveys and taxonomy**
- "Stop Overthinking" (arXiv 2503.16419, Mar 2025) sorts efficient reasoning into model-based, output-based and prompt-based methods. "RL with length reward design" is a major model-based category. It defines overthinking as unnecessarily extended reasoning that raises latency and exceeds compute budgets. [search] — [Sui et al., arXiv 2503.16419](https://arxiv.org/abs/2503.16419)
- Other surveys: "Efficient Reasoning Models: A Survey" (2504.10903) and "Don't Overthink It" (2508.02120, focused on R1-style LRMs). [search] — [2504.10903](https://arxiv.org/abs/2504.10903); [2508.02120](https://arxiv.org/abs/2508.02120)

**RL with length penalties or budgets (measured numbers)**
- L1 / LCPO (2503.04697) trains with RL toward a length given in the prompt, in an exact-length variant and a max-length variant, on DeepSeek-R1-Distill-Qwen-1.5B.
  - It beats s1 budget forcing by up to 100% relative / 20% absolute on math at matched budgets.
  - The 1.5B L1 beats GPT-4o at equal reasoning length.
  - Limitation: L1-Exact gives every problem the same budget regardless of difficulty.
  - [search] — [L1, arXiv 2503.04697](https://arxiv.org/abs/2503.04697)
- ThinkPrune (2504.01296) sets a token limit during RL: unfinished reasoning gets zero reward.
  - R1-Distill-Qwen-1.5B on AIME24: length halved for about a 2% accuracy drop.
  - DeepScaleR-1.5B: 5,914 → 3,370 tokens.
  - QwQ-32B: 8,763 → 4,494 tokens.
  - [search] — [ThinkPrune, arXiv 2504.01296](https://arxiv.org/abs/2504.01296); numbers via [Train Long, Think Short notes](https://phonism.github.io/LLMNotes/en/train-long-think-short)
- LASER-D (2505.15612): 63% fewer tokens with +6.1 points on AIME2024. [search] — [arXiv 2505.15612](https://arxiv.org/abs/2505.15612) via [survey notes](https://phonism.github.io/LLMNotes/en/train-long-think-short)
- ALP, "Just Enough Thinking" (2506.05256), an adaptive length penalty: about 50% average token reduction on DeepScaleR-1.5B with a "minimal performance drop". [search] — [arXiv 2506.05256](https://arxiv.org/abs/2506.05256)
- LAPO (length-adaptive policy optimization): up to 40.9% fewer tokens with +2.3% accuracy on math. The authors say it beats ThinkPrune and L1; that comparison is unverified. [search] — [LAPO review](https://liner.com/review/lapo-internalizing-reasoning-efficiency-via-lengthadaptive-policy-optimization)
- LEASH (2512.21540): about 60% average token reduction on R1-Distill-Qwen-1.5B and Qwen3-4B-Thinking-2507 across math, code and instruction-following, with "competitive" accuracy. [search] — [arXiv 2512.21540](https://arxiv.org/abs/2512.21540)
- ConciseRL (2505.17250): "12.5x fewer tokens" with +2.2% on TheoremQA. [search] — [arXiv 2505.17250](https://arxiv.org/abs/2505.17250)
- O1-Pruner (2501.12570): about 50% lower inference overhead on math, with accuracy improving rather than dropping. [search] — [arXiv 2501.12570](https://arxiv.org/abs/2501.12570)
- Kimi k1.5 (Jan 2025) introduced an RL length penalty and "long2short" methods: model merging, shortest rejection sampling, and DPO/RL toward shorter outputs. [prior] — [Kimi k1.5, arXiv 2501.12599](https://arxiv.org/abs/2501.12599)

**Adaptive thinking (deciding whether to think)**
- AdaptThink uses RL to skip reasoning when possible, with importance sampling to balance thinking and no-thinking samples. Its token-reduction figures were not found in this session. [search] — [SmartThinker paper describing AdaptThink, arXiv 2507.04348](https://arxiv.org/abs/2507.04348)
- SmartThinker (2507.04348) does step-level length control, compressing redundant steps and keeping key ones. [search] — [arXiv 2507.04348](https://arxiv.org/abs/2507.04348)

**Training-free and inference-time methods (for contrast)**
- s1 budget forcing (2501.19393) truncates thinking or appends "Wait" to extend it. Extending moved AIME24 from 50% to 57%. [search] — [s1, arXiv 2501.19393](https://arxiv.org/abs/2501.19393)
- DEER (dynamic early exit, 2504.15895): 19.1–80.1% fewer tokens with +0.3 to +5.0 accuracy across 10 benchmarks and 11 LLMs. [search] — [arXiv 2504.15895](https://arxiv.org/abs/2504.15895)
- NoWait (2506.08343) suppresses "Wait"/"Hmm" tokens: 27–51% fewer tokens on 5 R1-style families. [search] — [arXiv 2506.08343](https://arxiv.org/abs/2506.08343)
- ES-CoT (2509.14004): 41% average reduction with comparable accuracy. Answer-convergence / learn-to-stop (2506.02536): 40–48% reduction on NaturalQuestions with QwQ-32B. [search] — [2509.14004](https://arxiv.org/abs/2509.14004); [2506.02536](https://arxiv.org/abs/2506.02536)

**Shipped controls in releases**
- **gpt-oss** (Aug 2025) is trained for three reasoning levels: low, medium and high. The level is set in the system prompt (for example "Reasoning: low"), and higher levels increase average CoT length.
  - gpt-oss-120b, AIME 2025 without tools: 50.4 / 80.0 / 92.5 for low / medium / high.
  - gpt-oss-120b, AIME 2024 without tools: 56.3 / 80.4 / 95.8.
  - gpt-oss-120b, GPQA Diamond without tools: 67.1 / 73.1 / 80.1.
  - The model card plots accuracy against CoT length and recommends picking a size/level pair that balances cost and accuracy.
  - [fetched] — [gpt-oss model card (PDF)](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf); [arXiv 2508.10925](https://arxiv.org/abs/2508.10925)
- **Qwen3** (Apr–May 2025) is a hybrid model with thinking and non-thinking modes in one checkpoint. Switches are `enable_thinking` and the `/think` and `/no_think` soft tags. The report describes "thinking budget" control, where performance scales smoothly with the allocated budget.
  - The July 2025 "2507" refresh split this into separate Instruct and Thinking checkpoints.
  - [prior] — [Qwen3 Technical Report, arXiv 2505.09388](https://arxiv.org/abs/2505.09388); [Qwen3-235B-A22B-Thinking-2507 card](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507)
- **Qwen3.5** (Feb 2026) has thinking on by default, disabled with `enable_thinking: False`. It "does not officially support" the Qwen3 `/think` and `/nothink` soft switches.
  - Recommended max output is 32,768 tokens, or 81,920 for hard math and coding.
  - No separate thinking-budget parameter is documented.
  - [fetched] — [Qwen3.5-397B-A17B model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- **DeepSeek V3.1** (Aug 2025) is one hybrid model whose chat template switches between thinking and non-thinking. DeepSeek said V3.1-Think reaches answers faster than R1-0528, at comparable quality, with fewer tokens. [prior] — [DeepSeek V3.1 release note](https://api-docs.deepseek.com/news/news250821)
- **DeepSeek V4-Pro / V4-Flash** (Apr 2026) have three modes: Non-think, Think High, and Think Max. Think Max uses a special system prompt and needs a context window of at least 384K. [fetched] — [DeepSeek-V4-Pro card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- **DeepSeek V4.1-Flash** (2026, exact date not stated) has a "continuously controllable" integer reasoning effort from 1 to 100 that trades inference cost for accuracy. All instruct benchmarks use 100. [fetched] — [DeepSeek-V4.1-Flash card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- **GLM-5.3** (Aug 2026) has Max and High reasoning settings.
  - Max: 34.5% completion at about 75K output tokens.
  - High: 31.4% at about 50K tokens.
  - So roughly a third fewer tokens costs 3.1 points, on Z.ai's own measurement and an unspecified benchmark.
  - [search] — via [growthvector Sept 2026 roundup](https://growthvector.substack.com/p/notable-open-weight-models-sept2026), citing [Z.ai blog](https://z.ai/blog/glm-5.3)
- GLM-4.5 (Jul 2025) is a hybrid reasoning model with a thinking mode and a direct-answer mode. [prior] — [GLM-4.5 report, arXiv 2508.06471](https://arxiv.org/abs/2508.06471)

### Inferences
- The industry moved from "one hybrid checkpoint with a think toggle" (Qwen3, DeepSeek V3.1, GLM-4.5, 2025) toward two patterns:
  - separate or specialized checkpoints (Qwen3-2507);
  - graded effort levels: gpt-oss 3 levels, DeepSeek V4 3 modes, GLM-5.3 Max/High, DeepSeek V4.1-Flash 1–100.

  Graded effort looks like the 2026 default for reasoning-capable releases.
- Most academic token-saving results are on small (1.5B–32B) R1-distill models on math. The consistent result is that 40–60% shorter CoT is achievable at about ±2 points. Whether this transfers to frontier MoE or agentic workloads is less documented.
- gpt-oss's low→high gap (about 42 points on AIME25 for 120b) shows effort level is a large lever on hard math. On knowledge-heavy tasks (MMLU 85.9 low vs 88.0 medium) low effort loses little, so routing by task difficulty is where token savings come from.

### Gaps
- Exact token counts per reasoning level for gpt-oss: the model card has them only in figures, which were not extracted.
- AdaptThink's measured token reduction, and DeepSeek V3.1's quantified token savings vs R1, were not verified in this session.
- No independent, cross-vendor measurement was found of tokens per task across 2026 releases at each effort level. Artificial Analysis-style token counts were not fetched.
- Whether DeepSeek V4, Qwen3.5 or GLM-5 used explicit length penalties in RL is not stated in the sources fetched.

## Per-release speed techniques shipped (2025 → late 2026)

### Takeaway
By 2026 the standard flagship recipe is:
- fine-grained MoE with about 2.5–5% active parameters (256–512 experts, 6–10 routed plus 1 shared);
- FP8, with FP4 or INT4 for expert weights (natively trained or QAT);
- MTP heads for speculative decoding;
- an efficient-attention scheme: DeepSeek-style sparse attention (DSA/CSA/HCA), Gated-DeltaNet/KDA hybrid linear attention (Qwen3-Next/3.5, Kimi Linear lineage), sliding-window/global mixes (gpt-oss, Gemma, MiMo), or Mamba-2 hybrids (Nemotron).

MiniMax M2 is the notable counterexample: it went back to full attention.

### Cited Findings
**DeepSeek**
- V3 / R1 (Dec 2024 / Jan 2025): 671B total / 37B active, MLA, FP8 training, one MTP module usable for speculative decoding. [prior] — [DeepSeek-V3 report, arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- V3.1 (Aug 2025): hybrid think/no-think; trained with the UE8M0 FP8 scale format. [prior] — [V3.1 release note](https://api-docs.deepseek.com/news/news250821); [HF card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)
- V3.2-Exp (29 Sep 2025) introduced DeepSeek Sparse Attention (DSA).
  - DSA sits on top of MLA. A lightweight "lightning indexer" picks the top-k KV entries per query.
  - It matched V3.1-Terminus (for example, MMLU-Pro 85.0 for both) at much lower long-context cost.
  - API prices were cut by 50%+.
  - [search] — [DeepSeek announcement](https://api-docs.deepseek.com/news/news250929); [Techzine](https://www.techzine.eu/news/analytics/134996/deepseek-launches-v3-2-exp-with-breakthrough-in-sparse-attention)
- V3.2 (Dec 2025, arXiv 2512.02556): DSA plus scalable RL and agentic task synthesis; continued pre-training from V3.1-Terminus. [search] — [arXiv 2512.02556](https://arxiv.org/abs/2512.02556)
- V4-Pro: 1.6T total / 49B active. V4-Flash: 284B / 13B. Both released around 24 Apr 2026.
  - 1M context; hybrid Compressed Sparse Attention (CSA) plus Heavily Compressed Attention (HCA).
  - At 1M context, V4-Pro needs 27% of V3.2's per-token inference FLOPs and 10% of its KV cache.
  - Release weights mix FP4 (MoE experts) and FP8 (most other weights); base models are FP8.
  - Also uses mHC (manifold-constrained hyper-connections), the Muon optimizer, and pre-training on 32T+ tokens.
  - MTP is not mentioned on the card.
  - [fetched] — [DeepSeek-V4-Pro card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro). Release date is [search] — [apidog](https://apidog.com/blog/what-is-deepseek-v4/); sources disagree between 23 and 24 April.
- V4.1-Flash (2026, after V4; exact date not stated) changes architecture to a Causal Encoder-Decoder.
  - 40 layers: a 20-layer encoder plus a 20-layer decoder.
  - 552B backbone (the HF listing shows 763B total); 1 shared plus 384 routed experts, 6 active.
  - About 8B active parameters in prefill vs about 16B in decode.
  - Global KV cache in FP4 (E2M1 with one E4M3 scale per 16 channels), about 890 bytes per token, roughly 1/4 of V4-Flash.
  - Sliding-window KV is recomputed by replaying recent tokens instead of being stored (about 1/8 of V4-Flash's persistent SWA cache).
  - "DSpark" speculative decoding: semi-autoregressive drafts with confidence-scheduled verification.
  - MIT license; text and image input.
  - [fetched] — [DeepSeek-V4.1-Flash card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). Some secondary sources conflict with this; the card is authoritative.

**Qwen**
- Qwen3 (Apr 2025): dense 0.6B–32B; MoE 30B-A3B and 235B-A22B (128 experts, 8 active); hybrid thinking. [prior] — [arXiv 2505.09388](https://arxiv.org/abs/2505.09388)
- Qwen3-Next-80B-A3B (Sep 2025):
  - Hybrid of Gated DeltaNet and gated attention in a 3:1 ratio.
  - 512 experts (10 routed plus 1 shared), about 3.7% active.
  - Native MTP.
  - Qwen claimed more than 10x prefill/decode throughput vs Qwen3-32B beyond 32K context, at under 10% of the training cost.
  - [prior] — [Qwen3-Next blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd); [HF card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- Qwen3.5-397B-A17B (Feb 2026):
  - 60 layers as 15 × [3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)].
  - Gated DeltaNet: 64 V heads / 16 QK heads. Gated Attention: 32 Q / 2 KV heads, head dim 256, partial RoPE (64 dims).
  - 512 experts, 10 routed plus 1 shared, expert intermediate size 1024. About 4.3% active.
  - MTP "trained with multi-steps".
  - 262K native context, 1.01M with YaRN. Apache-2.0.
  - [fetched] — [Qwen3.5-397B-A17B card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- The Qwen3.5 family also includes MoE 122B-A10B and 35B-A3B, plus dense 27B/9B/4B/2B/0.8B. Small dense models reportedly use the same 3:1 DeltaNet:attention hybrid. [search] — [morphllm](https://www.morphllm.com/qwen-3-5); [NVIDIA Megatron Bridge docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/qwen/qwen35-vl.html)
- Later 2026: there are references to a "Qwen 3.6" (in NVIDIA docs) and to Qwen3.8-27B (a hybrid Gated DeltaNet layout, used as the base for PrismML's ternary Bonsai 2). Sources conflict on details. [search, unverified] — [growthvector](https://growthvector.substack.com/p/notable-open-weight-models-sept2026); [NVIDIA docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/qwen/qwen35-vl.html)

**Moonshot Kimi**
- Kimi K2 (Jul 2025): 1T total / 32B active (3.2%), 384 experts with 8 routed plus 1 shared, MLA, MuonClip optimizer. [prior] — [Kimi K2 report, arXiv 2507.20534](https://arxiv.org/abs/2507.20534)
- Kimi K2-Thinking (Nov 2025):
  - Same 1T / 32B, 384 experts (8 plus 1 shared), MLA, 256K context.
  - Native INT4 weight-only QAT on the MoE components during post-training.
  - Claims a "lossless 2x speed-up in low-latency mode".
  - Stable over 200–300 sequential tool calls.
  - [fetched] — [Kimi-K2-Thinking card](https://huggingface.co/moonshotai/Kimi-K2-Thinking)
- Kimi Linear (Oct 2025): Kimi Delta Attention (KDA) hybrid at 3:1 KDA:MLA; up to 75% lower KV cache and up to 6x decode throughput at 1M context. [prior] — [Kimi Linear, arXiv 2510.26692](https://arxiv.org/abs/2510.26692)
- "Kimi K3" (2.8T total / 104B active) appears only in an aggregator. [search, unverified] — [open-weight roundup search](https://www.techtwitter.com/articles/10-open-weight-llm-releases-in-january-and-february-2026)

**Zhipu / Z.ai GLM**
- GLM-4.5 (Jul 2025): 355B total / 32B active; GLM-4.5-Air 106B / 12B. Has an MTP layer for speculative decoding and hybrid thinking. [prior] — [arXiv 2508.06471](https://arxiv.org/abs/2508.06471)
- GLM-4.6 (Sep–Oct 2025): same base, 200K context, about 15% fewer tokens than GLM-4.5 on Z.ai's CC-Bench. [prior] — [GLM-4.6 blog](https://z.ai/blog/glm-4.6)
- GLM-5 (Feb 2026, arXiv 2602.15763):
  - 744B total / 40B active (the checkpoint tensor count is about 754B); 28.5T pre-training tokens.
  - First GLM with DeepSeek Sparse Attention; same structure as DeepSeek V3.2 in SGLang.
  - One MTP layer (`num_nextn_predict_layers: 1`) for speculative decoding.
  - [search] — [GLM-5 report](https://arxiv.org/html/2602.15763); [SGLang cookbook](https://cookbook.sglang.io/autoregressive/GLM/GLM-5); [Baseten](https://www.baseten.co/blog/how-we-built-the-fastest-glm-5-on-artificial-analysis/)
- GLM-5.2 and GLM-5.3 (mid/late 2026):
  - About 744B / 40B on the GLM-5.2 base, with sparse attention.
  - "IndexShare" reuses sparse-attention indices across groups of layers.
  - Max and High reasoning settings.
  - A "GLM-5.3-Flash" with hybrid KDA plus sparse attention and MTP is reported by aggregators.
  - [search, partly unverified] — [growthvector](https://growthvector.substack.com/p/notable-open-weight-models-sept2026); [Z.ai blog](https://z.ai/blog/glm-5.3)

**OpenAI gpt-oss (Aug 2025)**
- gpt-oss-120b: 116.8B total / 5.1B active, 36 layers, 128 experts. gpt-oss-20b: 20.9B total / 3.6B active, 24 layers, 32 experts. Top-4 routing in both.
- MoE weights (90%+ of parameters) are post-trained in MXFP4 at 4.25 bits per parameter. That lets 120b fit on one 80GB GPU and 20b run in 16GB.
- Attention alternates banded-window (bandwidth 128 tokens) and dense layers, GPT-3 style.
- GQA with 64 query heads of dim 64 and 8 KV heads; YaRN extends to 131,072 context.
- Each head has a learned softmax-denominator bias ("attention sinks" / off-by-one) so it can attend to nothing.
- Three reasoning levels (see above).
- [fetched] — [gpt-oss model card](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf)

**Google Gemma**
- Gemma 3 (Mar 2025):
  - 5:1 local:global attention with a 1024-token sliding window, cutting KV cache.
  - Official QAT checkpoints (int4) shipped in Apr 2025, so 27B runs on consumer GPUs.
  - [prior] — [Gemma 3 report, arXiv 2503.19786](https://arxiv.org/abs/2503.19786); [Gemma 3 QAT blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)
- Gemma 3n (Jun 2025): MatFormer nested model (E4B contains E2B; sizes in between via mix-and-match); Per-Layer Embeddings kept in CPU memory; KV-cache sharing for faster prefill. [prior] — [Gemma 3n developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- Gemma 4 (2026):
  - First Gemma MoE: 26B "A4B", 128 small experts, 8 routed plus 1 shared, about 3.8B active.
  - Also a 31B dense model and edge E2B/E4B models with Per-Layer Embeddings.
  - Local sliding-window and global attention alternate, with the final layer always global.
  - Context and parameter numbers conflict across sources.
  - [search] — [Kaitchup](https://kaitchup.substack.com/p/gemma-4-31b-and-26b-a4b-architecture); [Qubrid](https://www.qubrid.com/blog/google-gemma-4-technical-deep-dive-architecture-moe-benchmarks-production-guide)

**Meta Llama 4 (Apr 2025)**
- Scout: 109B total / 17B active, 16 experts. Maverick: 400B / 17B, 128 experts plus a shared expert, alternating dense and MoE layers.
- iRoPE: interleaved NoPE global layers with chunked-local RoPE layers; Scout supports 10M context.
- FP8 Maverick weights shipped.
- No reasoning mode at launch.
- [prior] — [Meta Llama 4 blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- A later "Meta Muse Spark Open" (816B / 42B, 1M context) appears only in aggregators. [search, unverified] — [open-weight roundup search](https://openrouter.ai/blog/insights/the-open-weight-models-that-matter-june-2026/)

**NVIDIA Nemotron**
- Nemotron Nano 2 (Aug 2025):
  - 9B hybrid Mamba-2/Transformer, about 92% Mamba-2 layers.
  - Pruned and distilled from a 12B base with Minitron.
  - Up to 6x throughput vs Qwen3-8B on reasoning-length generation.
  - Runtime "thinking budget" control.
  - [prior] — [Nemotron Nano 2, arXiv 2508.14444](https://arxiv.org/abs/2508.14444)
- Nemotron 3 Nano (Dec 2025): 30B-A3B hybrid Mamba-2 plus MoE with only a few attention layers. 3.3x throughput vs a size-matched Transformer MoE at typical reasoning lengths. [search] — [Nemotron 3 white paper, arXiv 2512.20856](https://arxiv.org/abs/2512.20856)
- Nemotron 3 Super (Apr 2026 report):
  - 120B total / 12B active.
  - First in the family pre-trained in NVFP4, and first to use LatentMoE (tokens projected into a latent space before routing).
  - Ablation: LatentMoE scored 52.87 vs 48.30 MMLU-Pro over a standard MoE of similar active size.
  - [search] — [Nemotron 3 Super tech report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- Nemotron 3 Ultra (about 550B / 55B, hybrid Mamba-2 MoE with MTP) is reported by aggregators. [search, unverified]
- Nemotron 3.5 Lightning (about Aug 2026): 30B / 3B, hybrid Mamba-2 plus MoE plus attention, speculative decoding, NVFP4 deployments. [search] — [NVIDIA dev blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-5-lightning-delivers-fast-accurate-specialized-task-execution-for-long-running-agents/)

**MiniMax**
- MiniMax-M1 (Jun 2025):
  - 456B total / 45.9B active hybrid of Lightning Attention (linear) and softmax attention.
  - 1M context with up to 80K thinking.
  - About 25% of DeepSeek-R1's FLOPs at 100K generation length.
  - CISPO RL algorithm.
  - [prior] — [MiniMax-M1, arXiv 2506.13585](https://arxiv.org/abs/2506.13585)
- MiniMax-M2 (Oct 2025) through M2.5 (12 Feb 2026) and M2.7 (18 Mar 2026):
  - About 229.9B total / 9.8B active (about 4.3%); 62 layers; 256 experts, 8 active.
  - **Full** GQA attention on every layer (48 Q / 8 KV heads). The M2 report presents this as a deliberate choice over linear or hybrid attention.
  - "Interleaved thinking" for agents.
  - [search] — [Vultr M2.5 docs](https://docs.vultr.com/models/minimax-m2/minimax-m2.5/nvidia); [SemiAnalysis InferenceX M2.7](https://inferencex.semianalysis.com/model/minimax-m27). Conflicting claim: one source says M2.1 reintroduced Lightning Attention; unverified.
- MiniMax-M3 (about 1 Jun 2026): introduces "MiniMax Sparse Attention" (MSA) for long-context decoding, with a claimed 15.6x response speed boost.
  - Parameter counts conflict: 229.9B / 9.8B vs 428B / 23B with 1M context.
  - [search] — [VentureBeat](https://venturebeat.com/technology/minimax-teases-upcoming-m3-model-with-new-sparse-attention-mechanism-and-15-6x-response-speed-boost); [DataLearner](https://www.datalearner.com/en/benchmark-compare/minimax-m2-5/minimax-m3)

**Other late-2026 releases (mostly secondary; verify)**
- Xiaomi MiMo-V2.6 (about Aug–Sep 2026), from the HF card cited in the roundup [search]:
  - Pro: about 1.02T / 42B.
  - Flash: about 309B / 15B; 48 layers, 39 sliding-window plus 9 global; 256 experts, 8 active.
  - A 5-layer multi-token speculative decoder; 1M context.
  - [growthvector](https://growthvector.substack.com/p/notable-open-weight-models-sept2026); [HF](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD)
- PrismML Ternary Bonsai 2 27B (on Qwen3.8-27B): ternary weights at about 1.76 bits per weight, 5.9GB, about 9x smaller than full precision. [search] — [PrismML](https://prismml.com/news/bonsai-2-27b)
- Solar Open 2: 3:1 KDA linear to softmax attention with NoPE. [search, unverified] — [open-weight search results](https://github.com/xigh/open-weight-models)

### Inferences
**Active-parameter ratios** (computed from the figures above):

| Model | Active / total |
|---|---|
| gpt-oss-120b | 4.4% |
| gpt-oss-20b | 17% |
| Kimi K2 | 3.2% |
| Qwen3-Next | 3.75% |
| Qwen3.5 | 4.3% |
| MiniMax M2 | 4.3% |
| DeepSeek V3 | 5.5% |
| DeepSeek V4-Pro | 3.1% |
| DeepSeek V4-Flash | 4.6% |
| GLM-5 | 5.4% |
| Nemotron 3 Super | 10% |

Flagships converged on about 3–5% active with 256–512 experts and 6–10 routed plus a shared expert.

**What became standard by 2026:**
1. Fine-grained MoE with a shared expert.
2. FP8 training or inference as the baseline, plus 4-bit experts, with the 4-bit format now *shipped* by the lab: MXFP4 (gpt-oss), INT4 QAT (K2-Thinking), FP4 experts (DeepSeek V4), NVFP4 pre-training (Nemotron 3 Super).
3. MTP or a built-in speculative drafter: DeepSeek V3, GLM-4.5/5, Qwen3-Next/3.5, MiMo, DeepSeek DSpark.
4. Sub-quadratic or compressed long-context attention. Exactly one approach has not won; several co-exist.
5. Graded reasoning effort.

**Attention approaches have split into families:**
- DeepSeek sparse attention: DSA → CSA/HCA. Adopted by GLM-5.
- Delta-rule linear hybrids: Qwen Gated DeltaNet, Kimi KDA, Solar.
- SWA/global mixes: gpt-oss, Gemma, MiMo.
- Mamba-2 hybrids: Nemotron.
- MiniMax went from lightning attention back to full attention (M2) and then to a sparse scheme (M3), a cautionary data point on linear attention's quality cost for agentic tasks.

**Trend toward decode-time cost:** DeepSeek V4.1-Flash's encoder-decoder (8B active prefill / 16B decode) and FP4 KV cache suggest that by late 2026 KV memory per token and decode cost are the main optimization targets, more than parameter FLOPs.

### Gaps
- Primary verification is missing for:
  - Kimi K3, Meta Muse Spark Open, Nemotron 3 Ultra, GLM-5.2/5.3 specs, MiniMax M3 parameter counts, Qwen3.6/3.8, and Gemma 4 context lengths;
  - exact release dates for DeepSeek V4.1-Flash, MiMo-V2.6 and GLM-5.3 weights.
- Whether DeepSeek V4 keeps MTP: the card is silent; V4.1-Flash uses DSpark instead.
- Measured end-to-end speedups from MTP acceptance rates per release, and independent INT4/FP4 accuracy-loss audits, were not collected.
- Llama 4, Gemma 3/3n, Qwen3-Next, Kimi K2/Linear, GLM-4.5/4.6, Nemotron Nano 2 and MiniMax-M1 facts are [prior] and were not re-fetched in this session because of the tool budget.
- Raschka and Interconnects analyses were not fetched in this session.
