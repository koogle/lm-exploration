# Post-training Compression and Architecture Conversion for Faster LLM Inference (state as of 2026-11-10)

Scope note: these notes cover structural changes to an existing open model (pruning, distillation, attention or architecture conversion, dynamic depth, AR-to-diffusion). Quantization and speculative decoding belong to other researchers and are mentioned only where they interact with this topic. Items marked **[background]** are from before 2025. Items marked **[not re-verified]** are well-known claims whose source URL is given, but I did not re-open that source in this session. The report writer should treat those as lower confidence.

Cross-cutting caveat: nearly every speedup below is the authors' own measurement, at its best-case setting (long generations, large batch, specific GPU). The baselines differ from paper to paper, so the multipliers cannot be compared directly. I found almost no independent replications.

## 1. Pruning + distillation (Minitron / Nemotron Nano / Puzzle, width/depth, 2:4 sparsity, MoE expert pruning)

### Takeaway
"Prune, then distill" is NVIDIA's production recipe. It takes a 12B model to 9B for Nemotron Nano 2, and its NAS form (Puzzle) takes 70B to 51B and 405B to 253B. It gives roughly 1.7x–2.2x throughput at about 98% of the original accuracy, using a small fraction of the original pretraining tokens. For MoE models, one-shot expert *pruning* (REAP) now beats expert *merging* at about 50% expert removal. 2:4 semi-structured sparsity recovers about 98% of accuracy after 13B tokens of distillation, but I found no published wall-clock speedups for it.

### Cited Findings
**Minitron / Nemotron Nano**
- [background] Minitron (2024): width pruning (heads, MLP neurons, embedding channels) and depth pruning (layers) of Nemotron-15B and Llama-3.1-8B, followed by logit distillation. The authors claim up to 40x fewer training tokens than training from scratch. [not re-verified] — [Compact Language Models via Pruning and Knowledge Distillation, arXiv 2407.14679](https://arxiv.org/abs/2407.14679)
- Nemotron Nano 2 (Aug 2025): Nemotron-Nano-12B-v2-Base was compressed with the Minitron strategy (pruning layers, FFN dims and embedding channels, plus distillation) to Nemotron-Nano-9B-v2. The 9B model runs 128k-token inference on a single A10G (22 GiB) — [arXiv 2508.14444](https://arxiv.org/abs/2508.14444v2)
- Nemotron Nano 2 claims "up to 6x higher inference throughput" than Qwen3-8B in reasoning settings (8k input / 16k output), with on-par or better accuracy. Part of this comes from the hybrid architecture (about 8% of layers are attention, the rest Mamba-2 plus FFN), not from pruning alone. One review quotes 6.3x — [arXiv 2508.14444](https://arxiv.org/abs/2508.14444v4); [Moonlight review](https://www.themoonlight.io/review/nvidia-nemotron-nano-2-an-accurate-and-efficient-hybrid-mamba-transformer-reasoning-model)

**Puzzle (distillation-based NAS)**
- Puzzle is a hardware-aware NAS. It runs blockwise local distillation of alternative block types (for example, attention removed or narrower FFNs) and then a constrained search. Llama-3.1-Nemotron-51B (from Llama-3.1-70B-Instruct) achieves a 2.17x inference throughput speedup, fits on one H100, and retains 98.4% of the parent's benchmark accuracy — [Puzzle, arXiv 2411.19146](https://arxiv.org/abs/2411.19146v3); [HF papers page](https://huggingface.co/papers/2411.19146)
- Llama-Nemotron family (2025): LN-Super (49B, derived from Llama-3.3-70B) has 5x the throughput of Llama-3.3-70B-Instruct at batch 256, TP1. LN-Ultra (253B, derived from Llama-3.1-405B-Instruct) is optimized for one 8xH100 node and has 1.71x lower latency than 405B — [Llama-Nemotron: Efficient Reasoning Models, arXiv 2505.00949](https://arxiv.org/html/2505.00949v2); [NIM model card](https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1/modelcard)

**2:4 semi-structured sparsity**
- Sparse-Llama-3.1-8B-2of4 (Neural Magic / Red Hat): pruned with an optimized SparseGPT (LLM-Compressor), then recovered with SquareHead knowledge distillation on 13B tokens. It scores 62.16 versus 63.19 dense on OpenLLM v1 (98.37% recovery) and 53.85 versus 55.34 on the Mosaic Eval Gauntlet (97.3%). The card claims "2x model size and compute reduction", which is theoretical — [HF model card](https://huggingface.co/RedHatAI/Sparse-Llama-3.1-8B-2of4)
- Fine-tuned 2:4 variants recover about 98.5% (ultrachat, AlpacaEval 61.1 vs 62.0). Combined with FP8-dynamic, the result is 99.4% of the dense fine-tune — [HF model card](https://huggingface.co/neuralmagic/Sparse-Llama-3.1-8B-ultrachat_200k-2of4-FP8-dynamic)

**MoE expert pruning / merging**
- REAP (Cerebras; arXiv Oct 2025; ICLR 2026) scores each expert by router gate value times expert activation norm, averaged over the tokens routed to it. It needs one calibration pass, then removes the lowest-scoring experts. The authors argue that merging causes "functional subspace collapse": the router loses independent control of the merged experts. Pruning keeps that control. — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v1); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10006871); [Cerebras blog](https://www.cerebras.ai/blog/reap)
- REAP was tested on 20B to 1T-parameter MoEs. The authors report near-lossless code generation on Qwen3-Coder-480B and Kimi-K2 with 50% of experts pruned, and say it beats merging and other pruning baselines on generative tasks, especially at 50% compression — [arXiv 2510.13999](https://arxiv.org/abs/2510.13999v1)

### Inferences
- The best-supported quality/speed numbers come from the NVIDIA prune+distill line, which has shipped open checkpoints. The rule of thumb: about 25–30% parameter reduction keeps about 98% quality at about 2x throughput, when distillation is used for recovery.
- Expert pruning reduces memory footprint (total parameters) more than per-token compute, because active parameters per token stay roughly constant. The speed benefit is mainly fitting on fewer GPUs.
- Without published vLLM latency numbers, the practical speedup of 2:4 sparsity should be treated as unproven. Combining 2:4 with FP8 seems to be how it gets used in practice.

### Gaps
- No measured end-to-end throughput for 2:4 sparse Llama in vLLM came up in this search.
- No independent replication of REAP. The search returned no throughput or latency numbers for REAP-pruned checkpoints.
- I did not check whether Llama 3.2 1B/3B (Meta's own pruned and logit-distilled models from Llama 3.1 8B) published quality-vs-cost numbers.
- Not verified: Nemotron Nano 3 or later 2026 Minitron releases.

## 2. Logit / knowledge distillation into smaller dense models (DeepSeek-R1 distills, Qwen3 small models, Gemma)

### Takeaway
Distillation from a large teacher is now the default way to make small open models. In Qwen3, on-policy logit distillation beats RL on the same student, at about 1/10 of the GPU hours. DeepSeek-R1's distills show that plain SFT on about 800k teacher samples, without logits, transfers reasoning to 1.5B–70B dense models.

### Cited Findings
- Qwen3 trains its small models (0.6B–14B, plus 30B-A3B) with "strong-to-weak distillation". The off-policy phase trains on teacher outputs in /think and /no_think modes. In the on-policy phase, the student generates responses and is "fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence" — [Qwen3 Technical Report, arXiv 2505.09388](https://arxiv.org/pdf/2505.09388)
- Qwen3 Table 21 (Qwen3-8B, all runs starting from the same off-policy-distilled checkpoint):
  - Off-policy distillation alone: AIME'24 55.0, AIME'25 42.8, MATH500 92.4, LiveCodeBench v5 42.0, MMLU-Redux 86.4, GPQA-Diamond 55.6
  - Plus RL: 67.6 / 55.5 / 94.8 / 52.9 / 86.9 / 61.3, at **17,920 GPU hours**
  - Plus on-policy distillation instead: 74.4 / 65.5 / 97.0 / 60.3 / 88.3 / 63.3, at **1,800 GPU hours**
  - Pass@64 on AIME improves with distillation (90.0→93.3 and 83.3→86.7) but not with RL — [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)
- The Qwen3 report says strong-to-weak distillation needs "only 1/10 of the GPU hours" of the full four-stage pipeline per small model — [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)
- On-policy distillation (as in Qwen3-14B) has a hidden cost: it needs frequent, real-time access to the teacher model — [Confucius3-Math, arXiv 2506.18330](https://arxiv.org/pdf/2506.18330)
- DeepSeek-R1 distills (Jan 2025): Qwen2.5 (1.5B–32B) and Llama (8B, 70B) were fine-tuned with SFT only, on about 800k samples curated with DeepSeek-R1, with no RL stage. [not re-verified] — [DeepSeek-R1, arXiv 2501.12948](https://arxiv.org/abs/2501.12948)
- Gemma 2 and Gemma 3 train their smaller models with logit (token-level) distillation from a larger teacher during pretraining. [not re-verified] — [Gemma 2 report, arXiv 2408.00118](https://arxiv.org/abs/2408.00118); [Gemma 3 report, arXiv 2503.19786](https://arxiv.org/abs/2503.19786)

### Inferences
- For anyone producing a small model from an open teacher, the Qwen3 numbers favor this order: SFT on teacher outputs, then on-policy logit-KL distillation, rather than RL for the small model. This requires teacher and student to share a tokenizer, or some cross-tokenizer alignment.
- Distillation here is a quality technique, not a speed technique. The speed comes from choosing a smaller (or architecturally cheaper) student. Sections 1 and 3 are the "structure" half and this is the "recovery" half.

### Gaps
- I did not verify exact DeepSeek-R1-Distill benchmark numbers or Gemma 3 distillation details in this session.
- I found no public cost comparison between logit distillation and sequence-level (SFT-on-samples) distillation at matched compute, other than Qwen3's RL-vs-distillation table.

## 3. Attention conversion (to linear/hybrid/SSM, MHA/GQA→MLA, sparse/sliding-window retrofits, KV compression)

### Takeaway
Converting a pretrained Transformer to linear attention or SSM now takes only 0.35B–20B tokens: RADLADS uses 350–700M tokens, LoLCATs tens of millions, and Llamba 12B. Decode throughput gains are typically 3–12x at long generation. Jet-Nemotron (PostNAS) claims up to about 47x decode throughput at 64K context against Qwen3-1.7B, because its KV cache is about 47x smaller. GQA→MLA conversion (TransMLA) reports 10.6x at 8K context in vLLM. DeepSeek-V3.2's DSA shows that a sparse-attention retrofit by continued training can work at frontier scale with near-parity quality.

### Cited Findings
**Linear / SSM / hybrid conversion**
- Mamba in the Llama (2024, NeurIPS): distills Llama-3-8B into hybrid Mamba/Mamba-2 models with about 20B tokens. It motivates the work by the roughly 5x higher throughput of linear RNNs. [background] — [arXiv 2408.15237](https://arxiv.org/html/2408.15237v4)
- LoLCATs (Oct 2024): linearizes Llama-3-8B, then 70B and 405B, by attention-transfer (output-matching) followed by LoRA. The authors claim 3x higher throughput and 64x larger batch than FlashAttention-2 for Llama-3-8B. [background] — [arXiv 2410.10254](https://arxiv.org/html/2410.10254v1)
  - Conflict: the Llamba comparison table lists LoLCATs at 0.04T (40B) tokens, but the LoLCATs paper describes a much smaller budget (about 0.4% of prior methods' tokens). The report writer should not quote a LoLCATs token count without checking.
- Llamba (Cartesia, Feb 2025): uses MOHAWK to distill Llama-3.x into Mamba-2-based models at 1B, 3B and 8B. Llamba-3.1-8B used 12B tokens, under 0.1% of Llama-3.1-8B's training data — [arXiv 2502.14458](https://arxiv.org/html/2502.14458v2)
  - Paper: Llamba-8B has higher throughput than its teacher even when the teacher generates 4x fewer tokens. Vendor blog: up to 12x higher throughput than Llama-3.1-8B on H100-80GB at 8192-token generation — [Cartesia blog](https://cartesia.ai/blog/llamba-distillation)
- RADLADS (May 2025; Recursal/EleutherAI): converts Qwen2.5 7B, 32B and 72B to RWKV-6 and RWKV-7 variants ("RADFinch", "RADGoose") with 350–700M tokens, under 0.005% of the teachers' training tokens. The 72B conversion cost under $2,000. Quality stays close to the teacher. Prior methods used far more data: SUPRA 100B, MOHAWK 3–5B, Mamba-in-the-Llama 20B. Released checkpoints include QRWKV6-7B/32B/72B-Instruct and QRWKV7-7B-Instruct. Apache-2.0, except 72B under the Qwen license — [arXiv 2505.03005](https://arxiv.org/html/2505.03005v4); [EleutherAI blog](https://www.eleuther.ai/papers-blog/radlads-rapid-attention-distillation-to-linear-attention-decoders-at-scale)
- Jet-Nemotron (NVIDIA, Aug 2025, NeurIPS 2025): PostNAS starts from a pretrained full-attention model (Qwen2.5) with frozen MLPs. It searches where to keep full-attention layers, chooses a linear-attention block (the new "JetBlock"), and tunes hyperparameters for hardware. Key finding: generation throughput is governed by KV-cache size more than by parameter count — [arXiv 2508.15884](https://arxiv.org/abs/2508.15884v3); [HF papers](https://huggingface.co/papers/2508.15884)
  - Jet-Nemotron-2B: up to 47x decode throughput at 64K context on H100. Reported table: 2,885 tok/s versus 61 tok/s for Qwen3-1.7B-Base, with a KV cache of 154 MB versus 7,168 MB. Jet-Nemotron-4B: 1,271 tok/s, about 21x. Secondary write-ups cite peaks of 53.6x decode and 6.1x prefill, which I did not confirm in the paper — [arXiv 2508.15884](https://arxiv.org/abs/2508.15884v3); [papernotes summary](https://en.papernotes.org/NeurIPS2025/llm_efficiency/jet-nemotron_efficient_language_model_with_post_neural_architecture_search/)
  - Against Qwen2.5-1.5B the gap is smaller (about 12x, by my reading of the table data as summarized). One anecdotal RTX 3090 test reports about 6.5x — [hardware-corner.net](https://www.hardware-corner.net/llm-vram-usage-45x-reduction-jet-nemotron-20250826/)
- Nemotron-H / Nemotron Nano 2 hybrid: about 8% attention layers, the rest Mamba-2. Up to 6x throughput over Qwen3-8B at 8k input / 16k output — [arXiv 2508.14444](https://arxiv.org/abs/2508.14444v2)

**MHA/GQA → MLA**
- TransMLA (NeurIPS 2025): converts GQA models (Llama-2/3, Qwen) to MLA so they run on DeepSeek inference stacks (vLLM, SGLang). Per-head RoPE blocks DeepSeek's "absorb" trick, so TransMLA isolates RoPE into a few dims (RoRoPE, FreqFold); the key up-projection can then be absorbed into the query projection. Results — [arXiv 2502.07864](https://arxiv.org/html/2502.07864v4); [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/75d13a472f570755af2a4ae4ac3d6724-Abstract-Conference.html):
  - 93% KV-cache compression on LLaMA-2-7B gives a 10.6x speedup at 8K context in vLLM.
  - At 68.75% KV compression without fine-tuning, the average drop over 6 benchmarks is 1.65%.
  - Recovery fine-tuning takes about 6B tokens.
  - The authors report a 21.85% drop for MHA2MLA in their comparison, and say TransMLA needs about 4.9% of MHA2MLA's training tokens. These are author-run comparisons.
- MHA2MLA (Ji et al., 2025, concurrent): converts MHA to MLA using partial-RoPE plus low-rank SVD of the KV projections. I did not open the paper; numbers come only from TransMLA's comparison above.

**Sparse attention retrofits**
- DeepSeek-V3.2(-Exp) DeepSeek Sparse Attention (DSA), added by continued training on V3.1-Terminus — [DeepSeek-V3.2 report, arXiv 2512.02556](https://www.emergentmind.com/papers/2512.02556):
  - Mechanism: a lightweight "lightning indexer" scores past tokens per query, and top-k (k=2048) KV entries are selected for full MLA attention. Core attention cost goes from O(L²) to O(L·k), but the indexer itself is still O(L²), just cheap — [Daily Dose of DS](https://blog.dailydoseofds.com/p/deepseek-cracked-the-ol-attention)
  - Training: warm-up with dense attention while training only the indexer (KL to the dense attention distribution), then sparse training of all weights.
  - Reported: near-identical benchmarks to V3.1-Terminus. Cost per million tokens at 128K falls from about $0.65 to about $0.35 for prefill and from about $2.4 to about $0.8 for decode (secondary source, read from the report's charts). API price cut by more than 50%. Secondary sources disagree on exact figures — [Daily Dose of DS](https://blog.dailydoseofds.com/p/deepseek-cracked-the-ol-attention); [DataCamp](https://www.datacamp.com/tutorial/deepseek-v-3-2-exp)
- NSA (Native Sparse Attention, DeepSeek, Feb 2025; ACL 2025): hierarchical compressed, selected and sliding-window branches, trained natively rather than retrofitted. It reports large forward, backward and decode speedups at 64K. [not re-verified, no numbers quoted] — [arXiv 2502.11089](https://arxiv.org/abs/2502.11089)

### Inferences
- Conversion is now cheap enough (under 1B to 20B tokens) for individuals to run at 7B scale. RADLADS-style conversion of a Qwen 7B model is plausibly a single-node job.
- The biggest throughput multipliers (Jet-Nemotron about 47x, TransMLA 10.6x) come from shrinking the KV cache at long context, where memory bandwidth and batch capacity bind. At short context the gains shrink sharply. Reported multipliers depend on context length, so long-context and short-context numbers should never be compared without stating the length.
- Hybrid designs (a few full-attention layers kept) are the dominant 2025–2026 production choice: Nemotron-H, Nemotron Nano 2, Jet-Nemotron. Pure linear conversions trade away more recall- and retrieval-heavy quality.
- MLA conversion has a distinct advantage: it reuses existing optimized DeepSeek kernels in vLLM and SGLang, so the speedup does not depend on new kernels.

### Gaps
- KV-cache compression and eviction (SnapKV, H2O, PyramidKV, KV quantization, 2025–2026 successors such as learned eviction) was not researched in this session. Another researcher may cover it; otherwise it is a gap.
- No independent benchmarks of Jet-Nemotron, RADLADS or TransMLA checkpoints in vLLM or SGLang.
- I found no DSA-style indexer retrofit applied to non-DeepSeek open models, and no measured DSA tokens/s numbers.
- Qwen3-Next / Qwen3.5 (Gated DeltaNet hybrids) and Kimi Linear are architecture choices at pretraining, not conversions. They were not researched here.

## 4. Early exit / layer skipping / dynamic depth

### Takeaway
The main open, reproducible result is still LayerSkip (Meta, 2024): about 1.8x–2.2x via early-exit self-speculative decoding. It requires training with layer dropout and early-exit loss, so it is not a free post-hoc retrofit. My searches did not find a 2025–2026 dynamic-depth method with comparably strong, widely adopted results.

### Cited Findings
- LayerSkip [background, 2024, ACL 2024]: training uses layer dropout (low rate in early layers, high rate in late ones) plus a shared early-exit loss. Inference drafts with the early layers and verifies with the remaining layers, sharing KV and activations, so it needs less memory than separate-draft speculative decoding. Speedups: 2.16x on CNN/DM summarization, 1.82x on coding, 2.0x on TOPv2. Code and Llama checkpoints are released — [arXiv 2404.16710](https://arxiv.org/abs/2404.16710); [Meta AI](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding)
- Static depth pruning (removing whole layers, then distilling) is part of Minitron and Puzzle, where Puzzle can replace whole blocks with no-ops or linear layers. This is the production form of "layer skipping" — [Puzzle, arXiv 2411.19146](https://arxiv.org/html/2411.19146v5)

### Inferences
- Dynamic per-token depth is awkward for batched GPU serving: different tokens exit at different layers, which breaks batch uniformity and leaves KV for skipped layers missing. Industry has mostly turned depth reduction into static pruning (Puzzle, Minitron) or self-speculation (LayerSkip), where the verifier keeps outputs exact.

### Gaps
- My single search did not find 2025–2026 dynamic-depth or router-based layer-skipping papers (for example mixture-of-depths retrofits or adaptive-computation work) with measured serving speedups. This area is under-covered here.

## 5. Diffusion LMs, including conversion from autoregressive models

### Takeaway
Diffusion LLMs have become a credible speed play. Closed models (Mercury, Gemini Diffusion) report about 700–1,500 tok/s at roughly small-AR-model quality. Open models (LLaDA, Dream, SDAR) depend on training-free accelerators such as Fast-dLLM (KV cache plus confidence-aware parallel decoding) to be competitive: about 4–8x typical speedups over the vanilla diffusion model, up to 27.6x in one best case. AR→diffusion conversion is now routine. Dream was initialized from Qwen2.5-7B; SDAR converts an AR model to block diffusion with about 50B tokens.

### Cited Findings
- Mercury Coder (Inception Labs, Jun 2025): Mercury Coder Mini and Small run at 1,109 and 737 tok/s on H100, independently measured by Artificial Analysis. The vendor claims 5–10x over speed-optimized AR models — [Mercury tech report, arXiv 2506.17298](https://arxiv.org/html/2506.17298v1); [Inception](https://www.inceptionlabs.ai/introducing-mercury)
  - Mercury 2 (about Feb 2026) claims 1,000+ tok/s and 5x over leading speed-optimized models. Vendor and launch-listing claim only — [hunted.space listing](https://www.hunted.space/product/mercury-412/launches/mercury-2)
- Gemini Diffusion (Google DeepMind, May 2025, closed and waitlisted): 1,479 tok/s sampling excluding overhead, plus 0.84 s overhead. Benchmarks comparable to Gemini 2.0 Flash-Lite — [Gigazine report on Google's announcement](https://gigazine.net/gsc_news/en/20250521-google-gemini-diffusion)
- Fast-dLLM (NVIDIA, ICLR 2026): training-free; adds a block-wise approximate KV cache and confidence-aware parallel decoding to LLaDA and Dream. Code: NVlabs/Fast-dLLM — [arXiv 2505.22618](https://arxiv.org/pdf/2505.22618v2); [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5d8d4e6061c3ba96c240b7fa1ae3471d-Abstract-Conference.html); [NVIDIA project page](https://research.nvidia.com/labs/eai/publication/fast-dllm)
  - KV cache alone: 2x–3.6x, with accuracy within 1–2 points.
  - KV cache plus parallel decoding: up to 27.6x end-to-end on LLaDA for GSM8K 8-shot at 1,024 generation length. Accuracy goes from 77.3 to 76.0, at 19.3 tok/s, against vanilla LLaDA.
  - Project summary: 8.1x on GSM8K at 256 tokens, 4.0x on HumanEval at 512 tokens. A secondary review cites 7.8x for Dream-Base on MBPP at 512 tokens.
  - Even after acceleration, LLaDA reaches only about 19 tok/s here. The multiplier is against a very slow baseline.
- SDAR (Oct 2025): converts a pretrained AR model to blockwise diffusion, keeping AR order across blocks and running parallel diffusion within each block, using about 50B extra tokens. The 2B chat conversion is roughly on par with AR-2B-Chat: MMLU +2.2, GPQA +1.9, MATH −3.1, HumanEval −2.1. A 30B MoE variant beats its AR counterpart on GPQA and ChemBench. I found no tokens/s figure — [arXiv 2510.06303](https://arxiv.org/html/2510.06303v3)
- Dream 7B (HKU / Huawei, 2025): an open diffusion LM initialized from AR Qwen2.5-7B weights. [not re-verified] — [Dream 7B blog](https://hkunlp.github.io/blog/2025/dream/)
- Block diffusion (BD3-LMs, ICLR 2025): interpolates between AR and diffusion, which makes KV caching across blocks possible. [not re-verified] — [arXiv 2503.09573](https://arxiv.org/abs/2503.09573)

### Inferences
- The open-source diffusion story in late 2026 is block diffusion: semi-autoregressive, AR across blocks. Converting from an AR checkpoint (Dream, SDAR) avoids pretraining from scratch, and KV caching becomes possible.
- Headline diffusion multipliers usually compare against vanilla diffusion decoding, not against an optimized AR model with speculative decoding. The fair comparison for a report is tok/s versus an AR model of similar quality on the same hardware. Only the Artificial Analysis Mercury numbers approach that.

### Gaps
- No independent tok/s comparison of open diffusion models (Dream, LLaDA, SDAR) against same-size AR models served with vLLM plus speculative decoding.
- I did not check LLaDA 2.x / LLaDA-MoE, Seed Diffusion, or other 2026 open diffusion releases.
- Gemini Diffusion's numbers come from Google's announcement as relayed by press. There is no technical report.
