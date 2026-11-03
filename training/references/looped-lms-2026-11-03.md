# Looped language models: Dory1.0-4B and Elastic Latent Recurrence — 2026-11-03

Two NVIDIA releases from October 2026 on looped (latent-recurrent) language models,
read against our own runs. Numbers below are the authors'; nothing here is measured
by us. The implementation proposal is [plans/LOOPED_LM.md](../plans/LOOPED_LM.md).

- **Dory1.0-4B**: [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
  (revision `f6c832e`, released 2026-10-16), its `config.json` and safetensors header,
  and the reference implementations in
  [vllm-project/vllm#60149](https://github.com/vllm-project/vllm/pull/60149)
  (`vllm/model_executor/models/dory.py`) and
  [ggml-org/llama.cpp#30306](https://github.com/ggml-org/llama.cpp/pull/30306).
- **Elastic Latent Recurrence (ELR)**: Ladhak, Acharya et al.,
  [One Model to Loop Them All: Elastic Latent Recurrence for Language Models](https://arxiv.org/abs/2610.12582)
  (arXiv:2610.12582v1, 2026-10-08). It trains Koi-4B; the Dory card does not cite it,
  and Dory is a different architecture (below).

## What a looped model is

The layer stack is split into an input block (prelude), a shared recurrent block and
an output block (coda). The recurrent block runs `k` times per token, so effective
depth is `L_in + k * L_rec + L_out` at a fixed parameter count; `k` is chosen at
inference. Earlier work: Universal Transformers (2019), Huginn (Geiping et al., 2025,
mean recurrence 32), Ouro (Zhu et al., 2025, whole stack repeated, no re-injection),
Parcae, LoopFormer, Retrofitted Recurrence and Relaxed Recursive Transformers (both
convert a trained non-looped model).

## Dory1.0-4B

Architecture, from `config.json`, the weights and the vLLM code:

| | |
| --- | --- |
| Parameters | 3,975,229,440 (BF16); untied 131,072 × 2,560 embedding and output layer, 671M of them |
| Layers | 72 single-sublayer layers, alternating attention and SwiGLU MLP: 36 attention + 36 MLP |
| Split | 16 input / 40 recurrent / 16 output layers, i.e. 8 / 20 / 8 attention+MLP pairs |
| Width | hidden 2,560, MLP 7,168, 16 query / 4 KV heads, head dim 256 |
| Attention | 3 sliding-window (1,024) layers per full-attention layer; 9 full layers: 2 input, 5 recurrent, 2 output. RoPE base 10⁴ (SWA) / 10⁷ (full); sigmoid output gate |
| Normalization | nGPT: hidden states, q and k live on the unit sphere; each sublayer is a learned per-channel slerp `h ← norm(h + α(norm(f(h)) − h))` instead of a residual add |
| Loop | `state = h_in`; each loop `x = norm(state + h_in)`, run the 40 recurrent layers, `state = norm(x)`; default `n_recurrent_loops = 2` |
| Context | 256K (6T pretraining tokens, ~4.5T of them in NVFP4, then 300B of context extension) |

Parameter split, computed from the weights: 734M input, 1,835M recurrent (46%),
734M output, 671M embedding/output. Non-embedding matmul weights touched per token
are 3.30B at one loop, 5.14B at two and 8.81B at four.

Per-loop KV caching gives the recurrent full-attention layers a separate cache per
loop. Full-attention KV is 4 KiB per token per layer (4 heads × 256 × K and V × 2
bytes), so 1 GiB per layer at 256K: 9 GiB at one loop, 14 GiB at two, 24 GiB at four.
`last_loop` mode keeps one cache per layer that every loop overwrites (earlier tokens'
final-loop keys, the current token's own loop-`i` key), so the cache stays at 9 GiB.

Model-card results (NVIDIA's harness; Qwen3.6 SWE-bench is reported, not rerun):

| Benchmark | 1 loop | 2 loops | 4 loops | Qwen3.5-4B | Qwen3.5-9B | Gemma 4 26B A4B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SWE-bench Verified | 50.3 | 59.5 | 61.8 | 54.7 | 61.0 | 57.4 |
| Terminal-Bench 2.1 | 12.5 | 18.1 | 22.2 | 15.0 | 24.0 | 35.0 |
| SciCode | 27.3 | 34.9 | 35.1 | 25.8 | 28.2 | 40.3 |
| GPQA Diamond | 69.2 | 72.4 | 72.2 | 77.7 | 80.9 | 80.4 |
| HLE (text) | 8.3 | 10.9 | 10.4 | 11.3 | 15.4 | 17.8 |
| IFBench | 74.3 | 77.9 | 79.0 | 53.8 | 59.0 | 77.9 |
| AA-LCR | 63 | 71 | 73 | 64.7 | 68.4 | 64.6 |
| AA-Omniscience accuracy | 10.0 | 10.1 | 10.0 | 15.0 | 18.1 | 21.1 |

- The second loop gives most of the gain; four loops help agentic tasks most
  (Terminal-Bench +9.7 points over one loop) and do nothing for knowledge
  (AA-Omniscience flat at 10.0, the lowest in the table).
- Higher loop counts produce fewer output tokens for the same tasks (card figure).
- The card does not state the training loop-count distribution or how the loop count
  is sampled.

## Elastic Latent Recurrence (arXiv:2610.12582)

The goal is a looped model with no **one-loop tax**: served at `k = 1` it must match a
non-looped model of the same size trained on the same data (Huginn trails its
fixed-depth baseline 34.9 vs 46.4 on ARC-Easy at one loop). About the middle half of
the layers are recurrent, the rest split evenly between input and output. The recipe:

1. **Adaptive-delta state update.** `s_0 = h_in`; each loop `s̃ = s + h_in`,
   `r = R(s̃)`, `β = σ(W s̃ + b) ⊙ softplus(γ)`, `s ← s + β ⊙ (r − s)`. W, b, γ start
   at zero (step ≈ 0.35) and add `d² + 2d` parameters (~0.2%). On GPT-0.5B (300B
   tokens, 8/14/8 split) it is the only rule matching the baseline at one loop and is
   lowest at every depth; replace (`s = r`) is worst at one loop and gets worse from
   four to five loops; residual (`s += r`) stops improving after three loops; a fixed
   per-channel rate is 0.006–0.007 behind. A model trained only at `k = 1` beats the
   baseline (1.660 vs 1.685): the gate acts as a learned skip.
2. **Mixed depths within each update.** `k ~ Uniform{1..K}` per microbatch, full
   backprop through all loops, plus a rotating fraction `f` of data-parallel ranks
   forced to `k = 1`, so one-loop and deep gradients share every update; `f` is chosen
   so `p_1 ≈ 1/3`. Without overrides one-loop loss is +0.0225 over the baseline.
   One `k` per step (Huginn's synchronized sampling) misses one-loop parity even at the
   same `p_1`. Independent per-rank `k` is best but 36% slower per step.
3. **Loop-conditioned modulation** (progress `i/k` and budget `k/K` through an MLP:
   AdaLN/FiLM scale and shift, a gated low-rank adapter and an output scale per layer,
   all zero-initialized). +6.3% parameters on GPT-0.5B; one-loop loss unchanged, −0.009
   at `k = 2`, −0.022 at `k = 5`.
4. **Delayed mixed depth.** Train at `k = 1` for most tokens, then switch to mixed
   depth (and graft the zero-initialized conditioning). Starting at 75% of tokens costs
   ~1.2× baseline compute (vs 1.79× from scratch) and keeps 84% of the five-loop gain;
   with conditioning it beats mixed depth from scratch at every `k`. Optimizer settings
   are the baseline's, unchanged.
5. **Last-loop KV caching** at inference, and **cross-loop route replay** for MoE
   (route on the first loop only: −11.7% update time, no measurable accuracy change).

Results against non-looped baselines on the same data (ten-task average: MMLU,
MMLU-Pro, ARC-C, GSM8K, WinoGrande, CSQA, RACE-high, HumanEval, MBPP, MATH-500):

| Model | Baseline | ELR `k=1` | `k=4` | Largest gains `k=1→4` |
| --- | ---: | ---: | ---: | --- |
| GDP-Hybrid-4B, 1T tokens, 14/24/14 split | 65.65 | 65.13 | 72.38 | GSM8K +11.8, MATH-500 +11.3 |
| GDP-Hybrid-30B-A3B, 300B tokens | 65.04 | 66.08 | 70.90 (replay) | MMLU-Pro +7.5, GSM8K +6.5 |

Two and three loops capture 79% and 96% of the 4B model's `k=1→4` gain. At four loops
the 4B model has 2.33× the baseline's time to first token and 2.06× its decode
latency. The GDP-Hybrid layout is 23 Gated DeltaProduct, 6 attention and 23 MLP
layers (Mamba-2 replaced by GDP in the Nemotron 3 Nano layout).

**Koi-4B** (7T tokens, mixed depth for the second half, 256K mid-training, SFT, 200
RLVR steps with loop counts 4:1:1:1 favouring one loop): estimated Artificial
Analysis Index v4.1 of 18, 26, 28, 29 at `k = 1..4` (Qwen3.5-4B: 18, Qwen3.5-9B: 22).
Output tokens per rollout fall 21.8%, 29.5%, 32.7% at `k = 2, 3, 4`, so decode time per
rollout rises only ~6% at two loops (1.36× per token × 0.78× tokens). Last-loop KV
caching at `k = 4` changes scores by −3.0 to +2.3 points (AA-LCR worst) at one-loop KV
memory instead of 2.5×; the GDP layers keep per-loop state, which is under 25 MiB per
request.

## Dory against ELR

| | Dory1.0-4B | ELR / Koi-4B |
| --- | --- | --- |
| Backbone | nGPT, SWA + full attention | Gated DeltaProduct + attention hybrid, RMSNorm |
| Recurrent share of layers | 40 / 72 | 24 / 52 |
| State update | replace, renormalized: `state = norm(R(norm(state + h_in)))` | gated adaptive delta |
| Loop conditioning | none | progress and budget modulation |
| KV modes | per-loop and last-loop | per-loop and last-loop |

ELR found replace the worst update rule in a pre-norm Transformer. In nGPT every
sublayer is already a learned-rate interpolation on the sphere, so that result does not
transfer directly; neither source compares the two.

## Compared with our experience

- **Architecture.** Our [720M hybrid](../hybrid/README.md) (12 KDA + 4 full-attention
  layers, 3:1) is close to ELR's GDP-Hybrid: KDA and Gated DeltaProduct are both gated
  delta-rule linear attention with fixed-size state. ELR's recipe and its measured
  per-loop linear-attention state cost apply to it more directly than Dory's nGPT
  design does.
- **What looping buys.** The [720M v1 benchmarks](../plans/HYBRID_720M.md) sit near
  SmolLM2-360M on commonsense tasks with MMLU, GSM8K and code near chance, and
  [Hybrid v2](../plans/HYBRID_V2.md) concluded that below 2B the gap is data and
  training, not parameters. Looping adds compute per token, not stored knowledge:
  AA-Omniscience is flat across Dory's loops and moves 8.1 → 9.3 for Koi. ELR's largest
  gains are GSM8K and MATH-500 (+11 points), the reasoning that Hybrid v2's stage 3
  targets. Looping is therefore no substitute for the data work, but a candidate for
  reasoning gains at fixed size.
- **Cost on our infrastructure.** The 720M 40B run measured 27.9% MFU at ~372.7K
  tokens/s on 8 × H100. ELR's delayed schedule (mixed depth for the last 25% at
  `K = 4`, `k̄ = 2.35`) costs about 1.16–1.2× that baseline's compute; mixed depth from
  scratch about 1.6–1.8×.
- **Repetition.** Our 124M runs failed on degenerate loops in greedy and
  low-temperature generation ([what didn't work](../NEGATIVE_RESULTS.md)), and lower
  held-out loss did not predict better free generation. Both sources report shorter
  outputs at more loops and Koi's last-loop mode reaches the output cap less often, but
  neither measures repetition; it would have to be measured with the
  [sampling toolkit](../analysis/sampling/README.md).
- **Phone inference.** The weights, not the loops, fix memory: a looped model keeps a
  4B footprint while spending 5–9B-class compute per token. Batch-one decode is
  bandwidth-bound and each loop re-reads the recurrent weights, so decode time grows
  roughly with effective depth (ELR measured 2.06× at four loops on GPUs); the loop
  count becomes a per-request latency/quality setting in
  [phone inference](../../inference/phone/README.md). One kernel set serves every loop
  count.
- **Speculative decoding.** With no one-loop tax, the same checkpoint at `k = 1` is a
  natural drafter for itself at `k = 4`: shared weights, no separate drafter training
  as in the [latent workspace](../../exploration/latent-workspace/) DFlash work.
  Neither source tests this.
- **Optimizer.** ELR trained its 30B-A3B models with Muon and AdamW, as our hybrid
  does, and kept baseline learning rates; Parcae reports looped models diverging at
  learning rates that non-looped models tolerate.
