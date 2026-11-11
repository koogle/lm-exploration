# Marin 535B-A23B: architecture decisions, their reasoning, and lessons for our hybrid model

Reviewed 2026-11-11 against Larry Dial,
[Architecture Decisions for Marin 535B-A23B](https://openathena.ai/blog/marin-535b-architecture/)
(Open Athena blog, 2026-11-09). All numbers below are author-reported; none are
our measurements. The post's code, ablations and fits are linked from
[marin-community/marin#8435](https://github.com/marin-community/marin/issues/8435);
`#nnnn` below are issues in that repository.

The value of the post is less any single component than the record of *why* each
choice was made under a fixed deadline. This note keeps that reasoning next to each
decision and tags what drove it, because a choice made for MFU or schedule risk at
535B does not carry the same weight for us as one made on measured quality.

Most ablations are Paloma macro loss at hidden width 768 or 1024 (d768, d1024).
The post quotes them as **compute-efficiency multipliers** against a fitted baseline
`L(C) = 1.6 + 88.90·C^-0.0941` (their May 2026 MoE recipe): "1.18×" means the
baseline would need 18% more compute for the same loss. A multiplier is relative to
Marin's baseline and optimizer, not a transferable speedup.

Driver tags: **Q** measured quality, **S** systems / realized MFU, **M** memory or
KV cache, **St** stability or risk, **T** deadline, **E** engineering constraint.
"No reason given" means the post states the choice without a rationale.

## How they decided

- **Risk dominates.** The author places the run in the "approaching the frontier"
  camp, where the main cost is the run failing or missing its December deadline.
  Three goals: the best model by December (with realized MFU the dominant lever), a
  demonstration of stable, predictable MoE training (a preregistered loss forecast
  and a live W&B project), and a base for SFT/RL with inductive biases close to the
  literature.
- **Deviation needs a bigger payoff.** Two prior runs (129B-A16B on 1T tokens and
  67B-A2B "Snowball" on 10T, both within 1% of their loss targets with no spikes)
  are the stability evidence. Any change away from them gives up part of that
  evidence and must earn it back elsewhere.
- **Every idea is gated on realized MFU on GB200.** Several components that the
  author reports as promising (KDA, MLA, attention residuals) were deferred only
  because they did not yet run fast enough.
- **Token dropping was judged the biggest single risk**, so any feature that reduced
  it was biased toward inclusion even with near-zero quality effect.
- **Find unknown unknowns on the hardware.** Run as long as possible on the real
  system and read plots by hand; this found the `lm_head` norm collapse (#8073) and
  the embedding gradient spikes.
- **Diagnose by adding a degree of freedom and watching it.** A learnable scalar on
  the latent MoE climbed to about 4, exposing an initialization scale problem.
- **Judgment and external validation where compute ran out**, stated openly (for
  example Kimi K2's 8-of-384 experts supporting their own granularity result).
- **A scaling ladder before the hero run**: four rungs from 5e19 to 9e21 FLOPs at 791
  tokens per active parameter, to confirm the projection, catch abnormal dynamics and
  give a forecast to track. The largest rung was also used to pre-screen SFT/RL.

## Decision log

### Size and shape

| Decision | Alternatives | Stated reasoning | Evidence | Driver |
| --- | --- | --- | --- | --- |
| 535B total, ~23B active | 129B-A15B (57 days), 360B-A23B (82 days, fits one rack with FSDP), 707B-A23B (99 days, two racks), 360B-A21B with MLA (120 days); all for 20T tokens | Six factors: deadline (768 GB200s; about 25 days setup, 80 training, 15 buffer), compute optimality, MFU (a 72-GPU rack, 64 healthy; spilling past one rack costs MFU), 23T deduplicated tokens, 2–3 weeks for RL, and serving cost (explicitly deprioritized). Target: the biggest model on one rack with enough active parameters to take about 80 days. Fixed-capacity expert parallelism made 535B fit where 360B was the FSDP limit (85 days) | #7673, #7201 | T, S, E |
| 18T tokens, about 13× past compute-optimal | ~60 tokens per active parameter (~1.4T tokens) | Isoflop curves put MoEs near 60 tokens per active parameter; 10× overtraining costs about 2× compute. Training time was fixed, so the token count followed from it | #6074; Figure 12 projections | T, Q |
| 48 layers, width 6,144 (width:depth 128) | Deeper models | Fewer, wider layers run faster on their stack; depth enhancers (attention residuals, mHC, skip connections) were not efficient in time, and without them very deep nets may not use their parameters. Command A+ at 32 layers shows depth is not required. Heterogeneous layers under `jax.lax.scan` did not work cleanly in time | No depth ablation; hero depth behavior differed from the ladder (#8818) | S, E, T |
| Batch 46.1M tokens at 4K | — | The most that fit in memory; more documents per batch lowers routing variance | — | M, St |

### Attention

| Decision | Alternatives | Stated reasoning | Evidence | Driver |
| --- | --- | --- | --- | --- |
| Sliding window 2,048 | 512 (Inkling), 1,024 (Gemma 4), 128 (gpt-oss) | Window 512 cost 1.10×, while vLLM on GB200 measured 2,048 only 2–4% slower than 512 on long-context serving. Trinity Large and Command A+ use 4,096 at scale | d768: 3.0208 vs 3.0334 at 512; #8017 | Q, S |
| 3 local : 1 global | — | No reason given | — | — |
| GQA 4:1 on local layers, 8:1 on global | MHA, no GQA, MLA | Kept for inference KV cache. Global caches grow with sequence length and dominate memory, so they get the aggressive ratio; local caches are capped at 2,048 tokens, so fewer KV heads there would save little and cost quality. No reason given for not using MLA in the hero | MHA 2.9985 vs 3.0208 (1.18×) at d768 | M, Q |
| 50% partial RoPE local, NoPE global | Relative position embeddings (deferred) | No reason given | — | — |
| QK RMSNorm with a fixed ×1.3 | No multiplier, learnable per-head gain, no QK-norm | Fixed beat learnable; the learnable scale drifted from 1.3 to about 2.0 without gain. Attention logits are a likely failure point, so any manual multiplier change for context extension (as with YaRN) should be settled in advance | d768: 3.0208 fixed, 3.0217 learnable, 3.0261 none, 3.0281 no QK-norm; d2048: 2.6477 vs 2.6484 | Q, St |
| QK multiplier 1.3 → 1.66 for 4K → 65K; for 262K, 1.66·(1 + 0.1·ln(n/65,536))² on global layers | — | Tuned by short tests on the hero model; a squared, floored variant of Inkling's log scaling, in the spirit of SSMax. The post presents the 262K rule as a design and does not say that stage ran | #8435 | Q, St |
| Gated attention | — | Standard in the literature (Qiu et al., arXiv 2505.06708); no individual ablation | — | — |
| Exclusive self attention (XSA) | None; zero-init attention output (MAI-Thinking-1), which hyperball would keep at zero; a learned zero-init scalar, judged too big a change to rush | Near-zero cost and it reduces early token dropping, the run's biggest risk. Intuition: attention starts nearer a no-op, so router inputs move less while attention learns | Quality: 3.0208 vs 3.0221 at d768. Early drop rate (steps 30–100, hero size, accelerated schedule): 5.6% vs 12.7% (#8227) | St, S |
| Short convolutions, kernel 4, on keys, attention output, MLP output; identity init; reset at document boundaries | Fewer sites, kernel 2–3, V conv, PKO-style K init | From Inkling. The model is shallow (48 layers vs 60+ in many large MoEs), so an extra cross-token path helps; a K conv allows one-layer induction heads. **No V conv** because it gave the smallest gain and blurs XSA: a value mixing its neighbors would lose local context when XSA removes the self direction. The PKO init learned faster early but the gap closed by about step 1,500 | d1024: all four sites 3.0378 (1.19×), k=3 1.18×, k=2 1.13×, global layers only 1.10×, MLP-out only 1.08×, V only 1.03×, none 3.0610. d768: final set 3.0208, none 3.0489, with V 3.0236 (#7585) | Q |

### Mixture of experts

| Decision | Alternatives | Stated reasoning | Evidence | Driver |
| --- | --- | --- | --- | --- |
| 8 of 384 routed experts | 4/192, 16/768 | 8/384 gave 1.13–1.16×; 16/768 added little for its throughput cost. With 2× latent compression the expert matrices are square, so Muon's orthogonalization balances channels and avoids the neuron imbalance NorMuon and Aurora address. Kimi K2 also uses 8/384 | d768: 3.0289 / 3.0091 / 3.0009; d1024: 2.7810 / 2.7674 | Q, S |
| 2 shared experts (⅓ of MLP neurons) | — | A stronger dense backbone in case routing failed | None | St |
| Latent MoE, 2× compression, learnable RMSNorm on the latent | Plain experts; a learnable scalar | Halves all-to-all traffic. It first lost (0.74–0.75 compute efficiency); a learnable scalar grew to about 4, cancelling the 0.25 scale from the two 0.5/√d initializations. An RMSNorm in MLA's position fixed it: 1.25× over no intervention | Figure 13; #6822 | S, Q |
| Sigmoid router; top-k weights renormalized to sum 2.5 | — | No reason given for either | — | — |
| Top-k quantile balancing on raw logits, 10,000-bin histogram, no auxiliary loss | Auxiliary loss (slower); bias after sigmoid (DeepSeek-V3, Kimi K3); per-GPU quantile averaging | Quantile balancing rebalances in 1–2 steps. Raw logits balance faster than post-sigmoid, which matters because the run drops tokens. Histograms are independent of GPU count, match averaging on loss and balance slightly better; 10,000 bins cost no visible MFU | Drop rate ~2% vs 3.4% (K3-style) vs 4.9% (averaging); entropy 4.848 / 4.821 / 4.786 of max 4.852 (#8033) | Q, St, E |
| No leading dense layers | Dense first layers | Routing entropy looked healthy without them, and more sparsity can help efficiency. Reconsidering them for dropping was blocked by `jax.lax.scan` heterogeneity. Open issue: routed experts in layers 1–2 get abnormally small gradients | #8818 | Q, S, E, T |
| Fixed-capacity expert parallelism, then ragged all-to-all at 21% of the run | FSDP only (caps at 360B); `ragged_all_to_all` (too slow at first) | Fixed capacity made 535B fit at the price of token dropping, a judgment call tested directly: drop rates rose 5.5% → 7.1% from d768 to d2048, four racks did not change the trend, and 4K → 65K raised drops from 7% to 37% while eval stayed within 0.01 of dropless FSDP. The concern is a feedback loop where a spike raises drops. An NVIDIA device-initiated kernel later brought drops to ~0.01% (4K), 0.14% (16K), 0.03% (65K) | #8062 | S, St, E |

### Norms, loss and embedding

| Decision | Alternatives | Stated reasoning | Evidence | Driver |
| --- | --- | --- | --- | --- |
| Embedding outside the hyperball, plain Adam, RMSNorm after lookup | Embedding in the hyperball | Inside the hyperball, gradient spikes grew with LR (about 60× at 1.2× LR, 700× at 1.4×). Under Adam the embedding norm grows freely (177 → ~7,800); the RMSNorm keeps the output at unit scale, as μP expects, and the growth acts as a built-in LR decay on the embedding | Removing the RMSNorm cost ~19% (d768, #6442); #4569, #5773 | St, Q |
| `lm_head` in AdamH | MuonH, Cautious AdamW, AdamW, Adam | AdamH beat Cautious AdamW by 1.08× at scale (Cautious AdamW won marginally at very small scale). The LR recipe was tuned on AdamH, and compute went to more pressing ablations | #8350 | Q, T |
| Learnable RMSNorm gain before `lm_head`, plus logit z-loss 1e-4 | No z-loss | Pinning `lm_head`'s norm does not fix logit scale because the gain is free to grow; without z-loss the loss blew up in the overtrained regime. No sweep of the weight is reported | d1024: loss rose after ~step 4,000 to 3.39 at 11,000 (reference ~2.1); gain 557 vs 183 | St |
| Gated norm (rank-128 gate) in MuonH | Adam, AdamH, Muon | Kept from prior runs for stability on Kaiyue Wen's advice; Qwen's architecture study reports it improves stability. Optimizer choice from a comparison with no clear trend | d1024: 3.1392 / 3.1390 / 3.1366 / 3.1352; differences under 0.011 | St, Q |

### Optimizer and schedule

| Decision | Alternatives | Stated reasoning | Evidence | Driver |
| --- | --- | --- | --- | --- |
| Hyperball optimizers: MuonH on matrices, AdamH on `lm_head`, plain Adam on embedding, router, attention gate, norm gains, convolution kernels | — | The hyperball keeps each matrix at its initial norm; prior runs gave stable, straight-line loss curves. Reasons are given for the embedding and `lm_head` exceptions only | Prior runs | St, Q |
| `adam_lr = 0.087571 · tokens^-0.3461 · hidden^-0.3448 · sqrt(tokens_per_batch)`, `muonh_lr = 13/3 · adam_lr` | — | Empirical fit in the style of their Delphi work; no reason given for the 13/3 ratio | #7856 | Q |
| 1% warmup | None | Not needed under Muon, kept for routing stability at the start | — | St |
| Linear decay to 5% of peak | WSD; `1 - sqrt(t/T)` | Linear beat WSD at d768. The author ties this to the hyperball: without it, parameter norms grow early and decay the effective LR even during a WSD stable phase; with fixed norms, decay must come from the schedule. `1 - sqrt` was slightly better (3.2642 vs 3.2670) only at 1.4× peak LR, which would endanger the riskiest first 3% of the run | #7717 | Q, St |
| Final LR 5%, not 0 | Decay to 0 (strongest in the sweep) | Leaves room for SFT and RL; checkpoint averaging may favor an even higher floor | #7717 | T |
| Context 4K → 16K → 65K with an improved data mix | — | Risk controlled through the drop-rate tests above; switch steps are only in figures | #8062 | St, Q |

### Deferred or rejected

| Decision | Stated reasoning | Driver |
| --- | --- | --- |
| No FP8 / NVFP4 | Speedup not worth the risk at 20%, later 30% MFU | S, St, T |
| No multi-token prediction | Memory, realized MFU, and separate drafters (DSpark) working well; not a quality comparison | M, S |
| No attention residuals, mHC or skip connections | Not efficient in time; attention residuals may fix the dead early experts | E, T, S |
| KDA, MLA, relative positions deferred | Reported as strong in combination, gated on realized MFU; no numbers given | S |
| No over-encoded hashed n-gram embeddings | MFU and memory; unclear behavior under data-mix changes or multiple epochs, and a risk for a first large RL recipe | S, M |

## Outcome

The ladder fit `L = 1.5 + A·C^-α` forecast 2.039 Paloma macro; the dropless final
loss was 1.982. One rung (27.7B-A1.2B) died at 81% from a hardware failure and was not
resumed. The post gives the hero's compute as 2.7e24 FLOPs in the text and 2.6e24 in
the Figure 12 table. Roughly 0.02 of the late macro improvement is attributed to
Paloma PTB contamination at step 362K (#9874). A cross-entropy kernel precision fix
(PR #8368) "may also have helped run stability".

Not stated in the post: vocabulary/tokenizer, peak learning rate, weight decay and
training precision.

## Author's open questions

Gradual alternatives to 1–2-step quantile rebalancing at data-mix changes; starting
with smaller local windows (128) and widening; learned per-head QK multipliers for
long-context retrieval; a final LR above 0 given checkpoint averaging; `lm_head`
output scaling at large widths; MFU-gated KDA, attention residuals, MLA and relative
positions; FP8/NVFP4; hashed n-gram embeddings; high-depth stress tests; the gated-norm
optimizer group at large widths; post-training and richer evals as feedback for
architecture; multi-token prediction; dead routed experts in early layers (perhaps a
temporary early auxiliary loss); and the decision process itself.

## Our hybrid 720M for comparison

| Component | Marin 535B-A23B | Our hybrid 720M ([README](../hybrid/README.md)) |
| --- | --- | --- |
| Shape | 48 layers, width 6,144, head dim 128 | 16 layers, width 1,536, head dim 128 |
| Sequence mixing | 3 sliding-window (2,048) : 1 global; GQA 4:1 local, 8:1 global | 3 KDA : 1 full attention; GQA 3:1 |
| Positions | 50% partial RoPE local, NoPE global | RoPE (base 1M) on full-attention layers |
| Attention extras | Gated attention, XSA, QK-norm ×1.3 | Sigmoid output gate, attention sink, QK-norm |
| Short convolutions | Keys, attention output, MLP output | Inside KDA layers only |
| FFN | Latent MoE, 8 of 384 + 2 shared | Dense SwiGLU |
| Loss | z-loss 1e-4 | z-loss 1e-4, MTP |
| Optimizer | Hyperball (MuonH/AdamH) plus Adam | Muon (0.02) on matrices, AdamW (3e-4) elsewhere, WD 0.1 |
| Schedule | 1% warmup, linear decay to 5% | Linear warmup over 0.5B tokens, constant, linear decay to 0 over the last 15% |
| Data and batch | 18T tokens, 46.1M tokens/batch | 40B tokens, 1M tokens/batch |

## What transfers to us, read through their reasoning

Items whose reasoning is **measured quality** transfer most directly; items driven
by GB200 MFU, expert parallelism or the hyperball optimizer need our own evidence.

1. **Short convolutions outside KDA (quality, strong).** The largest cheap gain
   (1.19× at d1024), and their reason, a shallow model wanting another cross-token
   path, applies more strongly to our 16 layers. Their reason for skipping the value
   convolution is XSA, which we do not use, so for us **include V** (1.03× alone at
   d1024). `kda_convolution.py` already has the document-masked depthwise
   convolution; identity init keeps an exported checkpoint's function unchanged at
   the start of continued training.
2. **Embedding RMSNorm (quality, but optimizer-specific).** The 19% gain comes from a
   setting where the embedding is in plain Adam and its norm grows 40× unchecked. Our
   tied table has weight decay 0.1 and also serves as the output matrix, so its norm
   is constrained differently. Normalize the lookup only, and expect a smaller effect.
3. **Linear decay to a floor (reasoning does not transfer).** The author attributes
   linear beating WSD to the hyperball fixing parameter norms; with our Muon/AdamW
   and weight decay, norms move and a WSD stable phase already decays the effective
   LR. Keep it as a test, not a presumption. The 5% floor was chosen to leave room for
   SFT/RL, which matches our pipeline (SFT and RL follow pretraining).
4. **Fixed QK multiplier (quality, small).** 0.005 at d768, and the stability
   argument (settle logit scale before extending context) applies to our 32K stage.
5. **Confirmations.** z-loss 1e-4 prevented a late loss rise there; we use 1e-4.
   Gated attention and QK-norm are in our model. GQA's 1.18× cost is the price of
   the KV cache in both.
6. **Asymmetric GQA by cache behavior (memory).** Their rule, compress KV heads where
   the cache grows with context, maps to us as: our full-attention layers carry all
   the growing cache (KDA state is fixed-size), so if long-context memory binds, that
   is where to raise the GQA ratio.
7. **MTP is not evidence against ours.** They skipped it for memory and MFU at 535B,
   not on quality; our MTP decision stands on our own measurements.
8. **NoPE on global layers (no reason, no ablation).** Relevant to the 32K stage of
   [HYBRID_V2.md](../plans/HYBRID_V2.md), but the post gives neither, so it is a
   hypothesis only.
9. **Process.** A preregistered forecast from a small ladder, compute-efficiency
   multipliers against a fitted control, "deviation needs a bigger payoff", and
   reading plots on the real hardware are all directly usable for our runs.

Not applicable at our size: MoE routing, quantile balancing, expert parallelism and
token dropping, batch-size reasoning driven by routing, and hyperball optimizers
(which would replace Muon/AdamW wholesale and need their own comparison).

## Proposed ablation (not run)

On the 720M hybrid pilot config, with the existing `trainer/loop.py` and hybrid
family, same data order, seed and token budget, compare against the current
control:

| Arm | Change |
| --- | --- |
| A | Control |
| B | Short convolutions (kernel 4, identity init, document-masked) on full-attention keys and values, attention output and MLP output |
| C | RMSNorm on the embedding lookup |
| D | Linear decay to 5% of peak after the existing warmup (`learning_rate_factor` in `hybrid/train.py`) |

Report held-out ClimbMix/code/math loss and, if the control is run at two budgets,
compute-efficiency multipliers. Arms B–D stay in the plan only if they beat the
control by more than seed-to-seed spread. Any launch follows the root Modal
runtime rules (timeout ≤ 86,400 s, no retries, independent deadline supervision).
