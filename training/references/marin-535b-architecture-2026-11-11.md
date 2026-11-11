# Marin 535B-A23B architecture decisions: lessons for our hybrid model

Reviewed 2026-11-11 against Larry Dial,
[Architecture Decisions for Marin 535B-A23B](https://openathena.ai/blog/marin-535b-architecture/)
(Open Athena blog, 2026-11-09). All numbers below are author-reported; none are
our measurements. The post's code, ablations and fits are linked from
[marin-community/marin#8435](https://github.com/marin-community/marin/issues/8435).

Most ablations are Paloma macro loss at hidden width 768 or 1024 (d768, d1024).
The post converts loss differences into **compute-efficiency multipliers**: how
much more compute the baseline needs to reach the same loss, read off a fitted
loss–compute curve. A multiplier is relative to Marin's baseline and optimizer,
not a transferable speedup.

## The model as described

| Component | Marin 535B-A23B | Our hybrid 720M ([README](../hybrid/README.md)) |
| --- | --- | --- |
| Shape | 48 layers, width 6,144, head dim 128 | 16 layers, width 1,536, head dim 128 |
| Sequence mixing | 3 sliding-window (2,048) : 1 global; GQA 4:1 local, 8:1 global | 3 KDA : 1 full attention; GQA 3:1 |
| Positions | 50% partial RoPE on local layers, NoPE on global layers | RoPE (base 1M) on full-attention layers |
| Attention extras | Gated attention, exclusive self attention (XSA), QK RMSNorm with fixed ×1.3 multiplier | Sigmoid output gate, attention sink, QK RMSNorm |
| Short convolutions | Kernel 4 on keys, attention output and MLP output; identity init; reset at document boundaries | Kernel 4 inside KDA layers only |
| FFN | Latent MoE (2× compression), 8 of 384 routed + 2 shared, SwiGLU; sigmoid router, weights renormalized to sum 2.5 | Dense SwiGLU |
| Load balancing | Top-k quantile balancing on raw router logits (no auxiliary loss listed) | — |
| Norms | RMSNorm on token embedding; learnable gain before `lm_head`; gated norm (rank-128 gate) | Pre-norm RMSNorm blocks, final norm |
| Loss | Logit z-loss 1e-4 | Logit z-loss 1e-4, MTP |
| Optimizer | Hyperball: MuonH on matrices, AdamH on `lm_head`, Adam on embedding, router, gates, norms, convolutions | Muon (LR 0.02) on matrices, AdamW (3e-4) elsewhere, WD 0.1 |
| Schedule | 1% warmup, linear decay to 5% of peak | Linear warmup over 0.5B tokens, constant, linear decay to 0 over the last 15% |
| Data and scale | 18T tokens trained, 46.1M tokens/batch at 4K; context 4K→16K→65K | 40B tokens, 1M tokens/batch at 4K; 32K mid-training |

Not stated in the post: vocabulary/tokenizer, peak learning rate, weight decay,
training precision (FP8/NVFP4 were deferred) and exact KV-head counts (12 local and
6 global follow from 48 query heads and the stated GQA ratios).

## Reported measurements worth keeping

**Scaling ladder.** Four rungs from 5e19 to 9e21 FLOPs, all at 791 tokens per
active parameter, fitted as `L = 1.5 + A·C^-α`, forecast 2.039 Paloma macro for the
final run; the dropless final loss was 1.982. One rung (27.7B-A1.2B) died at 81% from
a hardware failure and was not resumed. The isoflop study puts compute-optimal near
60 tokens per active parameter; 10× overtraining costs about 2× compute. The post
gives the final run as 2.7e24 FLOPs in the text and 2.6e24 in the Figure 12 table.

**Ablations (Paloma macro; lower is better).**

| Change | Result | Scale |
| --- | --- | --- |
| Short convolutions on K, V, attention output, MLP output vs none | 3.0378 vs 3.0610 (1.19×) | d1024 |
| Final convolution set (no V conv) vs none | 3.0208 vs 3.0489; adding V conv: 3.0236 | d768 |
| Removing embedding RMSNorm | about 19% compute-efficiency loss | d768 |
| QK-norm without the ×1.3 multiplier / without QK-norm | 3.0261 / 3.0281 vs 3.0208 | d768 |
| Learnable per-head QK gain vs fixed 1.3 | 3.0217 vs 3.0208; 2.6484 vs 2.6477 at d2048 | d768, d2048 |
| Sliding window 512 vs 2,048 | 3.0334 vs 3.0208 (2,048 is 1.10×) | d768 |
| Full MHA vs GQA | 2.9985 vs 3.0208 (1.18×); GQA kept for KV cache | d768 |
| Experts 4/192, 8/384, 16/768 | 3.0289, 3.0091, 3.0009 | d768 |
| XSA | 3.0208 vs 3.0221 without; halved early token-drop rate under EP (5.6% vs 12.7%) | d768 |
| Decay `1 - sqrt(t/T)` vs linear | 3.2642 vs 3.2670, only at 1.4× peak LR; not adopted | d768 |
| Gated-norm optimizer: Adam / AdamH / Muon / MuonH | 3.1392 / 3.1390 / 3.1366 / 3.1352 | d1024 |

The post also reports that linear decay beat WSD at d768, without printing the
numbers in the text (see issue #7717).

**Stability.** Without z-loss the d1024 ablation's training loss rose after about
step 4,000 to 3.39 at step 11,000 (reference about 2.1), and the `lm_head` input gain
grew to 557 instead of 183. Putting the embedding inside the hyperball caused
gradient spikes (about 60× at 1.2× LR, 700× at 1.4×), so it stays outside. A
sharded-numerics bug collapsed the `lm_head` norm on GPU (#8073). A cross-entropy
kernel precision fix (PR #8368) "may also have helped run stability".

**Learning-rate fit.**
`adam_lr = 0.087571 · tokens^-0.3461 · hidden^-0.3448 · sqrt(tokens_per_batch)`,
`muonh_lr = 13/3 · adam_lr` (#7856). The constants belong to hyperball optimizers
and do not transfer to our Muon/AdamW. The shape does: the fitted rate falls with
both run length and width.

**Negative and deferred results.** Latent MoE lost (0.74–0.75 compute efficiency)
until a learnable RMSNorm on the latent was added (then 1.25×). Learned output
scalars and a learnable QK scale gave nothing; a special key-convolution init
learned faster early but washed out by about step 1,500. Leading dense layers were
dropped; routed experts in the first two layers then get abnormally small gradients
(open question). Multi-token prediction was skipped for memory and MFU, not on a
measured quality comparison. MFU was 20%, later 30%.

## What transfers to us

1. **Short convolutions outside KDA.** Our KDA layers already have causal kernel-4
   convolutions with document resets; full-attention keys, attention output and MLP
   output do not. This was Marin's largest single cheap gain (1.19× at d1024), and
   `kda_convolution.py` already provides the masked depthwise convolution. Identity
   initialization keeps an exported checkpoint's function unchanged at the start
   of continued training.
2. **RMSNorm on the token embedding.** About 19% at d768 in their setup. Our tied
   table is read by both input and output, so normalize the embedding *lookup*
   only, not the table.
3. **Linear decay to a non-zero floor.** We hold the rate constant and decay to 0 over the
   last 15%. Marin found linear-to-5% better than WSD at d768. This is a
   schedule-only change in `hybrid/train.py` (`learning_rate_factor`), testable without model changes.
4. **Fixed QK multiplier after QK-norm.** Small (0.005 at d768) but free; a
   learnable gain was not better than the fixed 1.3.
5. **Confirmations, not changes.** z-loss 1e-4 prevented a late loss rise there;
   we already use 1e-4. Gated attention and QK-norm are already in our model.
   GQA costs them 1.18× against MHA, which is the price we also pay for the cache.
6. **NoPE on global layers for long context.** Their global layers carry no
   position encoding and long context is reached by raising the QK multiplier
   (1.3→1.66 for 4K→65K). Our full-attention layers use RoPE at base 1M. This is
   relevant to the 32K stage of [HYBRID_V2.md](../plans/HYBRID_V2.md) but has no
   ablation in the post, so it is a hypothesis only.
7. **Reporting as compute multipliers.** Fitting the control's loss–compute curve
   and quoting each arm as an efficiency multiplier makes small-scale ablations
   comparable across runs. Our hybrid ablations report raw loss differences.

Not applicable at our size: MoE routing, quantile balancing, expert parallelism and
token dropping, and hyperball optimizers (which would replace Muon/AdamW wholesale
and need their own comparison).

## Proposed ablation (not run)

On the 720M hybrid pilot config, with the existing `trainer/loop.py` and hybrid
family, same data order, seed and token budget, compare against the current
control:

| Arm | Change |
| --- | --- |
| A | Control |
| B | Short convolutions (kernel 4, identity init, document-masked) on full-attention keys, attention output and MLP output |
| C | RMSNorm on the embedding lookup |
| D | Linear decay to 5% of peak after the existing warmup |

Report held-out ClimbMix/code/math loss and, if the control is run at two budgets,
compute-efficiency multipliers. Arms B–D stay in the plan only if they beat the
control by more than seed-to-seed spread. Any launch follows the root Modal
runtime rules (timeout ≤ 86,400 s, no retries, independent deadline supervision).
