# Cached RoPE and compiled microbatch 4 — single-H100 results

Both variants completed 99,988,281 nonpadding tokens in 1,296 optimizer updates,
using the same 124.4M model, 8K context, global batch 32, seed, data snapshot and
100M-token schedule as the previous suite. Both include cropping and cached RoPE.

| Configuration | Steady nonpadding tokens/s | Peak allocated GiB | Peak reserved GiB | Whole-continuation tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Microbatch 4, eager | 63,397 | 31.81 | 39.37 | 61,982 |
| Microbatch 4, compiled | 83,671 | 26.02 | 32.90 | 78,746 |
| Prior microbatch 2, compiled | 74,017 | 14.36 | 17.64 | 65,851 |

Compiled microbatch 4 was 32.0% faster than its eager control and 13.0% faster
than the prior compiled microbatch 2 run, using about 81% more peak allocated
memory than compiled microbatch 2. These are single runs, not repeated estimates
with confidence intervals. Steady throughput excludes the first ten continuation
steps. Whole-continuation throughput includes startup, evaluation and checkpoint
costs but excludes the separate two-step initialization and correctness gate.

All twelve CUDA FP32/BF16 loss/gradient cases passed, including compiled
microbatch 4. Its maximum absolute gradient error was 3.35e-8 (FP32) and 5.64e-4
(BF16). Both runs matched saved diagnostic validation exactly after restoring
the two-step checkpoint in a fresh process. Final diagnostic losses were
3.54187 (eager) and 3.54764 (compiled); the small diagnostic is not a quality
parity claim. Same total tokens and optimizer steps were consumed.

Compiled microbatch 4 is now the fastest measured single-GPU configuration.
Eight-GPU scaling, communication and rank imbalance remain unverified. No new
training was launched during result collection, and the active 5B run was not
modified. Raw results and qualification logs accompany this report.
