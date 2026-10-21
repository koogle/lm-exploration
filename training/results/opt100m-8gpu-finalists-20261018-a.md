# Eight-H100 finalist results — 2026-10-18

All three variants completed 99,988,281 nonpadding tokens / 1,296 updates with
identical data exposures, 124.4M parameters, 8K context and global batch 32.
All include cropping and cached RoPE. One run per configuration, PyTorch 2.6.0.

| Configuration | Steady tokens/s | Whole-continuation tokens/s | Peak allocated per GPU (GiB range) | Eight/one GPU throughput ratio |
| --- | ---: | ---: | --- | ---: |
| rope-m4 | 213,354 | 189,226 | 31.28–31.31 | 3.37× |
| compile-m4 | 258,793 | 188,036 | 25.49–25.51 | 3.09× |
| compile-m2 | 275,631 | 198,217 | 14.33–14.36 | 3.72× |

Compiled microbatch 2 wins on eight GPUs: 6.5% higher steady throughput than
compiled microbatch 4 and 29.2% above eager microbatch 4. It uses 43.7% less peak
allocated memory than compiled microbatch 4. The single-GPU ranking reverses:
compiled microbatch 4 was faster there. Eight GPUs yield only 3.09–3.72× the
corresponding single-GPU throughput at fixed global batch, not linear scaling.

Compilation overhead matters for these short runs. Whole-continuation throughput
is 198,217 tokens/s for compiled microbatch 2 versus 189,226 for eager microbatch 4
(4.8% gain); compiled microbatch 4 is 188,036. This includes continuation startup,
evaluation and final checkpoint, but excludes the separate two-step initialization
and qualification gate. Steady measurements exclude the first ten continuation
updates. These are not repeated trials or confidence intervals.

## Verification and rank balance

All twelve CUDA FP32/BF16 loss/gradient cases passed on eight ranks, including a
fully masked rank and compiled microbatches 2/4. The retained qualification log
reports rank-zero maximum absolute errors of 4.47e-8 (FP32) and 5.95e-4 (BF16);
all ranks asserted tolerances before successful completion. Every variant's
fresh-process checkpoint restore matched its saved diagnostic validation exactly.
This does not establish bitwise trajectory equivalence or production-checkpoint
migration compatibility. Final small diagnostic losses were 3.53792, 3.54129 and
3.56100 respectively, not evidence of quality parity.

Supervised tokens per rank over measured steps ranged from 12,115,947 to
12,677,045, a max/min ratio of 1.0463 in all runs. Aggregate synchronized compute
time max/min ratios were 1.00345 (eager m4), 1.00473 (compiled m4), and 1.00538
(compiled m2). Mean per-step max/min ratios were 1.00484, 1.00727 and 1.00904.
These timings include DDP communication and waiting, so similar rank times do not
prove equal intrinsic compute work or low communication overhead. Per-rank
allocated and reserved memory are retained in the rank-summary JSON; maximum
reserved memory was 51.38, 44.79 and 21.73 GiB respectively.

Recommend compiled microbatch 2 with cropping and cached RoPE for future
eight-H100 runs, subject to workload and longer-run validation. No additional
training was launched during result collection and production runs were unchanged.
Raw results, qualification log and per-rank summaries accompany this report.
