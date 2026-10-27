# 100M-class KDA mixer comparison

## Main experiment results

All twelve runs, including two precision controls, completed 100,663,296 input
token positions each. RoPE without convolution has the lowest observed mean
held-out loss, and its fused kernel runs at essentially no-convolution speed.
Its additional loss gain over no convolution is small and positive in both
seeds. The precision-matched baseline controls preserve this ranking.

[Archived metrics, checks, logs and learning curves](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/results/kda-100m-1793059841).

| Variant | Seed 17 CE | Seed 29 CE | Mean CE |
| --- | ---: | ---: | ---: |
| Original baseline | 3.773981 | 3.802101 | 3.788041 |
| BF16-input baseline control | 3.801982 | 3.801073 | 3.801527 |
| No convolution | 3.666530 | 3.669406 | 3.667968 |
| RoPE without convolution | 3.644643 | 3.665334 | 3.654988 |
| Rotating memory, Triton | 3.666716 | 3.658386 | 3.662551 |
| Middle-layer RoPE | 3.778278 | 3.778374 | 3.778326 |

RoPE reduces CE by 0.021887 and 0.004073 relative to no convolution in the
paired seeds, averaging 0.012980. Rotating memory's small mean improvement
over no convolution does not repeat in both seeds, and its measured runtime
is substantially greater. Middle-only RoPE also has opposite-sign changes
relative to the original baseline across seeds.

RoPE and no convolution are the two useful candidates for further testing.
The fused RoPE kernel described below brings RoPE's runtime to parity with
the no-convolution control in its separate same-GPU benchmark. The RoPE advantage is
larger during parts of training than at the final checkpoint. This budget is
approximately one training token per parameter; final convergence and
behavior at larger context lengths remain untested.

## Setup

Follow-up to [the short mixer pilot](STUDY.md), based on PR77. The four best
pilot variants compete with the original conv-4 KDA baseline: no convolution,
four-mode rotating memory, RoPE without convolution, and RoPE replacing the
convolution in the middle KDA layer of each group. Full-attention layers are
unchanged. Rotating memory uses the custom Triton training implementation.

The architecture has 12 layers, width 768, SwiGLU width 1792, 12 KDA
heads of width 64, and full attention with 12 query / 4 KV heads. Its repeating
layout is KDA/KDA/KDA/full. Vocabulary size is 32,768 with tied embeddings.
Measured parameter counts range from 104,314,704 (no convolution and RoPE)
to 104,397,864 (rotating memory); baseline has 104,397,648 parameters.

Each of two seeds (17, 29) receives 12,288 updates of eight 1,024-token
sequences: 100,663,296 input token positions. The shared initial parameters,
data order, optimizer and learning-rate schedule match across variants.
AdamW uses peak learning rate 0.0003, 400-step warmup, cosine decay to 10%,
betas (0.9, 0.95), weight decay 0.1 and gradient clipping at 1. Training uses
BF16 autocast, FP32 parameters and the existing z-loss. Held-out cross-entropy
excludes z-loss and BOS labels. Evaluation uses the same 262,144 held-out
token positions every 512 steps, plus initial and first-step measurements.

Data preparation selects approximately 101M training tokens and 524K held-out
tokens from the same frozen ClimbMix/code snapshot and tokenizer as the pilot,
balanced approximately equally across its six sources. Documents are shuffled
with seed 2026 before packing. Runs consume a single prefix, without recycling
the pilot's small corpus. Preparation requires each source to supply its quota
and checks that selected document hashes do not overlap between splits.
The snapshot's existing incomplete corpus-audit status remains a limitation.

Before training, the custom rotating-memory kernel must pass its original
FP32/BF16 output/gradient tests and a BF16 check at the actual 8 × 1024 × 768
input shape, including document resets. This is a numerical qualification;
long-run training stability and quality remain experimental outcomes.

Each training worker has a three-hour execution timeout and no retries.
An independent remote supervisor cancels all tracked preparation,
qualification and training calls at the overall 3.5-hour launch deadline,
including queued work. Metrics are saved at each evaluation; model/optimizer
checkpoints are saved after the first update and every 2,048 updates. A
deadline exit saves a partial checkpoint and is reported as incomplete.
The latest checkpoint for each run lives on the training Modal volume.

Run from `training/` using the Modal environment with system CA trust:

```sh
python launch_kda_long.py
```

Executable configuration is in `hybrid/long_study.py`. The launcher archives
structured metrics and provenance to the private HF dataset
`jfrick/lm-exploration-results`, under `training/results/<run-id>`.
Compare final loss across paired seeds, learning curves at matched token
budgets, throughput and memory. Two seeds and 100M tokens can identify a
promising follow-up; they do not establish the ranking of fully pretrained
models or behavior beyond the trained context length.

The actual-shape kernel check passed: maximum relative L2 output/gradient
error was 0.0000265 (about 0.003%). Original FP32/BF16 checks also passed.
The first launch, `kda-100m-1793059455`, prepared 101,016,001 training tokens
and 527,104 validation tokens and passed qualification. Its local collector
caught the wrong timeout type when polling Modal and cancelled pending
workers before training. The failure logs and qualification evidence are
preserved in the HF results archive.

Run `kda-100m-1793059841` uses the corrected collector and hash-verified copies
of those exact prepared arrays. It retains the first launch's overall deadline.
All twelve training runs completed. The collector saved paired summaries and
plots to the same HF results folder. Each run has the complete 25-point
evaluation curve and its final model/optimizer checkpoint on the Modal volume.

After collection, run `python -m hybrid.analyze_long_study artifacts/results/<run-id>`
from `training/` to produce structured comparisons and learning-curve plots.

The speed benchmark runs the selected variants on one H100 at the same batch and
context sizes, with ten warmup updates each followed by four rounds of twenty
updates in alternating forward/reverse variant order. All models stay resident;
timings cover forward/backward/clipping/AdamW and exclude data transfer,
evaluation and compilation. Launch it with
`python launch_kda_benchmark.py --run <run-id> --long` from `training/`.

## Controlled speed results

The H100 80GB HBM3 benchmark completed on torch 2.14.1+cu130. The six-variant
repeat includes the precision control. Mean optimizer step times across its
four warmed rounds were:

| Variant | Milliseconds per update | Throughput relative to baseline |
| --- | ---: | ---: |
| Original baseline | 126.82 | 1.00× |
| BF16-input baseline | 128.38 | 0.99× |
| No convolution | 99.47 | 1.27× |
| RoPE without convolution | 108.46 | 1.17× |
| Middle-layer RoPE | 119.86 | 1.06× |
| Rotating memory, Triton | 142.99 | 0.89× |

These compare the current implementations. In particular, the baseline uses
PR77's unfused short-convolution operations, so the speed difference is not a
lower bound on the cost of convolution. The rotating-memory implementation
uses custom forward and backward kernels. Raw round timings are in the
study's `benchmark-precision.json` artifact. The initial five-variant timing
run is preserved in `benchmark.json`; it had the same speed ordering.

## Precision control

A CUDA autocast inspection confirmed that PR77's FP32 convolution weights
promote its q/k/v activations to FP32, whereas the no-convolution and plain
RoPE variants present BF16 q/k/v to the recurrence. This means their comparisons
with the original baseline include an input-precision difference. RoPE versus
no convolution has matching precision.

Two additional `baseline_bf16` runs retain the same convolution, initial weights,
data and training schedule while casting convolution outputs back to the
projection dtype before the recurrence. They use the original overall study
deadline. Their launch is recorded in `precision-launch.json`; a new six-variant
same-GPU benchmark is stored in `benchmark-precision.json`, preserving the
initial five-variant benchmark. Both controls completed the full budget:
3.801982 and 3.801073 CE, averaging 3.801527. No convolution and RoPE without
convolution both beat this precision-matched baseline in each seed. Their
gains therefore persist when recurrence input precision is matched.

## Fused RoPE kernel

`kernels/rope.py` fuses the q/k rotations and implements their adjoint while
retaining the KDA `[B,T,H,D]` layout. It uses the existing document-position
RoPE tables. Separate FP32 product rounding preserves the reference arithmetic.
Use study variant `rope_triton` for this implementation; `rope` is its PyTorch
reference. The long-run loss table above comes from the PyTorch reference.

The kernel had zero measured output and input-gradient error against that
reference in FP32 and BF16 at lengths 1, 17 and 1024, including the full
8 × 1024 × 12 × 64 training shape. A matched 100-update, 104M-model training
check on actual study batches recorded zero CE difference at every update.
This qualification does not constitute another full 100M-token training run.

A separate warmed H100 benchmark compared these three implementations on the
same device, with four alternating-order rounds of twenty updates each:

| Implementation | Mean milliseconds per update |
| --- | ---: |
| No convolution | 83.93 |
| PyTorch RoPE without convolution | 92.43 |
| Fused RoPE without convolution | 84.15 |

The fused path improves throughput by 9.8% relative to PyTorch RoPE in this
benchmark and is within timing variability of no convolution. Its absolute
times should be compared within this three-model run. The six-model timing
comparison above is a separate benchmark. Raw checks, the 100-update trace
and round timings are in `rope-kernel.json` in the study archive.
