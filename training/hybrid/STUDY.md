# PR77 input-mixer pilot

Base: PR #77, commit `1bda9094147a5ee7469f4d0f3f2db095904db44e`.

The pilot compares baseline short convolutions; SiLU projections with no
convolutions; no convolutions plus RoPE on every KDA layer; RoPE replacing the
convolution only on the middle KDA of each three-layer group; and four damped
rotating memories replacing each q/k/v convolution. Full-attention layers retain
the PR's RoPE, sink and gate. Shared weights start identically within each seed.

Each model has eight layers, width 256, four KDA heads of width 64, four query
heads and two KV heads in full attention, and MLP width 768. Runs use context
512, batch eight, 800 updates, seeds 17 and 29, AdamW with peak LR 0.0003,
50-update warmup and cosine decay to 10%, weight decay 0.1 and gradient clipping
at 1. Validation reports cross-entropy excluding the auxiliary z-loss. Evaluation
uses the same sixteen batches at each 100-update interval. Throughput excludes
the first ten training steps and evaluation, but uses eager surrounding PyTorch
operations and the PR's compiled FlexAttention/FLA kernels; it is not a comparison
of fully optimized implementations.

The rotating mixer uses four learned decay timescales initialized at 4, 16, 64,
256 and frequencies 0.5, 0.125, 0.03125, 0.0078125 radians/token. Each mode rotates
adjacent channel pairs, with learned per-channel mixing and a direct residual.
An exact doubling scan evaluates the recurrence, respecting document boundaries.
The checks compare outputs and gradients against a sequential recurrence and
verify whole-model causality, document isolation and parameter gradients.

Data comes from the existing ClimbMix/code v6 prepared snapshot, using the frozen
ClimbMix/code tokenizer. The pilot selects approximately equal token counts per
source, shuffles documents deterministically, then packs them with BOS/EOS. It
uses distinct prepared train/validation splits and checks document hashes do not
overlap. This is an architecture pilot on a bounded subset, not the production
mixture. The source snapshot reports `training_ready=false`; its incomplete corpus
audits limit claims beyond this controlled comparison.

Run from `training/` with Modal and truststore installed:

```sh
python launch_kda_study.py
```

The launcher preserves provenance locally under ignored `artifacts/results/`.
An independent remote supervisor starts before work, tracks call IDs in a Modal
Dict, and cancels work at a three-hour launch deadline. Preparation has a 20-minute
server timeout and GPU workers 90 minutes, all with retries disabled. Partial
metrics and prepared arrays are saved on the existing training volume. Results
must be archived to the repository's HF results dataset before final reporting.

## Measured outcome, 2026-10-26

Twelve runs completed: six variants × two seeds, each trained for 800 updates
(3,276,800 input token positions). Models have approximately 15.6M parameters.

| Variant | Seed 17 CE | Seed 29 CE | Mean CE | Difference from baseline |
| --- | ---: | ---: | ---: | ---: |
| Original short convolution | 7.192697 | 7.179218 | 7.185957 | 0 |
| No convolution, SiLU retained | 7.088135 | 7.095680 | 7.091908 | -0.094050 |
| No convolution + RoPE on all KDA | 7.108556 | 7.111240 | 7.109898 | -0.076059 |
| Rotating memory replacing convolution | 7.097314 | 7.092667 | 7.094990 | -0.090967 |
| RoPE replacing middle KDA convolution | 7.146749 | 7.128441 | 7.137595 | -0.048362 |
| RoPE added to middle KDA, convolution retained | 7.218373 | 7.187602 | 7.202988 | +0.017031 |

Lower is better. The held-out cross-entropy excludes z-loss and uses 65,536
fixed validation token positions, masking BOS targets. The selected training and
validation document hashes do not overlap. The data preparation produced 5,005,556
training tokens and 269,227 validation tokens; the runs consume bounded prefixes.

Removing convolution beats the original in both seeds. RoPE without convolution
beats baseline, but loses to the no-convolution control in both seeds, so the
improvement cannot be credited to RoPE. Adding RoPE to middle layers while
retaining convolution loses to baseline in both seeds. The rotating mixer is
within 0.0031 mean nats of the no-convolution control, with the paired difference
changing sign across seeds: there is no convincing incremental loss benefit.

These results support a longer no-convolution ablation. They do not establish
that convolutions are unnecessary in the 720M model, after substantial training,
or for long-context retrieval. Two seeds and 3.28M training token positions are
insufficient for such extrapolation. RoPE base, filter initialization and learning
rate were held fixed, not separately optimized per variant.

The additional convolution-retaining middle-layer control reuses the identical
prepared arrays:

```sh
python launch_kda_extra.py --run <run-id>
```

A separate warmed benchmark interleaves all six variants in forward and reverse
orders on one H100, with four rounds of twenty complete optimizer steps per
variant, following ten warmup steps each. Its timing excludes data transfers,
evaluation and initial compilation. It measures these eager surrounding
implementations, not hypothetical fused-kernel performance:

```sh
python launch_kda_benchmark.py --run <run-id>
python hybrid/plot_study.py artifacts/results/<run-id>
```

Raw metrics, data provenance, launch records and the loss plot are archived in
[the private results dataset](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/training/results/kda-mixers-20261026).

## Custom rotating-memory training kernels

`kernels/rotating.py` supplies Triton forward and backward kernels with a
PyTorch autograd wrapper. Each program handles one batch row and channel pair,
scanning four damped complex modes. The reverse scan computes state adjoints;
analytic derivatives cover the input, log-timescales, rotation angles and mixing
weights. Both scans reset at document boundaries. The backward preserves the
reference's BF16 rounding boundary between memory and residual gradients.
The prototype targets even widths and sequences up to 1024 tokens; these studies
use 512. It is a training kernel, not a streaming-decoding implementation.

Qualification covers FP32 and BF16, sequence lengths 1, 17, 128, 512 and 1024,
and packed document resets. All recorded relative-L2 errors pass 3e-5 for FP32
and 2e-4 for BF16 (or absolute error below 2e-5 for negligible gradients).
FP32 output/input errors are approximately 2e-7 relative; the largest BF16 input
gradient error after the rounding fix is 3.34e-5 relative.

A 100-update matched run alternated the reference and custom implementations on
one NVIDIA H100 80GB HBM3. Mean complete optimizer-step time after the first ten
updates was **84.583 ms reference versus 51.879 ms custom**, a **1.6304× end-to-end
speedup**. This includes all model layers and optimizer work, not just the mixer.
The maximum per-batch cross-entropy difference was **0.027850 nats/token**,
within the preset 0.03 training-agreement threshold. The implementations are not
bitwise equivalent; the full 800-update accuracy table above uses the PyTorch
scan, and is not a separate 800-update validation of the custom kernels.

The initial one-token test needed to represent an unused angle gradient as zero.
An initial fused backward rounded the sum of memory and residual gradients once;
its 100-step drift reached 0.032871 nats and failed the threshold. Matching the
reference's separate BF16 branch rounding reduced numerical gradient error and
passed the repeated check. Failed qualification logs are retained in the archive.

```sh
python launch_kda_kernels.py --run <run-id>
```

## Same-GPU reference benchmark

The completed interleaved benchmark on an H100 80GB HBM3 measured mean optimizer
step times of 75.197 ms (baseline), 61.040 ms (no convolution), 64.436 ms
(no convolution + RoPE), 71.172 ms (middle replacement), 80.628 ms (middle RoPE
with convolution), and 108.831 ms (PyTorch rotating mixer). Removing convolution
is **1.232× faster than baseline** in this controlled benchmark.

The custom-kernel timing above is a separate matched experiment with alternating
changing training batches and two models, rather than this fixed-batch six-model
benchmark. Compare ratios within each experiment; their absolute times are not
interchangeable. An interrupted preliminary benchmark and a TorchVersion result
serialization issue were retained in the provenance; the successful benchmark's
result was recovered without rerunning its measured computation.
