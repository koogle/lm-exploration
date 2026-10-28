# Local value convolution plus learned timescales

This follow-up preserves RoPE on q/k and the learned four-tap value convolution,
then adds four learned exponential memories to the value preactivation. It tests
whether additional timescales improve compression beyond local feature extraction.
It follows [the research review](LONG_CONV_RESEARCH.md).

For projected values `u`, each mode follows
`h[r,t] = a[r] h[r,t-1] + (1-a[r]) u[t]`, with
`a = exp(-exp(-log_tau))`. Initial e-folding times are 8, 32, 128, and 512 tokens.
Each layer learns four timescales and independent signed mixing coefficients for
every value channel. The output is
`SiLU(conv4(u) + sum_r mix[r] * h[r])`, cast to BF16 before KDA.
Both branches reset at document boundaries. The long branch includes the current
token; this is a residual bank of EMAs, not a lag-four-only FIR tail.

Mixing coefficients start at zero, preserving the local-only control's initial
function and shared parameters. Mixture gradients open the branch first;
timescale gradients are zero initially and become available as mixing learns.
There are 16,164 additional parameters across nine KDA layers. The KDA state
dimensions and decay/delta recurrence are unchanged. Exact streaming would need
four extra value vectors per layer; this experiment implements full-sequence
training, not an incremental decoding API.

The custom Triton segmented scan reuses the rotating-memory implementation with
fixed zero angles and its direct residual disabled. FP32 scan accumulations feed
the same local SiLU. GPU qualification compares outputs and all trainable input
and parameter gradients against a sequential PyTorch reference, at lengths 17
and 1024 in FP32 and BF16, with nonzero mixtures. It also checks causality,
document isolation, shared initialization, zero-branch function equality, and
nonzero mixture gradients through a full KDA layer.

Two seeds (17, 29), 122,070 updates / 999,997,440 positions each, use the existing
98M hybrid configuration, GLM tokenizer, prepared 70/20/10 data mixture, AdamW
schedule, and fixed validation windows. Controls are the local-only RoPE/value
convolution run `kda-rope-v-1b-1793165441` and the original baseline/no-conv/RoPE
run `kda-glm-1b-1793107241`. These are matched prior runs, not new concurrent
control replicas. Compare validation at equal token counts and report throughput
separately; worker timing is not a controlled kernel benchmark. Context is 1024,
so this cannot establish extrapolation beyond 1024 tokens.

Launch from `training/` with `python launch_kda_value_timescales.py`. The remote
controller starts training only after qualification and data preflight pass.
Workers have nine-hour execution limits, no retries, and an independent remote
supervisor enforces ten hours from launch including queue time. Metrics and
checkpoints persist on the training volume; the collector archives outputs to
the private `jfrick/lm-exploration-results` dataset.

Quality results are pending. A zero-initialized branch and additional memory
capacity do not guarantee better loss or throughput.
