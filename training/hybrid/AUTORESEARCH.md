# Bounded KDA architecture research

The user authorized implementing and testing the research directions for a
1–2B hybrid with long context. This campaign retains the original learned q/k/v
convolutions, uses a larger proxy than the embedding-heavy 98M experiments, and
separates numerical correctness, quality, and systems outcomes.

## Registered arms

| Arm | Change relative to baseline | Question |
| --- | --- | --- |
| baseline | Original KDA and 3:1 layout | Matched control |
| fused | Custom segmented convolution/SiLU forward and backward | Can implementation improve speed without material quality change? |
| head_timescales | Initialize heads across 8–8192 token e-folding scales | Can existing state support persistent and fast memories? |
| slow_bank | Two independently addressed banks per original head, fast/slow initialization | Does extra persistent memory earn its additional state? |
| banks2 | Two half-key-width banks sharing each value projection | Does independent addressing help at equal recurrent state bytes? |
| key_wide | Double key width, unchanged value width | Does greater addressing capacity help? |
| heads_narrow | Double heads, halve key and value widths | What happens at fixed projected width and half the recurrent state? |
| kv8 | Zero-extend key/value filters to eight taps | Does longer local feature construction help? |
| kv16 | Zero-extend key/value filters to sixteen taps | Same question at greater support |
| dynamic_kv | Input-dependent signed residual eight-tap filters on k/v | Does adaptive local filtering help? |
| input_gate | Residual two-layer nonlinear input-conditioned beta predictor | Does a richer write policy help? |
| memory_gate | Beta also sees prediction-error and memory-read energy | Does memory-aware writing justify losing affine chunk parallelism? |
| layout_dense | Two KDA layers per full-attention layer | Does more exact retrieval help? |
| layout_sparse | Five KDA layers per full-attention layer | Can less global attention preserve quality? |
| layout_front | Move full attention earlier in each four-layer group | Does placement matter at equal layer counts? |

All candidate definitions, parameter/state counts, and qualification outcomes are
saved. Equal recurrent state is not equal total parameters or compute. Banks
share projected values but have separate keys, decay, and write gates; outputs
are averaged before the original output projection. The memory-aware gate has
an explicit differentiable sequential implementation; it is not falsely passed
through the affine FLA kernel. More than 4x baseline layer time rejects an arm
from the expensive training screen, while preserving its correctness result.

## Protocol and boundaries

The proxy has 12 layers, width 1024, eight 128-dimensional KDA heads, and the
existing 154,880-entry GLM vocabulary. Numerical parameter counts are generated
from the actual modules, including a 24-layer, width-2048 target configuration.
The target configuration is for accounting; target-scale training is a separate
follow-up after proxy confirmation, not an outcome claimed by this campaign.

All qualified arms train with seeds 17 and 29 to approximately 500M input token
positions each at 4K context. Data, order, batch token count (8192), AdamW,
clipping, and the two-billion-token learning-rate schedule are shared. Screening
checkpoints are intermediate points on that schedule, not separately converged
models. Baseline and at most two candidates continue from exact saved optimizer,
RNG, and data-cursor states toward 2B tokens. Both seeds must complete screening;
selection ranks mean CE gaps, retaining candidates within +0.01 nats of baseline
to avoid overconfident early elimination. This band is not a significance test.
No candidate is called a winner from the screening ranking alone.

The campaign has a 23-hour absolute deadline including queue time, no retries,
eight training workers, four qualification workers, and two evaluation workers
at most. Training workers have 20-hour execution limits and checkpoint before
the absolute deadline; the independently running remote supervisor explicitly
cancels all registered child calls. A stage may finish with partial evidence if
the budget runs out. Jobs are not automatically restarted after timeout.

## Checks and evaluations

GPU qualification compares KDA outputs and gradients against the sequential
reference, verifies causality and document isolation, and checks that zero-added
branches preserve the baseline's initial function. The fused convolution also
has separate FP32/BF16 forward and backward comparisons. Same-device layer
benchmarks exclude compilation, but do not substitute for full-model throughput.

Screening uses a fixed held-out prefix, with per-source and paired-window CE.
Long-document evaluation selects a different held-out region, scores the same
last 256 target tokens with increasing context, and records document coverage.
It also runs single-key, many-key, latest-update, and order-sensitive synthetic
prompts. The small prompt set is diagnostic; teacher-forced exact answer scores
are explicitly not free-generation accuracy or a powered retrieval benchmark.

Screening checkpoints are evaluated through 32K, and completed confirmation
checkpoints through 128K if the deadline permits. These are extrapolation tests
of 4K-trained models. They do not establish successful 32K training, context
curriculum benefits, or production cached-decoding latency. Target-scale
training, extended-context adaptation, and cached-decoding deployment tests must
follow on the shortlist before choosing a 1–2B production architecture.

## Execution and evidence

From `training/`, run `python launch_kda_autoresearch.py`. The launcher requires
the pinned public GLM tokenizer in ignored
`artifacts/autoresearch/glm-tokenizer.json`. Run
`python collect_kda_autoresearch.py RUN_ID` to archive metrics and decisions to
the private `jfrick/lm-exploration-results` dataset, under `training/results/RUN_ID`.
Weights and optimizer states remain in the persistent Modal training volume.
The source branch and hashes are stored with each launch. Failed qualification,
failed training, timeout, and systems rejection are distinct outcomes.
