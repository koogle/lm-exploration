# KDA continuation to one billion tokens

Continue the original KDA baseline and convolution-free RoPE from both seeds of the
[100M-token study](LONG_STUDY.md). RoPE uses the qualified fused Triton kernel.
Each model retains weights and AdamW state. Architecture: approximately 104M
parameters, 1,024-token contexts, batches of eight sequences.

Target: 122,070 total updates, or 999,997,440 input token positions per run.
The first 12,288 updates are inherited. Fresh documents come from the same six-source
mixture, excluding every document selected for earlier training and validation.
Documents are shuffled with seed 2027. The existing validation array stays fixed.
The snapshot remains experimental with incomplete audits, as in the earlier study.

All four runs share a 400-update linear rewarm from approximately 3e-5 to 3e-4,
then cosine decay to 3e-5 over the remaining updates. This follows a completed short
cosine schedule; it is not a one-billion-token run from scratch. Validation and
checkpoints are saved every 2,048 updates and at completion. Resumed validation CE
must match the original checkpoint within 0.002 nats before the first update.
Original checkpoints remain intact.

Run from `training/` with `python launch_kda_continuation.py`. CPU preparation has
a two-hour timeout; each H100 worker has a nine-hour timeout, retries disabled.
An independent remote supervisor enforces ten hours from launch, including queue
time, cancelling tracked calls. Workers save partial results before the deadline.
Metrics live on the training volume and are collected into the private HF dataset
`jfrick/lm-exploration-results` under `training/results/kda-1b-*`.
