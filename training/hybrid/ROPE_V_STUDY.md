# RoPE q/k with learned value convolution

This follow-up tests whether retaining local value construction improves the
convolution-free RoPE candidate. Every KDA layer applies SiLU and the qualified
fused RoPE kernel to projected q/k, and a learned four-tap causal depthwise
convolution followed by SiLU to projected v. There are no q/k convolutions.
The value convolution is evaluated with FP32 coefficients and its output is cast
back to BF16 before the recurrence, matching the recurrence input precision of
the completed RoPE control. Document resets apply to both convolution and KDA.

Two runs, seeds 17 and 29, use the same shared initialization, prepared GLM data,
70/20/10 mixture, batch size, 1,024-token context, AdamW settings and 1B-token
budget as [the completed study](GLM_STUDY.md). Each model has 98,263,572 parameters.
The existing baseline, no-convolution and RoPE controls are reused for comparison.
Training starts from scratch for a matched architectural comparison.

`python launch_kda_rope_v.py` from `training/` launches a GPU qualification,
full-budget data preflight, and a detached remote coordinator. Qualification
checks common initialization, output and parameter/input-gradient agreement
with reference RoPE, value-filter gradients, and isolation across document
boundaries. Training begins only after qualification succeeds. The two workers
use nine-hour server-side timeouts with retries disabled, and an independent
supervisor enforces ten hours from launch including queue time. Checkpoints and
metrics follow the existing study protocol.

The launched run is `kda-rope-v-1b-1793165441`, with artifacts collected into the
private HF dataset `jfrick/lm-exploration-results` under its corresponding
`training/results/` folder. Startup qualification observed zero output/gradient
differences in its test case and passed document isolation. Training quality
results are pending.
