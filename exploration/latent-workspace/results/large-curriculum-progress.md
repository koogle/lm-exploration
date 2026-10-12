# Prose/code stage and translation handoff

Exact reports and curves: [archive](large-curriculum-prose-code.json),
[chart](large-curriculum-prose-code.png). Both fresh 79M drafts use frozen
Qwen2.5-0.5B and 131072 prose/code examples, with common initialization/validation.

| Response cap | Updates | Validation prose NEW | Code NEW | Translation NEW | Prose/code validation NLL | Training diagnostic NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 17985 | 2.750 | 2.648 | 0.121 | 3.423 | 2.503 |
| 256 | 11472 | 4.939 | 4.492 | 0.103 | 1.414 | 1.868 |

Prose/code validation NLL continues to decrease overall. Short-cap acceptance
improves slowly with some fluctuations; long-cap acceptance also increases.
There is no sustained held-out-loss increase indicating overfitting at these
snapshots. This does not establish absence of overfitting elsewhere: the training
diagnostic is fixed and tiny (eight responses), and no full-training audit exists.
Repetitive teacher trajectories can be easier to match; no human quality claim.

All-domain validation NLL is 5.874/4.200; overall NEW 1.840/3.178. Translation has
not been trained in this stage, so its NLL around 10 must not be interpreted as
an ordinary matched-domain generalization gap. Continue with translation rather
than early-stop these fits based on the all-domain aggregate alone. The two
checkpoints have different update counts; this table is a status snapshot, not
an equal-exposure response-length effect estimate.

Full [teacher pool](long-teacher-216k-07.json) completed with 71196 prose, 71563
code, 71238 translation. Both worker-02 handoffs failed before training because
reused Modal mounts lacked the newly committed teacher files. Reload the runs
volume at worker input start; a focused regression verifies load order. Coordinator
08 launched worker-03 retries from worker-01 checkpoints with unchanged training
source, full 196608 data, and own optimizer/RNG restoration. New full-domain
checkpoints are pending. Preserve failed reservations; current epoch total $91.125.
Test remains sealed and quality qualification incomplete.
