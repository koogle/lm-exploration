# Qwen3.5 buffered distillation targets

This follow-up to PR #36 is gated on correctness and aggregate throughput versus
batched vLLM for the same pinned Qwen3.5-9B teacher. It does not assume the prior
batch-one speedup over Transformers predicts rollout throughput.

The CUDA entry point can read a token ID directly from device memory. Argmax or
sampling therefore feeds the next decode without a host scalar synchronization.
`generate_buffered` stores full logits or normalized BF16 hidden states and
transfers them after a fixed block. The latter reprojects hidden states with one
batched vocabulary projection; generation still needs a head projection at every
step to choose its next token. Fixed-block EOS tails are discarded.

`targets.trajectory` uses blocks of 64 tokens and GPU sampling, then retains raw
teacher top-64 log probabilities, selected-token log probability, full-vocabulary
logsumexp and residual mass. EOS targets are included. Top-64 probabilities are
not renormalized; treating them as a complete distribution would be incorrect.
The vLLM path requests raw log probabilities with the same top-64 target budget.

The first H100 qualification passed: buffered logits and deferred projections
matched the sequential custom kernel exactly on six prompt/length combinations
(32 and 128 generated tokens). Worst reference-logit NRMSE was 2.47%; lowest
reference top-1 agreement was 99.22%; worst mean reference-to-kernel KL was about
0.00060 nats. Matched median timings gave only 1.015x speedup from buffering.
This is short-context evidence, not long-context or production qualification.
Sparse target alignment, mass conservation, seeded sampling and EOS checks also
passed on H100 in the subsequent qualification attempt.

The first matched-throughput attempt failed in report summarization after saving
its samples (missing prompt_tokens). It is preserved as negative evidence; the
metadata fix is measured in a new attempt that checks source hashes against the
passed qualification. vLLM image setup initially failed because the upstream
image exposes python3 without python, then because an environment layer followed
a runtime source mount. The fixes preserve the pinned image and use copy=True;
the SDK overlay pins protobuf 6.33.5 to respect the runtime's <7 requirements.
None of these failed attempts are counted as throughput measurements.

`campaign_modal.py` implements an eight-hour window, with independent teacher
subprocess deadlines and controller cancellation. It remains unlaunched until
qualification and comparison finish. The selected backend is recorded before
spawning. A kernel selection requires at least a 10% aggregate throughput win;
otherwise use the verified faster vLLM path. A canary must verify actual trajectory
persistence before expanding to the requested 256-H100 ceiling. Actual allocation
may be lower. Workers stop retrying after two failures. The archive keeps all
raw trajectories with an explicit syntax/format acceptance flag, including
rejections, in a private HF dataset. No semantic-correctness claim is made.

Qwen3.5 uses a different vocabulary from the old 215.9M student. Native teacher
IDs and tokenizer identity are saved separately. These targets cannot be fed to
the old student's token-wise KL objective. Retokenized text can support sequence
distillation; probability distillation requires an explicit vocabulary solution.
Old Qwen3-8B data and checkpoints are preserved. No student is changed here.

CPU validation: 18 tests pass. Generated outputs and full reports are archived
on HF rather than committed to Git. The new qualification campaign conservatively
retains the previous $5 reservations and reserves each bounded H100 attempt before
launch; no failed launch is automatically refunded. The user explicitly authorized
this follow-up and an eight-hour teacher campaign, superseding the earlier pilot
scope. Failed attempts and actual GPU execution remain distinguished.

## Batch and fusion follow-up

`megakernel/batch_probe.py` is a partial-batching prototype: independent cache and
recurrent state per lane, sequential fused transformer calls, and one shared
batched vocabulary projection. It is not a batched transformer. On the October 22
H100 probe, batches 4/8/16/32 matched the original logits exactly and improved the
short teacher-forced decode measurement by 1.091/1.106/1.114/1.118x respectively.
Batch-one's first timing contains warmup overhead and is not evidence of a gain.
These short measurements do not qualify a new free-running production backend.

The original matched fixture benchmark completed all 288 vLLM requests, with all
288 accepted. Concurrency 8 measured 24.08 and 24.52 saved tokens/s, compared with
16.08 in the kernel's completed round; its second round was preempted. Per-record
volume commits dominated these measurements. Full outputs and negative attempts
are archived under `jfrick/lm-exploration-results/qwen35-buffered-targets/`.

`qualify_campaign` now compares actual curriculum generation with batch-level
writes, preserving native token IDs and top-64 targets. vLLM sweeps 1/4/8/16/32
and the reverse order; the kernel uses two sequential 32-example trials. Each
trial uses the same prompt indices. Benchmark repeats remain isolated from the
fresh production corpus. The two H100 test children share a 30-minute absolute
deadline, enforced by a separate remote CPU supervisor. Earlier canaries failed
container imports; enabling source inclusion and the explicit `/experiment`
module path fixed deployment. Those attempts are not timing results.

`launch_campaign.py` requires two complete curriculum trials per candidate,
reference/target correctness and the executable vLLM fixtures, records measured
rates and report hashes, and only then launches the chosen backend. The
production window is eight hours, with a 256-worker ceiling and an additional
absolute launch-to-shutdown bound. Its default invocation is a dry run.

The parent PR #36 was also checked at `8be3c2c`: its eight-build H100 sweep keeps
the original manual 512-thread/132-block kernel. The best ThunderKittens variant
showed less than 0.01% difference, so no alternative fused build is promoted here.
Our CUDA math uses that original configuration. The parent's later checkpoint
batching work independently identifies the same persistence bottleneck.

The first corrected curriculum attempt exposed truncation at the 1,024-token cap
(only 2 of its first 6 vLLM records were complete/accepted). The next comparison
uses 2,048 for both backends. Invalid and truncated outputs are retained with
accepted=false; neither syntax checks nor the teacher's explanations establish
semantic correctness. The tokenizer call explicitly requests return_dict=false,
validated locally against the pinned tokenizer (187 integer IDs for prompt 0).

The production ranking uses accepted tokens divided by durable-output seconds,
pooled across two complete trials. Acceptance fractions are reported separately;
rejected outputs contribute zero usable tokens. A candidate with zero accepted
examples in either round is excluded. This avoids optimizing raw token counts
or treating an arbitrary acceptance fraction as a throughput objective. Before
launch, an actual selected-backend shard must pass SHA256, token/target length,
top-64/selected-probability and residual-mass integrity checks. The source report
hashes, qualified commit and production source hashes enter the campaign manifest.
HF files, tokenizer and indexes are namespaced by campaign, so another run cannot
overwrite the same producer/chunk names. Final controller state is archived after
the generation deadline and drain period.

## Selected backend and launched campaign

The 2,048-token curriculum comparison selects vLLM, batch 32. Its two complete
32-example trials measured 904.83 and 839.47 accepted tokens/s, pooled **871.62**;
53/64 examples passed JSON/syntax checks. The fused kernel's two sequential
32-example trials measured 97.74 and 97.91, pooled **97.82**, with 54/64 accepted.
The selected vLLM path is **8.91x** faster on this accepted-token/durable-write
metric. First-pass vLLM batches 1/4/8/16 measured 76.83/217.46/364.56/640.81.
After both batch-32 trials beat every smaller first-pass rate, remaining slower
repeat trials were cancelled; this is not a complete ten-trial sweep. Partial
results and the early-stop decision are preserved. The partial-head prototype
remains experimental and is not the production backend.

Production run `qwen35-python-math-8h-20261022-01` started at 2026-10-22 06:06:17
UTC and has an automatic generation deadline of **14:06:17 UTC**. The deployed
app is `lm-qwen35-distillation-teachers`; controller call
`fc-01M5H3F88C8E7S3Z8FZAWA927D`. The configuration requests 256 H100 workers; this is
a ceiling, not evidence of actual allocation. Data, native tokenizer and sparse
targets go to private HF dataset `jfrick/qwen35-python-math-trajectories-20261022`,
under campaign-specific `data/` and `runs/` paths, as well as the Modal volume.
Qualification reports are in `jfrick/lm-exploration-results`, folder
`qwen35-buffered-targets/qwen35-batch-qualification-20261022-04*`.

23 CPU tests, Ruff, the pinned-tokenizer check and H100 target payload audits pass.
The gates establish numerical/storage integrity and measured throughput; they do
not establish semantic correctness of every generated exercise. Only complete,
format/syntax-accepted records contribute to the reported usable token rate.

## Interrupted controller and continuation

The initial controller stopped at 06:13:02 UTC after about 405 seconds. Its
unconditional finally block cancelled teachers, and a restarted invocation then
rejected the existing campaign. The dataset retained 3,137,643 generated tokens,
including 2,403,082 format/syntax-accepted tokens, all archived.

The corrected controller adopts saved call IDs on restart and cancels teachers
only at the fixed deadline. `maintain_campaign` independently watches and
restores the controller, preserving the same deadline across restarts, and
cancels outstanding queued/running calls at expiration. Explicit continuation
keeps worker progress and archive indexes. Tests simulate interruption without
cancelling teachers, adoption without deadline extension, and deadline cleanup.
27 CPU tests and Ruff pass. No live worker was interrupted to test recovery.

The unused 28,395 seconds were resumed under epoch `resume-20261022-01`, with a
new fixed deadline of **2026-10-22 18:07:00 UTC**, superseding the original deadline.
Guardian call: `fc-01M5HHME95SB028S89NH0H6NQD`. The same campaign/dataset paths are
used; trajectories continue from each producer's saved index. Further kernel
experimentation must not redeploy or interrupt this active campaign.

### Continuation toward 5B fresh tokens

The user increased the cumulative generation target to 5 billion fresh output
 tokens and explicitly requested parallel vLLM generation. The first completed
window saved 322,522,448 generated tokens (247,617,420 JSON/syntax-accepted)
on the Modal volume. Only 39,286,782 generated tokens reached HF before its
256-commits/hour rate limit stopped the archiver. The archiver now spaces commits
and retries HTTP 429 using the same batch, without advancing the durable index
until the upload succeeds. Existing bounded eight-hour windows and independent
shutdown remain in force; a single continuation does not complete the 5B target.
The 5B target counts generated output tokens, not repeated training exposures.

### L40S elapsed-time benchmark

`hardware_benchmark.py` is an isolated, explicitly requested comparison of
Qwen3.5-9B BF16 on L40S and exact H100 hardware (`H100!`), at batches 32 and 128.
Each configuration generates the same 256 curriculum prompts and saves native
top-64 probability targets. Four workers run concurrently, with no retries and
an independent controller cancelling remaining calls at the persisted 20-minute
absolute deadline. Production teachers remain stopped.

Results include hardware identity, wall time including startup, per-batch time
through the durable Volume commit, generated/format-accepted tokens per second,
and a first-shard probability/storage integrity audit. The audit does not prove
semantic correctness or numerical equivalence to another GPU. Rates exclude
startup and prompt preparation; report wall time separately. This small test
selects candidates for sustained load validation, not a guaranteed fleet rate.

The initial controller import failed; the missing `/experiment` module search
path was fixed before the second attempt. A redeployment/cancellation race let
one old L40S batch-32 child start and delay the new batch-128 child. Its persisted
call was explicitly cancelled after discovering it in the logs. The old attempt
is excluded from the comparison. Benchmark receipts and
results belong under `qwen35-buffered-targets/` in the existing HF results repo.

All four 256-example runs completed. H100 batch 32 reached 1,343.7 generated /
961.0 format-accepted tokens/s; H100 batch 128 reached 1,243.1 / 895.4. L40S
batch 32 reached 524.7 / 383.9; L40S batch 128 reached 1,153.4 / 882.9.
All first-shard target-integrity audits passed. L40S needs larger batches to
approach H100 throughput on this workload; the tested H100 batch-32 configuration
still leads by 16.5% in generated tokens/s and 8.8% in accepted tokens/s.
Fleet ETAs are linear projections, not measured scaling results. With the saved
production corpus at 641.6M generated / 492.8M format-accepted tokens, ten H100s
at this benchmark rate would need about 7.4 / 14.7 hours to reach the respective
1B milestones. Ten L40S at batch 128 project to 8.6 / 16.0 hours. Fifty H100s
would project to 1.5 / 2.9 hours, subject to actual quota, allocation and sustained
pipeline throughput. All benchmark GPU and controller functions returned to zero
running inputs and zero backlog after completion; production stayed stopped.

### Maximum-capacity continuation

The user authorized resuming generation with the maximum available parallel GPU
capacity after the hardware benchmark. Reused the qualified vLLM/H100 batch-32
configuration, with up to 256 requested producers and the existing independent
eight-hour shutdown enforcement. The manually stopped prior campaign was recorded
as stopped before explicit resume; original producer offsets and archive indexes
are preserved. Launch receipts are archived in the results repository.

The environment API reports no explicit environment GPU override and no spend
limit block. It does not expose the inherited workspace GPU quota. The resumed
campaign reached 10 running H100 teachers with additional calls queued, matching
the earlier measured allocation ceiling. This confirms allocated capacity, not
a formal account quota or a guarantee of future availability. No plan, quota,
or billing settings were changed.

### Reduction to five H100 teachers

At the user's request to free GPUs for other jobs, cancelled the previous guardian
and controller, cancelled all 251 producer calls outside IDs 0–4 (including queued
calls), and preserved the five selected teacher calls. Applied a live Modal
function autoscaler hard cap of five containers and changed the persisted active
producer count to five before starting the replacement guardian/controller. The
original deadline is unchanged. The existing archive call continues scanning all
256 historical producer IDs; `dataset_workers=256` records that coverage, while
`workers=5` is now the active count. Status totals must include historical producers,
not just IDs below the active count. Capacity-change receipts are archived in HF.
No teacher inference source or deployed images changed for this reduction.
