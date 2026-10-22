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
