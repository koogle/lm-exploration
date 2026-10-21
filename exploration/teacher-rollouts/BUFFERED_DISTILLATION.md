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
