# Frozen longer-rollout and sealed-test evaluation

User authorized both remaining tasks: longer verified generation and the sealed
held-out drafting test. Evaluate only the validation-selected long-response head
(step70585, inherited in `fresh-grid-196608-r256-11/best.pt`). No architecture,
weights, sampling parameters or checkpoints are selected using test results.

Predeclared settings: [protocol](../configs/qualification-long-196k.json).
First hash and verify checkpoint, archived training report, prompt manifest and
standalone evaluator, and commit `frozen.json` before test access. Require the
existing validation macro>=2 NEW/7 and every-domain>=1 gate. Use the same frozen
Qwen revision, tokenizer, five normalized draft layers and block8 architecture.

Long check: the first eight archived validation prompts/domain, 24 total, up to
256 output tokens (EOS may stop earlier). Compare plain greedy and verified draft
outputs on the same stored target weights upcast to FP32. Record every output,
accepted NEW lengths/call, source identity, length and repetition. Single timing
measurements are diagnostic, not a speed benchmark. Any parity failure leaves
test labels unopened and fails this qualification input.

Only after parity passes, audit all reserved test sources/prefixes against the
manifest train/validation records, then generate target test continuations under
the original BF16 arithmetic, maximum256 tokens, batch32. Exclude under-block
responses and exact response duplicates against the existing teacher pool or
earlier test responses, matching the prior preparation policy. Report every
exclusion and per-domain coverage. Test has192 reserved prompts (64/domain).
Evaluate the same deterministic up-to-eight anchor positions/response and report
macro/domain NEW, NLL, draft-position accuracy and per-example metrics. The
working test gate is macro>=2 and every-domain>=1 NEW/7. No test-based retuning.

Standalone `qualification_worker.py` and `modal_qualify.py` leave the frozen
training package untouched. One single-use L40S input, 3600s hard bound/3300s
work deadline, conservative$5 reservation. Save progress/artifacts to the
research volume after every rollout and teacher batch. No automatic retries;
run directories refuse overwrite. Interrupted test work must preserve the same
frozen selection/config and already-generated labels; never launch a new selection.

Implemented local safeguards: source-group and exact-prefix leakage rejection,
parity fails closed on any mismatch/empty check, domain floor cannot be hidden by
macro success. GPU evaluation is still pending at protocol publication.
