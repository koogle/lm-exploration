# 215.9M shared-Qwen continuation to 4.5B tokens

The corrected user target is **4,500,000,000 shared-Qwen-tokenizer training
targets from a single pass over freshly generated data**, including the completed
4,500,000-token update. Dataset replay is disabled. This supersedes the
large student's original target in [the historical sweep plan](SCALE_RUN.md).

Launched 2026-10-21 on [Modal](https://modal.com/apps/radiantai/main/ap-7elUlhJSfXhz3tZEqgV2uD).
Run output on volume `lm-exploration-distillation`:
`python-distillation-500m-20261021-01/shared-qwen-215m-4p5b/`.

- Exactly 215,910,656 parameters; existing shared Qwen tokenizer, unchanged
  vocabulary and weights. No randomly initialized replacement model.
- Initialize from the completed sibling `existing-216m/latest.pt`, preserving
  optimizer, RNG, source cursor, 817 updates and 4.5M consumed targets. A migration
  record retains the old settings; the original checkpoint remains intact.
- Continue consuming the newly generated Qwen3-8B Python/math problem, code and
  explanation corpus. Wait for more shards if the first pass catches generation.
  After the initial 500M **original-tokenizer** corpus finishes, append shards
  from `python-distillation-unique-qwen-20261021-01`. Never wrap to the start.
  Appending only after the original corpus closes preserves its consumed order.
- Count every loss-bearing Qwen target exactly, trimming the final batch to 4.5B.
  Checkpoints separately retain first-pass tokens and the zero-based corpus pass.
  Corpus pass remains zero. The migration rejects a checkpoint already on a
  repeated pass; the selected 21,639,566-token checkpoint had no replay.
- All-token cross entropy sequence distillation, context 512, H100, BF16 autocast,
  AdamW with existing moments, batch four accumulated four times. No probability
  KL in this phase. Teacher code is syntax checked; semantics remain unverified.
- LR ramps from approximately 3e-6 to 3e-5 over the next 1M tokens, then follows
  cosine decay to 3e-6 at 4.5B. The cosine uses the full corrected token horizon;
  ramp and schedule do not reset between segments.
- Validation every 10M, diagnostic generations every 50M and at completion;
  checkpoint optimizer/cursor/RNG at least every 10M and each segment end.
  Weights retained after each up-to-50M segment. No loss-based early stop.
- A CPU supervisor starts one training worker at a time, persists its call ID,
  and continues across the 24-hour invocation limit. Workers retry twice from
  the last durable checkpoint; terminal errors are recorded in supervisor.json.
  GPU workers exit when they exhaust available data; the CPU waits for new shards.

This is about 20.84 training tokens per total parameter. It is **not a claim of
Chinchilla optimality**: this is continued training on synthetic data, and
the vocabulary embedding accounts for a large fraction of
parameters. Generalization must be judged on held-out execution tasks.

Launch once from the repository root:

```sh
modal deploy -m distillation.modal_billion
python -c 'import modal; print(modal.Function.from_name("lm-distillation-215m-4p5b", "supervise").spawn().object_id)'
```

Inspect `supervisor.json` for running/waiting/failed state and active function
call, `progress.json` for durable trained-token count, `metrics.jsonl` for updates,
`validation.jsonl` and `generations-*.json` for diagnostics. The original 500M
corpus and small-student jobs retain their separate configuration.

Validation for the fresh-data policy: all 21 distillation tests passed, including
exact partial-record resume, Qwen EOS alignment, stable append ordering and no
replay after exhausting both corpora. Ruff checks passed.

## Supervisor recovery

On 2026-10-21 the initial run-app supervisor was cancelled at 14:47 UTC after
saving 21,639,566 targets (3,923 steps). Validation CE was 1.55013, versus
1.62070 at 4.5M. This was an orchestration interruption, not a validation stop.
The app was subsequently deployed and the supervisor spawned through its deployed
function (`fc-01M5FN8X0CWRTD15M2EPVXGGCS`) to resume the existing checkpoint.
Use the deployed launch above for persistent background work.
Deployment: https://modal.com/apps/radiantai/main/deployed/lm-distillation-215m-4p5b

## Fresh-data correction

The user rejected the repeated-pass plan before any repeated pass occurred.
The controller and worker were cancelled, then the 21,639,566-token checkpoint
was migrated with optimizer, RNG, cursor and learning-rate schedule preserved.
`unique-policy-migration.json` records the before/after settings on the volume;
`source-unique-policy/` archives the new executable source. The new deployed
supervisor call is `fc-01M5FNFQX6D5RBMMQQ2YW3VPGY`.

Four additional H100 Qwen3-8B producers generate a continuation corpus. Each has
an upper bound of 1.125B accepted **shared-Qwen** tokens, not original-tokenizer
tokens. Prompt indices start at 100,000,000 and sampling seeds change, avoiding
reuse of the first corpus's prompt sequence. The first corpus also counts toward
the student's 4.5B target, so some generated tail can be unused. The supervisor
cancels remaining continuation producers when the student reaches exactly 4.5B.
It resumes producers at their committed cursors after cooperative time limits.

Fresh means newly sampled records consumed once, not 4.5B distinct vocabulary
items or a guarantee of semantic uniqueness. Exact full-record deduplication is
within each producer; cross-producer and semantic duplicates remain possible.
Generation still checks syntax rather than independently verifying correctness.

Launch verification: all four continuation producers published shards (over 490K
accepted Qwen tokens in the first observed snapshot), the persisted training
request has `repeat_completed_corpus: false`, and live optimizer updates resumed
through 24,060,054 tokens. The original corpus supervisor was also restored as a
deployed background call (`fc-01M5FNMC6Y60B93361KYQ8FD7Y`), reattaching its persisted
producer calls so the initial corpus can finish and unblock the continuation.
