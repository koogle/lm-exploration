# 215.9M shared-Qwen continuation to 4.5B tokens

The corrected user target is **4,500,000,000 shared-Qwen-tokenizer training
exposures**, including the completed 4,500,000-token update. This supersedes the
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
  Repeat the full corpus only after all four producers finish their 500M total
  **original-tokenizer** token budget. That budget differs from the Qwen count.
- Count every loss-bearing Qwen target exactly, trimming the final batch to 4.5B.
  Checkpoints separately retain first-pass tokens and the zero-based corpus pass.
  Repeated exposures are not unique examples or fresh teacher generation.
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
Chinchilla optimality**: this is continued training, much of the budget repeats
synthetic data, and the vocabulary embedding accounts for a large fraction of
parameters. Generalization must be judged on held-out execution tasks.

Launch once from the repository root:

```sh
modal run --detach -m distillation.modal_billion
```

Inspect `supervisor.json` for running/waiting/failed state and active function
call, `progress.json` for durable trained-token count, `metrics.jsonl` for updates,
`validation.jsonl` and `generations-*.json` for diagnostics. The original 500M
corpus and two small-student jobs continue under their existing supervisor.

Validation before launch: all 19 distillation tests passed, including exact
partial-record resume, Qwen EOS alignment, no repetition of an unfinished
corpus, and resume across a completed-corpus pass boundary. Ruff checks passed.
