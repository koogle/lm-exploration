# Authorized 500M-token sweep

Run `python-distillation-500m-20261021-01` was launched on Modal on 2026-10-21:
https://modal.com/apps/radiantai/main/ap-ed39mfL1LCi5IGKaCOluxg

The user explicitly requested the large run after the earlier bounded pilots.
There is no additional approval or validation-loss gate. The requested larger
model is the existing **215,910,656-parameter** checkpoint (described by the user
as 250M); no separate 250M pretrained checkpoint was found. The 4.5 million
token request is interpreted literally, not silently increased to 4.5 billion.

| Work | Target | Initialization |
| --- | ---: | --- |
| Fresh synthetic corpus | 500M accepted original-tokenizer training tokens | Qwen3-8B teacher |
| 26,776,960-parameter student | 500M exposures, one corpus pass | Random seed 42 |
| 48,308,408-parameter student | 500M exposures, same order | Random seed 42 |
| Existing 215,910,656-parameter student | 4.5M additional shared-tokenizer targets | `qwen4b-shared-50m-20261021-01/step-000007162.pt` |

## Actual generation policy

Four H100 workers each generate 125M accepted original-tokenizer targets. Each
samples 128 prompts at a time from 62 Python/math topics, varying scenario,
difficulty, boundary conditions, explanation focus and numeric constants.
Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16, temperature
0.85, top-p 0.95, at most 1,024 output tokens; thinking disabled. JSON responses
contain problem, code and explanation including worked examples. No source-code
database or previous small corpus is repeated to reach the target.

Acceptance requires a complete JSON response, nonempty fields, parseable Python,
and a top-level `solve` function. **This broader corpus is syntax-checked, not
independently semantically verified.** Generated code is not executed during
generation. Explanations and worked examples are teacher supervision, not proven
facts. Exact full-record duplicates are excluded within each producer; semantic
and cross-producer duplication are not eliminated. Rejections and raw responses
are retained. Topic combinations improve breadth over eight constant-substitution
families but do not establish independent examples or benchmark generalization.

Compressed shards retain text, original 32K IDs, and IDs from the existing large
student's archived tokenizer. IDs include EOS but not the input-only BOS. Only
the last original-tokenizer record of a producer can be truncated to meet its
125M-token budget exactly; it is flagged. The 500M budget includes problem text,
code and explanations receiving CE loss. Teacher generation tokens and rejected
outputs are counted separately. The large student's 4.5M budget uses its own
tokenizer and exact target trimming.

## Training and evaluation

This is **sequence distillation with all-token CE**, not full-vocabulary KL.
Both fresh models use the original frozen 32K tokenizer. The existing model
retains its shared Qwen vocabulary and all pretrained weights. No vocabulary
resize occurs in this run. Context 512, document-local windows, every target once,
padding ignored. Fresh models use microbatch 8 and accumulation 4; the existing
model uses microbatch 4 and accumulation 4. AdamW betas 0.9/0.95, weight decay
0.1 and gradient norm cap one; BF16 autocast with FP32 master weights.

Fresh models: peak LR 3e-4, 1M-token warmup, cosine decay over the full 500M
budget. Existing model: peak LR 3e-5, 100K-token warmup, cosine decay over 4.5M.
The schedule follows total exposure count across segments and does not restart
at each worker invocation. Segment ends do not imply learning-curve plateaus.

Evaluate on the fixed accepted validation records from curriculum batch 02:
every 10M tokens for fresh models and every 1M for the existing model, plus each
segment end. Save matched eight-task generations initially, every 50M for fresh
models, every 1M for the existing model, and at the final target. This remains a
small diagnostic suite, not a broad Python benchmark. Raw token CE differs
between tokenizers and cannot rank the 216M model against the small models.
Equal data exposure does not imply equal training compute.

## Persistence and operation

```sh
modal run --detach -m distillation.modal_scale \
  --run-id python-distillation-500m-20261021-01
```

Use the usual truststore wrapper in the managed environment. Do not start two
supervisors for the same run. The dedicated distillation volume stores:

- `RUN/plan.json`, `status.json`, `calls.json`: targets, progress and remote call IDs.
- `RUN/producer-N/progress.json`: accepted targets, raw teacher tokens, rejection
  counts, next prompt index and committed shard hashes.
- `RUN/{fresh-27m,fresh-48m,existing-216m}/latest.pt`: atomic resume checkpoint
  with weights, optimizer, RNG, token count, configuration and exact data cursor.
- Model `progress.json`, `metrics.jsonl`, `validation.jsonl`, checkpoint
  generations, and weights-only snapshots at completed segment boundaries.

Read the current status without modifying jobs:

```sh
python -m distillation.scale_status python-distillation-500m-20261021-01
```

The snapshot distinguishes total accepted corpus tokens from the deterministic
prefix currently consumable by students; one slower producer can hold back that
prefix while the others continue publishing. Producer totals are live reads,
not a single atomic cross-worker snapshot.

At most four teacher and three student H100 workers can run concurrently.
The CPU supervisor waits for published data. Fresh students start at 5M available
tokens and run segments of up to 10M; the existing student starts at 6M original
tokens to cover tokenizer differences. Students checkpoint approximately every
2M tokens and at each segment end, then release their GPU while waiting for data.
Round-robin shard order is deterministic despite different producer speeds.

Generator workers stop cooperatively before Modal's 24-hour timeout and are
resubmitted at their durable cursor. The CPU supervisor can continue into another
bounded invocation and reattach its recorded child calls. Work ends at the finite
targets. Validation regression never stops training. Nonfinite training values
and child failures are recorded as failures; they do not silently restart an
unbounded loop. An interrupted run can be explicitly relaunched with its same
run ID and saved state after ensuring the previous supervisor has stopped.

Status at launch: four teacher workers started; student launches are automatic
once data thresholds are met. Read live `status.json` for completion instead of
treating this launch record as a completed 500M result.
