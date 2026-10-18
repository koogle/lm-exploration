# Scholarly/code subword pilot (historical)

The duplicate pilot launcher and loop described below have been removed. Use
[`../distributed_training/`](../distributed_training/README.md) for new runs,
including one GPU. The commands/settings below document the measured historical
experiment, not an available entrypoint or settings identical to the current
trainer. Its code remains in Git history; its checkpoints require that code.

`modal_scripts/modal_scholarly_pilot.py` connects the merged modern 100M-family model to the
frozen 32K tokenizer and `TokenMixture`. It preserves the merged 14×768 shape
(124,389,632 unique parameters), with context increased to 1,024. This is a
bounded execution/learning pilot, not the planned 20B-token run.

From `training/`, first prepare a fresh snapshot with `modal_scripts/modal_prepare_scholarly.py`.
Then:

```sh
modal run --detach modal_scripts/modal_scholarly_pilot.py \
  --run-id scholarly-124m-pilot-20261016-01 \
  --snapshot scholarly-124m-20261016/snapshots/pilot-001
```

One L40S worker, four CPUs, 16 GiB host RAM, no retries, and a 30-minute hard
worker timeout. Training stops at 600 optimizer updates, 5M loss-bearing tokens,
20 minutes including intermediate checkpoint overhead, or source exhaustion.
The existing loader enforces at most two passes per source. No full pretraining
or automatic continuation is launched. The pilot deliberately permits snapshots
with pending corpus audits; this override must not be used for substantial runs.

Optimization: FP32 parameters, BF16 autocast, AdamW (betas 0.9/0.95, weight decay
0.1), gradient norm clipping at 1, linear 30-update warmup to 3e-4 then constant
learning rate. Batch size 2 with four accumulated microbatches; gradients are
normalized by actual nonpadding targets across the complete optimizer batch.
This short pilot does not select a production learning-rate schedule.

The CPU qualification runs modern-model, tied-embedding and corpus-loader tests.
The GPU worker verifies tiny-set fitting, then reloads the actual step-10
checkpoint and checks identical next-batch IDs, labels and logits. Checkpoints
include optimizer/RNG/data state, source hashes, model settings, token counts and
manifest/tokenizer identities. Checkpoints and logs use fresh paths on
`lm-exploration-training/<run-id>/`; the final report is also copied to `results/`.
This pilot checkpoint format is separate from the character-only `lm-generate`
format. Cross-job resume and generation CLI integration remain future work.

Validation uses one fixed document-local window per source, reporting each source
and the token-mixture-weighted mean. This small provisional validation sample is
for detecting learning/failures, not capability claims or production checkpoint
selection. Samples use three fixed prompts, seed 42, full-softmax temperature 1,
and up to 64 new tokens. PAD/BOS/UNK are excluded from sampled outputs.

The full corpus still needs split reconciliation, global near deduplication,
provenance/eligibility and contamination audits, and capacity evidence. Measure
throughput before selecting the substantial-run hardware and budget. The current
model uses causal SDPA, introduced separately from the pilot. The historical
pilot used explicit attention, with its source preserved alongside the run.

In the managed cloud environment, Modal's client needs the system CA trust:
install `modal[api-proxy-support]==1.6.2` and `truststore`, call
`truststore.inject_into_ssl()` before running the Modal CLI module. Keep inherited
proxy settings and TLS verification enabled.

## First measured run

[The 2026-10-16 report](../results/scholarly-124m-pilot-20261016-01.md) records
600 updates / 4,078,340 tokens, weighted pilot validation loss 10.5811 → 6.0272,
23 passing qualification tests, and a successful checkpoint/data reload.
The final checkpoint is `lm-exploration-training/scholarly-124m-pilot-20261016-01/step-000600.pt`.
Samples remain incoherent; full corpus audits and substantial training are pending.
