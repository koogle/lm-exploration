# Completed modern 9.85M / 5B run

The 9,849,536-parameter model completed 72,805 optimizer updates and
4,999,934,057 nonpadding next-token exposures. The final checkpoint is on
`lm-exploration-training` at
`scholarly-modern-10m-5b-20261017-s02/step-000072805.pt`.

The target stopped 65,943 tokens below 5B to avoid exceeding the ceiling with
another whole global batch. Source exposures match the planned mixture to three
decimal places. The frozen corpus contains 2,745,331,556 eligible training tokens;
all per-source exposures remain within two passes. Tokens are exposures, not a
claim of five billion unique tokens.

The three recorded worker sessions total 11,182.67 seconds (3.11 hours), about
447,114 trained loss tokens/second including worker-side overhead. Corpus
preparation, GPU scheduling and this evaluation are excluded. The run used three
H100s, context 2,048, global batch 48 windows, BF16 autocast/FP32 parameters and
the original synchronous raw-text loader. The producer/consumer feature in PR #15
is separate and did not produce this checkpoint.

## Integrity and continuity

All 72,805 metric rows have consecutive step numbers. Their target counts sum
exactly to the checkpoint total. All recorded loss, gradient norm, learning rate
and duration values are finite. All final parameters and optimizer states are
finite, optimizer step counters equal 72,805, source exposures sum correctly,
and the diagnostic validation at both resume boundaries matches exactly.
The source/model/tokenizer/data hashes are pinned in the retained artifacts.

Inference export passed strict loading, exact safetensors weight round trip,
exact CPU logits round trip and tied-embedding storage checks. Optimizer and
per-rank RNG state stay in the trusted Modal checkpoint; the Hub bundle is for
inference and new fine-tuning, not exact optimizer continuation.

## Quality evaluation

The broader evaluation selects 32 distinct held-out parents per source (384
parents total), with at most four document-local 2,048-token windows per parent.
Final and seed-42 random models are scored on identical targets. The final report
records per-source token counts, cross-entropy and perplexity, with mixture-weighted
aggregate loss. It also retains all greedy and sampled completions for ten fixed
prompts, including poor outputs, rather than selecting a preferred seed.

Published model: https://huggingface.co/jfrick/scholarly-code-modern-gpt-10m-5b .
Immutable revision: `772934ca988ed63701346c97ce3152d3c8c36ab9`.
The publicly downloaded safetensors SHA256 matches the exported weights.

Across 1,798,432 held-out targets, the final model has mixture-weighted loss
2.6926 nats/token (exp(loss) 14.77), versus 10.4369 for the matched seed-42 random
baseline. Samples remain repetitive and fail basic arithmetic/code prompts.

See `scholarly-modern-10m-5b-validation-20261018.json` and the public model card
for exact measured quality results. The training-time one-window/source diagnostic
ended at 2.6051 nats/token; that is a separate, much smaller sample.

The model learns text patterns but generated examples exhibit repetition,
invented terminology and factual errors. It is a small experimental base model,
not an instruction-tuned assistant. Pending near-duplicate, cross-source parent,
tokenizer-split, eligibility and contamination audits mean these measurements
are exploratory rather than a sealed benchmark result.
