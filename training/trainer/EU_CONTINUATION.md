# Simple LLM: one-pass ECB/EMA continuation

Continue the completed **124,389,632-parameter / 4,999,981,176-token** checkpoint,
not a five-billion-parameter model. The run used optimization PR #17 at
`43cee6bcb7bf4d1d68ea31d73b780787f840a5e1`; #17 has since merged into main.

`modal_eu_continuation.py` deploys a separate app. It never modifies the old run,
its app, or any Hugging Face repository. EU text, checkpoints and generated samples
stay on our existing private Modal volumes. The source datasets are accessed using
the existing `simple-llm-hf` secret; credentials never enter source or artifacts.

## Data and reuse

Pin `jfrick/eu-ecb` and `jfrick/eu-ema` to immutable Hub revisions. Download only
`data/**/*.parquet`, excluding much larger raw PDF/HTML snapshots. Downloads are
bounded and completed shards are tracked transactionally. Disable per-file Xet
token requests, which exhausted the HF API quota in the first attempt; bounded
quota/server-error retries preserve completed work.

Preparation retains the latest retrieval per source/document identity, normalizes
Unicode/line endings and removes exact text duplicates. Assign explicit document
families to train/validation/test using a deterministic 97/2/1 hash split, keeping
recognized translations and format variants together. Exclude exact duplicates
that conflict across splits. Cross-language parent reconciliation is incomplete;
near deduplication, contamination and source-eligibility audits remain pending.
No readiness flag is silently promoted.

Count selected documents with the original frozen 32K tokenizer. Store compressed
raw text, source provenance, counts and an immutable checksummed manifest. Discard
temporary token IDs. Training reuses the existing asynchronous, just-in-time raw
tokenizer. Reusing a completed snapshot verifies its checksum and skips preparation.
Changing source revisions or processing identity requires a new snapshot name.

ECB and EMA retain their natural token proportions in a single EU stream, ordered
by content hash. The target is **exactly one pass over selected training tokens**;
validation/test documents are excluded. No original-corpus replay is mixed in.
The continuation-only loader mode pads the final partial global batch with ignored
labels. It trains every selected target once, including EOS, without a second pass.
Historical two-pass loaders and strict resume policy keep their original defaults.

## Migration and optimizer

Parent: `simple-llm-124m-8k-5b-20261018-s03/step-000064227.pt` on
`lm-exploration-training`. The explicit migration checks architecture, tokenizer,
PyTorch, finite model tensors and Adam moment shapes before writing a new seed.
It preserves model weights, Adam moments/update counts and per-rank RNG. It resets
the phase step/token counters and EU data cursor; checkpoint lineage records the
parent hash, original source hashes, step and token count. This is a new data phase,
not an exact replay of the old training trajectory. Ordinary resume still requires
identical config, source hashes, tokenizer/data identity and PyTorch.

Use a constant `3e-5` learning rate, continuing the parent's terminal rate without
restarting its original warmup. This is a conservative starting choice, not an
optimized domain-adaptation schedule. Record phase and lifetime tokens separately.

## Eight-GPU execution and evaluation

Use eight H100s, 8K context, global batch 32 windows, microbatch two, trailing-padding
cropping, cached RoPE, dynamic compilation and four prefetched batches. Chunked loss
is off. PR #17's independently retrieved completed eight-GPU comparison measured
275,631 nonpadding tokens/sec for compiled microbatch two versus 258,793 for compiled
microbatch four and 213,354 for eager microbatch four on the original corpus. Those
are reference measurements, not a throughput promise for longer EU documents.
Raw evidence is archived on HF as
[`pr-25/training/results/eu-continuation-pr17-eight-gpu-evidence-20261018.json`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-25/training/results).

The new workflow runs the eight-rank CUDA loss/gradient gate, then two updates,
then a fresh-process restore for two updates. Exact diagnostic validation agreement
is required before the full continuation. Checkpoints every 500 updates and session
ends are durable. Segments are capped at three training hours, with five total
segments; failures stop the controller. No GPU is allocated while waiting for CPU
preparation. Compilation cost, startup and GPU scheduling are part of wall time.

Matched before/after checks use up to 32 held-out document prefixes per EU institution
and four original-corpus validation windows per source at 2K, plus ten fixed greedy
128-token completions. Loss uses BF16; generation uses full FP32 with TF32 disabled.
These are bounded diagnostics, not a whole-corpus or long-context evaluation.
Original-domain loss checks can reveal forgetting from EU-only continuation.
Artifacts remain on the private run volume; nothing is uploaded to the public model.

## Operations

Run from `training/`. Deploy with the configured Modal CLI, injecting system TLS
trust in this environment:

```sh
python -c 'import truststore,runpy; truststore.inject_into_ssl(); runpy.run_module("modal",run_name="__main__")' deploy modal_eu_continuation.py
```

Spawn `prepare(snapshot, pinned_datasets)` through the authenticated SDK. Then spawn
`after_preparation(run_id, snapshot, prepare_call_id, provenance)` to durably queue
the bounded GPU workflow. The request folder records its controller call ID once
preparation completes. Deployment alone starts no work.

The launch record (archived in [`pr-25/training/results/`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-25/training/results))
stores call IDs and dataset revisions. Inspect using
`modal.Function.from_name("simple-llm-eu-continuation", "inspect").remote(...)`.
Run status lives in `/runs/<run-id>/workflow.json`; each segment has metrics, a
report and checkpoints. The before/after evaluation files are separate from
training completion. Estimates must use the actual prepared token count and
observed EU throughput; neither is inferred from word/character totals.

Local checks cover optimizer preservation, counter/data resets, strict resume
rejection, latest-record selection, preparation reuse, family splits, final partial
batch coverage, one-pass exhaustion and exact loader replay. Existing optimization,
prefetch and distributed-configuration checks are retained.
