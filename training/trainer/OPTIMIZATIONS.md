# Eight-H100 training optimization experiments

These opt-in execution settings are stacked on PR #16. Defaults retain the
uncropped eager dense-loss path. Do not deploy `modal_simple_llm_8k.py` to evaluate
these changes: the active 5B-token application retains its original source image.

- `crop_padding`: trim each microbatch after its last nonignored label, retaining
  the entire causal prefix and interior masks. Fully ignored microbatches retain
  one position and contribute zero gradients while joining DDP collectives.
- `microbatch`: compare 1, 2 and 4 at global batch 32 / eight GPUs. The same loader
  produces the same windows. DDP gradients remain normalized by the global count
  of nonpadding labels, not by each microbatch's length.
- `loss_chunk_size`: checkpoint the output projection and sum cross entropy in
  chunks of this many token positions. Backward recomputes each chunk, bounding
  live vocabulary logits. This is standard PyTorch chunking, not a custom fused
  kernel; it may trade speed for memory and must be measured.
- `cache_rope`: precompute FP32 trigonometric tables once per attention layer.
  Nonpersistent buffers keep checkpoint tensor names unchanged.
- `compile_model`: compile the model in place with dynamic sequence dimensions.
  Keep the original module and optimizer parameters for checkpoint serialization.

Run from `training/`, using a fresh prefix. For unattended execution, deploy only
the isolated benchmark app and spawn its controller:

```sh
uv run python -c 'import truststore; truststore.inject_into_ssl(); import runpy; runpy.run_module("modal", run_name="__main__")' \
  deploy modal_training_optimizations.py
```

Then use the authenticated SDK to call
`modal.Function.from_name("simple-llm-optimization-benchmarks", "suite").spawn(prefix, provenance)`.
Record its function-call ID. The earlier detached local entrypoint was canceled
while waiting for GPUs; it produced no GPU verification or benchmark measurements.
A deployed, spawned controller does not depend on that local CLI session.

The isolated app runs one eight-H100 worker at a time, with 32 CPUs and 64 GiB
host RAM. It does not publish weights or modify the active 5B app.
Each variant has a 100M nonpadding-token ceiling, a one-hour continuation limit,
and no retry. A ceiling can end slightly early to avoid consuming an incomplete
optimizer batch. Time-limited results must not be reported as completed 100M runs.
The matrix uses identical initialization, data order, global windows, context,
20M warmup and a shared 100M-token cosine schedule. This shortened schedule is
for comparing optimizations, not continuing the production schedule.

The CUDA gate compares loss and gradients against full-batch FP32, including a
fully masked rank, FP32/BF16, accumulation, chunked loss and compilation. Each
variant saves after two steps, starts a new torchrun process, restores all saved
training state and requires exact diagnostic validation agreement before its
results are accepted. This validates restore, not bitwise equality of training
trajectories between different microbatch sizes.

Resume keeps the existing strict configuration/source/PyTorch identity checks.
Old #16 checkpoints are intentionally rejected by the changed training source;
this PR does not authorize migrating the live run. Tensor state remains strictly
loadable, including tied embeddings, and cached RoPE does not add saved tensors.

Per-step JSON includes nonpadding tokens/sec and all eight ranks' peak allocated
and reserved CUDA memory, supervised token counts and synchronized forward/backward/
optimizer time. Rank times include DDP communication/waiting, not pure kernel time.
Report both token skew and time skew; GPU utilization alone includes padding work.
The benchmark summary excludes the first ten continuation steps from throughput,
but records whole-session wall time and peak memory including startup/evaluation.
Compilation may recompile beyond warmup; its costs remain in those measurements.
Results and failed variants are retained in the dedicated volume prefix. Raw
metrics, source snapshots, checkpoints, provenance and logs accompany each run.

## Single-GPU path

Pass `single_gpu=True` when spawning `suite` to use one H100 per variant.
The same seven variants retain global batch 32, real snapshot order, 8K context,
and 100M nonpadding-token ceilings. Accumulation becomes 32/16/8 rounds for
microbatches 1/2/4. The correctness gate runs on one CUDA rank and each variant
still restores its two-step checkpoint in a fresh process. These results measure
single-GPU performance; they do not establish eight-GPU scaling, communication
cost, or inter-rank imbalance. Those require the separate eight-GPU suite.
Only one single-GPU experiment runs at a time; the active 5B app is untouched.

For the independent single-GPU deployment, deploy
`modal_single_gpu_optimizations.py`, then spawn
`modal.Function.from_name("simple-llm-single-gpu-optimizations", "run_suite").spawn(prefix, provenance)`.
This is the current launch route: `opt100m-1gpu-20261018-b`. It avoids the older
eight-GPU controller's deployed signature. The initial single-GPU submission
failed before GPU allocation with an unexpected-keyword error and has no results.

## Eight-GPU finalist comparison

`modal_eight_gpu_finalists.py` runs exactly three variants: cropped/cached-RoPE
microbatch 4 eager, microbatch 4 compiled, and microbatch 2 compiled. Each uses
eight H100s, global batch 32 and a 100M nonpadding-token ceiling. It runs the
12-case CUDA qualification on all eight ranks, including an empty rank, before
training. Compiler worker count is limited to one per rank to bound host memory.
The run prefix is `opt100m-8gpu-finalists-20261018-a`; each variant retains full
per-rank token load, synchronized compute times and memory metrics. No production
training or checkpoint is modified.
