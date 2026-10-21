# Simple LLM 124M: eight H100s, 8K context, 5B exposures

The authorized experiment starts the modern 124,389,632-parameter model from random
initialization at context 8,192 on eight H100 80GB GPUs on one Modal host. This is
separate from the completed 9.85M / 5B run; its weights are not continued here.

`distributed_training/recipes/h100x8-8k-5b.json` fixes the schedule and batch: 32 document-local
windows, microbatch one per GPU, four accumulation rounds, 20M-token warmup, peak
learning rate 3e-4, 5B cumulative nonpadding target ceiling. Short documents are
padded. A full optimizer batch may finish slightly below the ceiling.

`modal_simple_llm_8k.py` deploys a durable controller. Explicitly spawning `start`
checks mixture capacity in the existing frozen raw snapshot, runs two steps, then
resumes for two more and checks exact diagnostic validation agreement. Only then
does it continue in six-hour segments to the 5B ceiling. It probes eight visible
GPUs and the 8K Flash SDPA path before each segment. Allocation, capacity, finite
training and resume failures prevent automatic continuation. The data is an
explicitly authorized experiment with pending audits; no readiness flags are changed.
The existing roughly 2.75B unique-token corpus supports this exposure budget with
at most two passes per source, so preparation is reused.

After qualification, a separate CPU publisher watches durable checkpoints and
uploads inference weights, exact model code, tokenizer, config, provenance and a
progress card to **jfrick/scholarly-code-modern-gpt-124m-step2**. The original smoke
artifact is archived on branch `smoke-step2-before-8k`. Later uploads replace the
current inference bundle; older weights remain addressable by Hub commit revision.
The completed 10M repository is unchanged. Full optimizer/per-rank RNG/data cursors
remain on the Modal volume for exact training resume.

Checkpoints are written every 1,000 updates and at each segment end. The publisher
polls every 30 seconds and can coalesce intermediate checkpoints if upload is slow;
it retries publication errors without blocking GPUs. Its `hub.json` records the
last successful revision and `hub-error.json` records an upload error. Training
completion and final publication are distinct states and should both be checked.

Run commands from `training/`. Configure the Modal secret `simple-llm-hf` with
`HF_TOKEN` through the authenticated SDK; never commit or print the token.

```sh
modal deploy modal_simple_llm_8k.py
```

Deploying does not start training. Use the authenticated Modal SDK to explicitly
spawn `modal.Function.from_name("simple-llm-124m-8k-5b", "start")` with a fresh
run ID and source provenance. The returned call IDs and `/runs/<run-id>/workflow.json`
track execution; each segment has `torchrun.log`, metrics, reports and checkpoints.
No background process in the local workspace is required after launch.

Eight-GPU/8K performance is not established by the shorter-context benchmark.
Replace the earlier 3–6-hour estimate with observed throughput after warmup;
preparation reuse does not remove GPU queueing or padding cost. Evaluate long-range
quality separately from successful execution at an 8K input length.
