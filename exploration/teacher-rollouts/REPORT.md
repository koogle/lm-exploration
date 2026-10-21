# Qwen3.5-9B teacher baseline: implementation and measurement status

2026-10-21. **No GPU job launched. No measured baseline, best concurrency or
bottleneck identified yet.** CPU tests and source inspection establish harness
behavior and advertised runtime support, not model correctness or H100 performance.
The repository's prior allocations do not authorize this separate experiment
(`training/README.md`, Token and compute planning). The default launch is closed
until an explicit budget is recorded. Current root AGENTS.md additionally requires
a remote launch-deadline supervisor; this implementation includes one with
persisted call IDs, remaining-time checks and explicit child cancellation. Requested next step: a $5 campaign, one
H100, at most two 20-minute attempts including startup; no student training.

## Provenance and runtime decision

Selected **vLLM 0.32.0**, release source
[`5df63a7`](https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be),
with its official amd64 image pinned by digest in `config.json`. Its
[`qwen3_5.py`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/model_executor/models/qwen3_5.py)
supports the Qwen3.5 conditional-generation architecture and DeltaNet key/value
head counts. Text-only loading is exposed by `language_model_only`; the OpenAI
chat endpoint accepts the template's `enable_thinking=False` flag. Continuous
batching, per-request seeds, usage accounting, Prometheus metrics and Torch
profiling make this a suitable baseline candidate. This is a source-based choice,
not a claim that vLLM outperforms SGLang. The container build and GPU startup still
need an actual smoke run.

Also inspected **SGLang 0.5.22**, release
[`baa9bef`](https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d).
Its [`qwen3_5.py`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/models/qwen3_5.py)
contains dense Qwen3.5, Gated DeltaNet and text-only paths. Its test tree contains
9B tests on NPU/XPU; those are not proof of this H100 workload. Keep SGLang as a
future matched runtime comparison rather than spending the initial budget across
two unmeasured setups.

Teacher **and tokenizer** pin:
[`Qwen/Qwen3.5-9B@c202236`](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a).
The pinned config confirms 32 layers, width 4,096, MLP width 12,288, 24 linear and
eight full-attention layers, and DeltaNet key/value heads **16/32**. Weights are
BF16; no quantization or teacher updates. Runtime package inventory, CUDA/driver
information, source hashes, Git commit, prompts and sampling configuration are
saved per run. SDK dependencies are pinned separately from the immutable runtime.

## Measurement table

| Metric | Current evidence |
| --- | --- |
| Generated tokens/s; completed examples/hour | Pending authorized H100 run |
| Accepted examples/hour; estimated GPU $/accepted | Pending; null if no acceptance |
| Actual input/output lengths; truncation; acceptance | Pending server usage and validators |
| Startup/model load; steady state; total pipeline | Instrumented, unmeasured |
| Peak GPU memory | 200 ms device sampling implemented; not an exact allocator peak |
| Filtering/verification overhead | Separate post-generation CPU timing implemented |
| Actual billed cost | Unknown; reconcile provider billing separately |
| Prefill/decode/DeltaNet/scheduling breakdown | Metrics and Torch profile collection implemented; analysis pending |

The public [Modal price page](https://modal.com/pricing), inspected 2026-10-21,
lists H100 SXM5 at $0.001097/s = $3.9492/h. Cost estimates use that rate, explicitly
exclude billing reconciliation, and include separate steady-state and whole-job
GPU-residency estimates. The launch reserve uses $6/h for bounded GPU+CPU+memory,
plus a campaign reserve for build/storage. Neither a price estimate nor a local
ledger is an account-wide provider billing cap.

The protocol uses matched prompts, seeds, sampling and output limits across
concurrency 1/4/16/32, two order-reversed rounds, a separate warm-up and a separate
profile replay. Raw throughput counts all completed responses; accepted throughput
counts fixture passes only. Persisted errors/rejections remain available. HTTP,
scheduling and synchronous durable-output overhead are included in generator
throughput; verification is timed separately. A partial resume cannot yield a
valid rate and is excluded from ranking. The profile does not enter the steady
benchmark rate, but its GPU cost remains part of total pipeline cost.

Limitations: six hand-authored synthetic prompts, eight repeated seeds, short
outputs, tiny fixed code tests, weak technical prose checks, no production-corpus
length distribution, and no quality/generalization claim. Repeated benchmark
exposures must not be treated as independent unique training examples. Seeds do
not guarantee bitwise determinism across batch shapes. The short run may exhaust
its cap during model download/compilation; keep that negative result and request
any further allocation explicitly. Build and GPU execution remain unverified.

## Prior research reverified

- Repository starting point: [`training/references/megakernels-2025.md`](../../training/references/megakernels-2025.md).
- [HazyResearch/Megakernels at `7309cec`](https://github.com/HazyResearch/Megakernels/tree/7309cec801537b61fea3b50d7dfe454a6cde578e):
  current main remains the Llama-oriented low-latency release. The separately
  documented throughput release is not this model or benchmark.
- [ThunderKittens at `67845f5`](https://github.com/HazyResearch/ThunderKittens/tree/67845f5fa48d05343dbbc1ba1403f11061f08d2d):
  useful kernel building blocks; no source inspected here establishes a drop-in
  Qwen3.5-9B rollout engine or a speedup over batched vLLM.
- [Luce at `ad4de66`](https://github.com/Luce-Org/lucebox/tree/ad4de665f32571ad662224ee7f7b0006d8070cb3/optimizations/megakernel):
  current `model.py` and `kernel.cu` still hard-code 24 layers and width 1,024;
  README explicitly scopes it to Qwen3.5-0.8B's 18 linear/six full-attention layers.
  Its 16/16 DeltaNet heads differ from this teacher's 16/32. A port must redesign
  state layout, projections/head grouping, scheduling and resource usage, then
  verify recurrent state and logits; dimension substitutions are insufficient.
- [Mirage at `ffe38df`](https://github.com/mirage-project/mirage/blob/ffe38dff251017cdddeb1891cc808b43e7f51f25/demo/qwen3/demo.py):
  demo defaults to **Qwen/Qwen3-8B**, not Qwen3.5-9B. Do not transfer its performance
  claims to the hybrid DeltaNet teacher.

## Recommendation and profile decision rule

**No custom CUDA work is justified by measurements yet.** First complete the
bounded baseline and choose concurrency using pooled accepted examples per GPU
second, checking both rounds and failures. Keep end-to-end cost and throughput
alongside raw token throughput. Do not infer aggregate throughput gains from
batch-one latency numbers.

Inspect `*-before.prom`/`*-after.prom` deltas for prefill/decode durations, queue
waits and request/token distributions. Open the best-concurrency Torch trace in
Perfetto/PyTorch tooling; inspect CPU scheduling gaps and GPU kernel timelines.
Separate prefill from decode using engine annotations and correlate DeltaNet
kernels with their surrounding projections. Repeated weight GEMMs alone do not
prove weight-bandwidth domination; that needs hardware counters (a separately
bounded Nsight Compute experiment). Avoid summing overlapping kernel durations
as elapsed-time percentages. Compare verification/persistence time with generation
before proposing GPU changes. Torch profiling changes execution overhead; use its
replay for attribution, not the headline throughput.

If scheduling or persistence dominates, improve the host pipeline first. If
prefill dominates, investigate length buckets and supported chunked-prefill tuning.
If decode is memory-bound, measure higher useful concurrency before fusion. Only
if DeltaNet or launch/synchronization overhead remains a substantial measured
fraction at the best batched operating point should a small kernel experiment be
proposed, with BF16 correctness, recurrent-state checks and full-pipeline accepted
throughput as gates. A full megakernel remains subsequent work.

## Validation and artifact status

**12 passing CPU unit tests** cover semantic code failures and prohibited code, strict structured
output checks, technical rubric limits, corruption/conflict refusal, partial and
complete resume, cumulative budget exhaustion, explicit child cancellation on deadline/queue timeout,
and zero-acceptance metric handling.
Modal 1.6.1 definitions import locally. Ruff checks and formatting pass; the dry-run manifest path also passes without
dispatching a worker. None of these is a GPU/runtime compatibility test.

No generated dataset or detailed GPU result exists yet; therefore no HF artifact
index is claimed. The runner stages outputs on its dedicated Modal volume and
`archive.py` uploads an explicit run to a chosen existing HF dataset, verifies
bytes at the immutable commit and creates the required checksum index. Preserve
failed runs and profiles there too. Git contains only source, intentional fixtures
and this concise status report.
