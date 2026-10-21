# Fixed-teacher rollouts on Modal

Separate experiment for distillation data from **Qwen/Qwen3.5-9B**. No student
training, pretraining-plan changes or large generation campaign. Includes a
[Qwen3.5-9B CUDA megakernel](megakernel/) and a separate continuously batched vLLM
baseline. **The CUDA port compiles and passes its H100 correctness smoke gate.**
The initial pre-GPU runtime/research inspection report is archived on HF; see
[Findings so far](#findings-so-far).

Run all commands from `exploration/teacher-rollouts/`:

```sh
uv venv --python 3.12
uv pip sync requirements.lock
.venv/bin/python -m unittest discover -s tests -v
# Manifest dry run; does not import Modal, build images or call the GPU function.
.venv/bin/python launch.py --run-id baseline-001 --attempt attempt-001
```

After an explicit budget is authorized, copy `config.json` into ignored
`artifacts/authorized.json`, set `budget_authorized` and a concrete
`authorization_reference`, and recheck current Modal prices against the rate
ceiling. Do not change these fields merely to make the launch work.

```sh
.venv/bin/python launch.py --run-id baseline-001 --attempt attempt-001 \
  --config artifacts/authorized.json --launch
```

The user has authorized a **$100 total campaign cap**, including prior
reservations. Keep individual attempts bounded at **$2.10** compute reservation:
one H100, four CPU cores, 32 GiB host memory, 20 minutes per attempt, plus a
60-second shutdown reserve at a conservative **$6/resource-hour** ceiling.
$3.9492/GPU-hour comes from the public [Modal pricing](https://modal.com/pricing)
rate of $0.001097/second inspected 2026-10-21; workspace billing may differ.
Remaining campaign headroom covers build/storage overhead; confirm that this
covers the actual Modal workspace prices before approval/launch. Runtime and
reservation controls cap configured compute exposure, not the provider invoice.
There is no API here that imposes an account-wide billing cap. If the verified
rates/overhead cannot fit the approved cap, shorten the job or do not launch.

Local `.budget/ledger.json` reserves the full attempt before importing Modal or building images; reservations
are never refunded automatically, including failed launches. Keep this ledger
between attempts and use one launch host. This is not a distributed account-wide
ledger. Do not launch from another checkout or bypass the entrypoint. No automatic
retries; `max_containers=1`; worker timeout is at most 1,200 seconds, with a shorter
subprocess deadline that kills the entire server process group.
A separate 0.125-core/256-MiB remote supervisor enforces an absolute deadline
(launch + configured runtime + 60 seconds), including image setup and queue time.
It records both call IDs and the deadline in a dedicated Modal Dict and worker
metadata, bounds the child timeout by remaining time, refuses late starts, and
explicitly cancels unfinished GPU calls via the Modal API. Its own server timeout
is 1,320 seconds with no retries. Cancellation is independent of the local client.
The supervisor adds a small CPU cost covered by the resource-rate ceiling. The model and
runtime pins cannot be overridden by a launch config. Credentials use normal
Modal configuration; public teacher weights require no forwarded secret.

## Protocol and records

The image digest pins vLLM 0.32.0 and its CUDA/PyTorch dependencies; the SDK overlay
and local launcher dependencies are pinned in `requirements.lock`. Runtime package
versions, server argv and `nvidia-smi` output are also archived. BF16 weights,
text-only model loading, one GPU, no prefix caching, no speculative decoding,
4,096-token context, max 512 output tokens, temperature 0.6/top-p 0.95/top-k 20,
and thinking disabled are the default recipe. The measured follow-up uses
greedy sampling, top-p 1 and a 128-token cap to match the custom kernel; all
settings remain fixed within each sweep. Seeds are sent to vLLM;
bitwise equality across scheduler/batch shapes is **not** promised.

Six explicitly synthetic fixtures cover code, technical answers and JSON. Eight
seeded repetitions yield 48 requests per trial. Two rounds use concurrency
1/4/16/32 in ascending then descending order, followed by a separate profiling
replay of the best pooled accepted-example rate. This gives at most 432 requests
plus one 16-token warm-up. It is intentionally a tiny fixture suite, not a
representative corpus, held-out evaluation or a production dataset. Repeated
prompt/seed pairs are deliberate matched benchmark exposures, not unique training
examples. Do not concatenate these trials as a deduplicated distillation dataset.

Code acceptance requires a single `solve` function in a small expression grammar
and correct outputs on all fixed cases. The interpreter never executes generated
Python. JSON tasks require exact keys, values and types and reject duplicate keys
and markdown fences. Technical tasks require the exact answer plus bounded prose
containing task-specific terms; these checks are weak proxies, so every technical
acceptance still requires human review before training. Empty outputs, non-stop
finish reasons and truncation are rejected. Request errors remain separate from
completed examples. Every rejected output retains reasons.

Each record contains prompt ID/text, seed and full sampling request, teacher and
tokenizer revisions, runtime, manifest hash, output, raw response, finish reason,
actual usage token counts, request duration and validation. Raw records are fsynced locally immediately and Modal-volume committed before
validation. `checkpoint_every=1` preserves the original per-example protocol.
The batched experiment uses 16 records per remote checkpoint, with a 5-second
interval checked on each request completion, plus a mandatory final partial-batch
flush. This is not a background five-second durability guarantee: an in-flight
request can delay the time-triggered checkpoint until it completes. Interrupted
uncommitted outputs can be recomputed on resume under their stable IDs. Count,
interval and actual raw-checkpoint count are recorded per trial. Validated records are
separate. CPU verification is outside generation timing; its duration and output
persistence duration are reported separately. Generation throughput includes HTTP
and durable-output overhead, so it measures this generator rather than isolated
engine kernel speed. Startup-to-ready includes downloads, loading and compilation;
weight-loading time is extracted from the server log when available (otherwise
null). Total worker/pipeline times, 200 ms sampled device memory, raw Prometheus
snapshots and a separate Torch profile are retained. Sampling can miss brief
memory peaks. GPU cost includes GPU residency during CPU verification; CPU dollar
costs use the reservation ceiling, not a fabricated bill. Actual billed cost is
null until reconciled against Modal billing.

Use the same run/config/source commit and a **new attempt ID** to resume. Stable
example IDs skip persisted records; a changed manifest or corrupt/conflicting
record fails closed. A request interrupted before persistence can be recomputed,
but produces only one stored ID. Errors already persisted are not retried. Partly
resumed trials have null throughput/cost rates; rerun with a fresh run ID for a
valid timing comparison. Completed trials retain their original measurements.
One writer per run is required. A forced timeout may leave no final report or
profile: preserve its raw records, server log and worker timing as negative evidence.

## Findings so far

Measured on Modal H100, 2026-10-21. The reproducible measured recipe is
[`config.greedy-batched.json`](config.greedy-batched.json): pinned BF16
Qwen3.5-9B, vLLM 0.32.0 (PyTorch 2.13.0+cu130, Transformers 5.19.0),
greedy sampling, top-p 1, 128 output tokens, six synthetic
prompts × eight repetitions × two rounds. Both configs require explicit budget
authorization before launch. The default checkpoint policy is now 16 records,
with the interval and final-flush behavior described above.

Rates below pool counts over elapsed generation time across both rounds;
completed and automatically accepted rates are equal (384/384 passed, zero
truncations or request errors). Generation timing includes raw output persistence.

| Concurrency | Output tokens/s | Accepted fixtures/hour | Estimated steady GPU USD/accepted |
| ---: | ---: | ---: | ---: |
| 1 | 84.2 | 8,575 | 0.000461 |
| 4 | 170.3 | 17,399 | 0.000227 |
| 16 | 234.1 | 23,847 | 0.000166 |
| 32 | 291.1 | 29,683 | 0.000133 |

Concurrency 32 was best observed: 28,867 and 30,546 accepted fixtures/hour in the
two rounds. This is not a general optimum for other workloads. Prompts contained
45–58 tokens (mean 52.17); outputs 11–70 (trial means 35.21–35.33, including EOS).
Greedy outputs can differ slightly across batching shapes. These repeated
fixtures are not distinct training examples; technical acceptances still need
human review. Do not treat 100% fixture acceptance as general teacher quality.

**Measured pipeline improvement:** the original per-example remote-commit trial
at concurrency one completed 48 requests in 134.67 seconds, including 121.28
seconds blocked on persistence (90.1%). Verification took 2.25 ms. After bounded
checkpoint batching, concurrency-one throughput was 6.68× higher pooled across
two rounds, with estimated steady GPU cost reduced from $0.003078 to $0.000461
per accepted fixture. Combining batching with concurrency 32 reached 23.1× the
initial pipeline throughput and 95.7% lower steady GPU cost. The costly original
attempt was stopped after preserving its first complete trial and 22 partial
next-trial records; it is not presented as a completed concurrency sweep.

**Megakernel comparison:** the [eight-build launch sweep](megakernel/README.md)
completed both rounds. Keep the Luce-derived manual 512-thread/132-block build;
the best ThunderKittens register-vector variant was within 0.01%, not a meaningful
speedup. Both 256-thread ThunderKittens builds passed numerical/task checks but
failed exact reference-completion matching and are excluded from qualification.
The kernel-only rate of 62.37 non-EOS output tokens/s excludes output persistence.
A separate matched persistence run qualified the same manual build, then produced
96/96 accepted fixtures over two rounds at concurrency one: **3,869/hour**, 36.90
non-EOS output tokens/s, estimated steady GPU cost **$0.001021/accepted**. Both
pipelines use the same checkpoint policy; vLLM includes HTTP overhead and the
custom adapter calls CUDA directly. No batched custom decoder is implemented.

**Startup and memory:** vLLM took 114.60 seconds to become ready, with 7.79 seconds
loading weights within 12.81 seconds for total model loading/allocation;
these latter two values come from the preserved server log. The original report
parser missed the new weight-loading log phrase; `loading-times.json` preserves
the corrected extraction and the parser is fixed for future runs. Total pipeline
including the profiling replay was 261.76 seconds (worker 263.77, client including
build/queue 312.57), with estimated pipeline GPU compute $0.2872 and 68,691 MiB
peak sampled whole-device memory. The kernel persistence run loaded in 14.43
seconds, took 164.98 seconds including qualification (worker 172.67), and peaked
at 18,185 MiB; its estimated pipeline GPU compute was $0.1810. vLLM intentionally
reserves KV-cache memory at 85% utilization. Actual billed costs remain unknown.

**Profile and recommendation:** the concurrency-32 Torch trace contains 1.218
seconds of observed GPU activity in an 8.666-second device-event span (14.1%).
Named `nvjet` GEMM kernels account for 83.6% of summed GPU kernel time; named
DeltaNet kernels 5.9%, with causal convolution another 1.0%. These are summed
kernel durations, not wall-clock fractions. Mean per-request engine timings at
concurrency 32 were 117 ms prefill, 376 ms decode and 36 microseconds queue wait;
concurrent request times overlap and cannot be added into a wall-time breakdown.
Post-hoc verification took 4.20 ms for those 96 examples. The client was blocked
on persistence for 10.69 of 11.64 unprofiled generation seconds; GPU work can
overlap that wait. Focus next on asynchronous/CPU-side output persistence with
bounded backpressure, then remeasure. These measurements do not justify further
DeltaNet CUDA work or a full custom megakernel for aggregate rollout throughput.
Weight-bandwidth saturation was not measured and cannot be inferred from kernel
names. Retain the working adapted kernel as an experimental batch-one backend.

Limitations: short synthetic prompts, two timing rounds, warm weight caches,
weak automatic technical-answer checks, and variable remote-checkpoint latency.
The two pipeline jobs overlapped on separate H100s and shared a Modal volume;
checkpoint contention may affect absolute rates. Profiling perturbs throughput
(the separate profiling trial is excluded from ranking). Long-context correctness
and production qualification remain outstanding.

Detailed artifacts are byte-verified in the private HF dataset
[`jfrick/lm-exploration-results`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-36/exploration/teacher-rollouts),
under `pr-36/exploration/teacher-rollouts/`:

- `kernel-sweep-20261021-v1`: all launch settings, correctness, exact-output gates.
- `kernel-pipeline-20261021-v1`: matched custom-kernel persistence measurements.
- `baseline-batched-20261021-v6`: full vLLM sweep, metrics, trace and profile summary.
- `baseline-greedy-20261021-v5`: original persistence bottleneck and partial trial.
- `baseline-greedy-20261021-v3` and `baseline-greedy-20261021-v4`: pre-GPU setup failures; fixed the Python
  alias and Modal source-copy ordering without changing the pinned runtime.
- Earlier smoke/research and failed image-import evidence remain in `REPORT.md`,
  `kernel-9b-20261022-v3`, `baseline-20261022-v1` and `baseline-20261022-v2`.

Source inspection also verified SGLang 0.5.22 support; it remains an unmeasured
alternative. Luce's original kernel targets 0.8B; Mirage's cited example targets
Qwen3-8B. Neither establishes support or performance for this 9B port.
All apps launched here are stopped with zero tasks. The ledger retains **$17.60**
in conservative reservations against the **$100** authorized cap; reservations
are not billed costs. Validation: **21 CPU tests and Ruff pass**, all eight CUDA
variants compile in the pinned image, and GPU qualifications/results are scoped
as described above. The PR remains draft and unmerged.

## Artifacts

Summarize an explicit downloaded Torch trace with:

```sh
python profile_summary.py --trace artifacts/run/trace.json.gz \
  --output artifacts/run/profile-summary.json
```

The helper reports kernel/copy durations and the union of observed device
intervals. Summed durations can overlap; they are not wall-clock fractions.
Kernel names alone cannot establish weight-bandwidth saturation or attribute
idle gaps to scheduling. Keep verification timings and prefill/decode metrics
separate, and retain the raw trace alongside the summary.

The dedicated `lm-exploration-teacher-rollouts` Modal volume is a staging store.
Git contains source, config, intentional fixtures and this README only; run
outputs go to the HF dataset `jfrick/lm-exploration-results` under
`pr-36/exploration/teacher-rollouts/<run-id>/`. Download the explicit run into
ignored `artifacts/`:

```sh
.venv/bin/modal volume get lm-exploration-teacher-rollouts baseline-001 artifacts/baseline-001
uv run --with 'huggingface-hub==0.36.0' python archive.py \
  --run-dir artifacts/baseline-001 --repo-id jfrick/lm-exploration-results \
  --prefix pr-36/exploration/teacher-rollouts/baseline-001 \
  --index artifacts/baseline-001-index.json
```

Copy the local launch/supervision JSONs into the downloaded run before archiving.
The dedicated `lm-exploration-teacher-rollout-deadlines` Modal Dict also retains
supervision state when the local client disconnects. Archive raw and validated
records, errors, runtime metadata, metrics, logs and traces, including failed runs
and profiles. The exporter verifies each downloaded file at the immutable upload
commit and writes a local checksum receipt to the ignored `artifacts/` dir; do not
commit it. Update the findings above with concise results (negative ones too).
Do not delete staged artifacts until verification succeeds. Run publication status is recorded in the findings above.
