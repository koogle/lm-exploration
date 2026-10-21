# Fixed-teacher rollouts on Modal

Separate experiment for distillation data from **Qwen/Qwen3.5-9B**. No student
training, pretraining-plan changes, custom kernels or large generation campaign.
**Implemented and CPU-tested; GPU benchmark and profile pending budget approval.**
The dated implementation/status report (runtime inspection, prior-research
reverification, profiling decision rule) is archived on HF; see
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

The proposed campaign cap is **$5**, at most two reservations of **$2.10** each:
one H100, four CPU cores, 32 GiB host memory, 20 minutes per attempt, plus a
60-second shutdown reserve at a conservative **$6/resource-hour** ceiling.
$3.9492/GPU-hour comes from the public [Modal pricing](https://modal.com/pricing)
rate of $0.001097/second inspected 2026-10-21; workspace billing may differ.
The remaining $0.80 is a reserve for build/storage overhead; confirm that this
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
and thinking disabled are fixed across each sweep. Seeds are sent to vLLM;
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
actual usage token counts, request duration and validation. Raw records are saved
and Modal-volume committed incrementally before validation. Validated records are
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

Full 2026-10-21 status report: [`pr-36/exploration/teacher-rollouts/REPORT.md`](https://huggingface.co/datasets/jfrick/lm-exploration-results/tree/main/pr-36/exploration/teacher-rollouts)
in the private HF dataset `jfrick/lm-exploration-results`.

- **No GPU job launched; no throughput, cost, memory or bottleneck measured.**
  Every metric (tokens/s, accepted examples/hour, $/accepted, startup, peak memory,
  prefill/decode/DeltaNet breakdown) is pending an authorized H100 run.
- Runtime: vLLM 0.32.0 (`5df63a7`) chosen from source inspection (Qwen3.5
  architecture, text-only loading, per-request seeds, metrics, Torch profiling).
  SGLang 0.5.22 also supports Qwen3.5 but is kept as a later matched comparison.
  Neither choice is a performance claim.
- Teacher/tokenizer `Qwen/Qwen3.5-9B@c202236`: 32 layers (24 linear, 8 full
  attention), width 4,096, DeltaNet key/value heads 16/32, BF16.
- Prior megakernel work does not transfer directly: Luce is hard-coded to
  Qwen3.5-0.8B (16/16 heads), Mirage targets Qwen3-8B, Megakernels is
  Llama-oriented. **No custom CUDA work is justified by measurements yet**; first
  choose concurrency by pooled accepted examples per GPU second, then profile, and
  only propose a small kernel experiment if DeltaNet or launch overhead stays a
  substantial measured fraction at the best batched operating point.
- CPU validation: 12 unit tests pass; Modal definitions import; ruff passes.
  Build and GPU execution remain unverified.

## Artifacts

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
Do not delete staged artifacts until verification succeeds. No run has been
uploaded yet.
