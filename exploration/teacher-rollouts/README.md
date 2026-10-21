# Fixed-teacher rollouts on Modal

Separate experiment for distillation data from **Qwen/Qwen3.5-9B**. No student
training, pretraining-plan changes, custom kernels or large generation campaign.
**Implemented and CPU-tested; GPU benchmark and profile pending budget approval.**
See [the baseline report](REPORT.md) for inspected runtime support and limitations.

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
subprocess deadline that kills the entire server process group. The model and
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

## Artifacts

The dedicated `lm-exploration-teacher-rollouts` Modal volume is a staging store.
Git contains source, config, intentional fixtures, this concise report and verified
artifact indexes only. Download the explicit run into ignored `artifacts/`:

```sh
.venv/bin/modal volume get lm-exploration-teacher-rollouts baseline-001 artifacts/baseline-001
# Use the existing HF dataset repository selected for this experiment.
uv run --with 'huggingface-hub==0.36.0' python archive.py \
  --run-dir artifacts/baseline-001 --repo-id OWNER/EXISTING-DATASET \
  --prefix teacher-rollouts/baseline-001 --index baseline-001-index.json
```

Archive raw and validated records, errors, runtime metadata, metrics, logs and
traces. The exporter verifies each downloaded file at the immutable upload commit
and writes repository ID/type, revision, paths and SHA256 checksums. Commit only
that index and concise findings. Do not delete staged artifacts until verification
succeeds. No dataset repository is guessed or created; none has been uploaded yet.
