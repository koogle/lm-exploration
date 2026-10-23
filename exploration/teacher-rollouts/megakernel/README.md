# Qwen3.5-9B persistent decode kernel

This is a real CUDA implementation adapted from the Apache-2.0 Luce 0.8B kernel;
see [NOTICE.md](NOTICE.md). It executes the text transformer's **32 layers in one
cooperative GPU launch**, including DeltaNet, full attention, normalization and
MLPs. A conventional BF16 cuBLAS vocabulary projection follows it. This is not a
single launch for the entire sampling loop.

The initial implementation targets one sequence on one H100, BF16 weights and
FP32 recurrent state. It is separate from the continuously batched vLLM runner.
There is no claim that batch-one speed improves aggregate rollout throughput.
Sequential token processing supplies prefill for correctness and short probes;
optimized prefill and batching are not implemented for this backend.
The probe uses a 1,024-token cache and greedy generation capped at 128 output
tokens; output counts exclude terminal EOS. The parent's sampling/concurrency
configuration describes the vLLM sweep. Only its teacher pin, authorization,
resource limits and accounting settings are shared with this fixed smoke probe.

## Changes beyond dimensions

- Q/K retain 16 heads; V, decay, beta and recurrent state have 32 heads. Value
  head `h` reads Q/K head `h // 2` while maintaining its own state.
- Every convolution channel has exactly one writer before a grid barrier. Two
  value heads must not shift their shared Q/K convolution history twice.
- State storage uses `[value, key]` internally; validation explicitly transposes
  it to the reference `[key, value]` layout.
- Larger MLP scratch uses dynamically allocated shared memory and an occupancy
  query. CUDA cooperative launch/grid synchronization replaces spin barriers;
  launch fails if the complete grid cannot be resident.
- Decay parameters remain FP32; BF16 rounding is preserved at projection,
  residual, gating and RoPE boundaries. Numerical differences from the reference
  are measured rather than assumed away.
- All device storage is retained by the Python decoder. Token IDs and cache
  bounds are checked, weights use a pinned revision, and architecture mismatches
  fail before launching.

## Run

From the parent `exploration/teacher-rollouts/` directory, using the existing
budget authorization and ledger:

```sh
uv run --with-requirements requirements.lock python launch.py \
  --backend megakernel --run-id kernel-9b-001 --attempt kernel-001 \
  --config artifacts/authorized-kernel.json --launch
```

The CUDA 12.8.1 development image is pinned by digest in `modal_app.py`.
`megakernel/requirements.lock` pins PyTorch 2.10.0, Transformers 5.4.0, and all
Python dependencies. NVCC compilation occurs in the image build before GPU
allocation. The same remote supervisor, deadline, runtime limits and spending
ledger used by the baseline apply. Managed cloud clients add system CA trust
using `truststore`; TLS verification stays enabled.

`probe.py` loads only the pinned teacher's language weights and untied head,
compares against Transformers eager inference, and saves results incrementally.
It checks every layer, full-attention key cache, every recurrent state and complete
vocabulary logits on teacher-forced short sequences. Preregistered smoke gates:
logit normalized RMSE <= 0.04, cosine >= 0.999, top-1 agreement >= 95%, and
recurrent-state normalized RMSE <= 0.08. These tolerances do not establish exact
arithmetic equivalence, long-context stability or production readiness.

Only after passing those gates does it test reset reproducibility and run matched
greedy, 128-output-token synthetic fixture generations against Transformers eager.
Both backends use sequential prefill. Their comparison is **not a vLLM comparison**.
Actual outputs, finish reasons, validation, token counts and costs are retained.
An unsuccessful correctness gate stops generation benchmarking and is preserved
as a negative result. No student training is launched.

See the parent README findings and its linked HF results folder for measurement status.
Do not interpret the existence of this source or a successful compilation as a
successful GPU correctness result.

Measured H100 smoke result: all 25 checked next-token predictions agree, all six
greedy fixture output sequences match, and reset reproducibility passes. Generation
including sequential prefill was 4.58× faster than Transformers eager fallback
(63.51 versus 13.87 output tokens/s). See the parent findings for cost, provenance
and limits; optimized batched performance is a separate question.

## Prepared launch-geometry and ThunderKittens sweep

`--backend kernel-sweep` builds eight libraries: manual and ThunderKittens
projection variants, each with 256/512 threads and 66/132 requested cooperative
blocks. Occupancy can reduce the actual grid. Two rounds use forward and reverse
order, with identical six synthetic prompts, greedy sampling and a 128-token
output cap. Each trial runs the numerical/state/reset checks above before timing;
only two complete passing trials with exact reference fixture token sequences
enter the accepted-examples/hour ranking. All failed checks and individual trial
measurements remain in the report. Two short trials are screening evidence, not
statistically reliable production throughput estimates.

The ThunderKittens variant uses its register-vector multiply/add/reduction
operations in QKV and gating projections. It is a limited change to the adapted
Luce kernel, not a tensor-core implementation or a full ThunderKittens rewrite.
The measured sweep below qualifies the 512-thread variants on the short probe;
the 256-thread variants fail the exact-completion gate. No speedup is established. Both
variants use C++20 and the same optimization flags; dependency revision and
archive SHA-256 are pinned in `build_sweep.py`. The downloaded MIT license is
retained alongside the headers in the build image. See [NOTICE.md](NOTICE.md).

The teacher loads once per sweep. Each trial persists outputs immediately and
reports its own generation/verification times. The parent report includes total
pipeline and initial teacher-loading times. Existing attempt IDs are rejected;
an interrupted kernel sweep needs a fresh attempt and budget reservation, and
its partial results must be retained. This does not change resumable baseline
dataset generation. Peak allocation is PyTorch memory, not total device usage.

```sh
uv run --with-requirements requirements.lock python launch.py \
  --backend kernel-sweep --run-id kernel-sweep-001 --attempt sweep-001 \
  --config artifacts/authorized-sweep.json --launch
```

The user authorized a $100 total campaign cap, including prior reservations.
Use the ignored authorized configuration and the existing ledger; preserve the
1,200-second GPU job cap and independent supervisor. Start with one kernel sweep
and one vLLM concurrency sweep; launch geometry is not request concurrency.
The previous $5 campaign remains reserved. No automatic retries or student
training are authorized by this experiment.

CPU validation: 17 tests and Ruff pass; all eight variants compile locally.
Local CUDA compilation is a build diagnostic only; the workspace uses GCC 14 and newer glibc, requiring temporary
`noexcept` compatibility declarations in the downloaded CUDA 12.8 math headers.
Those local compiler changes are not part of the Modal image or kernel source.
All eight subsequently built unmodified in the pinned Ubuntu 22.04 Modal image
and completed the H100 sweep.


## Measured launch sweep (2026-10-21)

Eight builds, two rounds in forward/reverse order, one H100 per job, pinned BF16
teacher, six synthetic prompts, greedy generation capped at 128 output tokens.
All builds passed the numerical/state/reset gates and 25/25 teacher-forced top-1
checks in both rounds. These are repeated checks of the same 25 positions, not
400 independent samples. Worst logit NRMSE across trials was 1.916%, minimum
cosine 0.999817, and worst recurrent-state NRMSE 2.927%.

| Projection | Threads / requested blocks | Mean output tokens/s | Exact fixture tokens, both rounds |
| --- | --- | ---: | --- |
| Manual | 256 / 66 | 42.83 | pass |
| Manual | 256 / 132 | 56.54 | pass |
| Manual | 512 / 66 | 53.49 | pass |
| Manual | 512 / 132 | 62.37 | pass |
| ThunderKittens | 256 / 66 | excluded | fail |
| ThunderKittens | 256 / 132 | excluded | fail |
| ThunderKittens | 512 / 66 | 53.42 | pass |
| ThunderKittens | 512 / 132 | 62.37 | pass |

The two excluded builds produced different wording for the cache-hit explanation
while retaining the correct answer, 80%, and passing all automatic task checks.
Their raw measurements and rejection from the strict comparison are preserved.
Do not describe them as exact-reference-qualified builds.

Keep the original manual 512/132 configuration. Its mean generation time was
3.303 seconds for six examples (206 non-EOS output tokens), including sequential
prefill: about 6,540 automatically accepted fixtures/hour and estimated steady
GPU cost $0.000604/accepted fixture. The best ThunderKittens configuration differs
by less than 0.01%; two short rounds do not support a speedup claim. All fixture
outputs passed automatic checks, with zero truncation; technical answers retain
the human-review requirement. Prompt lengths were 45–58 (mean 52.17), outputs
10–69 (mean 34.33). Task verification took about 0.65 ms per six-example trial.

Teacher loading from the warm cache took 15.06 seconds. Whole sweep pipeline:
716.48 seconds; worker: 721.55 seconds; client including build/queue: 838.34 seconds.
Estimated worker GPU compute was $0.7915; actual billing remains unknown. Peak
PyTorch allocation was 18.21 GB, not total device memory. These times include all
16 correctness probes and eager-reference comparisons; they are not steady
production-generation costs. The app is stopped with zero tasks.

Detailed reports, outputs, compiler commands, runtime versions, source hashes,
launch logs and final app state are in HF dataset `jfrick/lm-exploration-results`,
folder `pr-36/exploration/teacher-rollouts/kernel-sweep-20261021-v1`.

## Matched durable-output pipeline

`--backend kernel-pipeline` compiles the selected manual 512/132 build with the
sweep's C++20 flags, runs the full qualification probe, then measures two rounds
of 48 requests at concurrency one through the same record, checkpoint and
acceptance implementation as vLLM. The fixed kernel uses greedy generation with
128 output tokens and a 1,024-token cache; other sampling/limit settings fail
closed. Its request adapter records the actual CUDA runtime label and non-EOS
token-count convention. vLLM's usage counts include terminal EOS. Compare accepted
examples/hour directly; keep the token-count convention explicit.

The measured pipeline includes remote output checkpoints and local writes; the
kernel-only sweep table above excludes them. vLLM also incurs local HTTP overhead,
while the kernel adapter calls the decoder directly. Model startup, qualification,
steady generation, verification, total pipeline time and whole-device memory
samples are reported separately. No batched custom decoder is claimed.

```sh
uv run --with-requirements requirements.lock python launch.py \
  --backend kernel-pipeline --run-id kernel-pipeline-001 --attempt pipeline-001 \
  --config artifacts/authorized-batched.json --launch
```
