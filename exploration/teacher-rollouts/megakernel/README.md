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
It checks every layer, full-attention KV cache, every recurrent state and complete
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

See the parent report and immutable artifact index for actual measurement status.
Do not interpret the existence of this source or a successful compilation as a
successful GPU correctness result.
