# Third-party provenance and modifications

`kernel.cu`, `half_type.h` and `rope.h` derive from Luce-Org/lucebox,
commit `ad4de665f32571ad662224ee7f7b0006d8070cb3`,
`optimizations/megakernel/`, licensed Apache-2.0 (see LICENSE.luce).
Upstream: https://github.com/Luce-Org/lucebox/tree/ad4de665f32571ad662224ee7f7b0006d8070cb3/optimizations/megakernel

Modified here for Qwen3.5-9B: 32-layer dimensions; 16-key/32-value head grouping;
single-owner convolution history followed by a grid barrier; independent state
per value head; FP32 decay parameters; BF16 operation boundaries; dynamic shared
memory with occupancy queries; CUDA cooperative grid synchronization; optional
layer snapshots; and a checked ctypes interface. The upstream LM-head/argmax
kernel is removed: the full 32-layer transformer runs in one persistent kernel,
followed by a standard BF16 PyTorch/cuBLAS vocabulary projection and sampling.
Python model loading, validation and Modal integration are new code.

This is a batch-one decode implementation. Sequential decode also provides a
correctness-first prefill path; no high-throughput prefill or continuous batching
is claimed for this backend. Measurements and correctness gates are documented
separately. Retained upstream comments describing 0.8B tensor shapes are historical
and are superseded by the constants and validated tensor shapes.

The optional `USE_THUNDERKITTENS` projection path calls HazyResearch
ThunderKittens register-vector operations. `build_sweep.py` pins and verifies
commit `67845f5fa48d05343dbbc1ba1403f11061f08d2d` and retains its MIT LICENSE
in the downloaded source tree. This optional integration is new code here;
the underlying transformer and other projections remain Luce-derived.
Upstream: https://github.com/HazyResearch/ThunderKittens
