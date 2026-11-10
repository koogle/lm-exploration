"""Shared name-based grouping for live and offline CUDA trace summaries."""

import re

# First match wins; names are CUDA kernel names as the profiler reports them.
KERNEL_CATEGORIES = (
    ("communication", r"nccl"),
    ("flex_attention", r"flex_attention|triton_tem_fused"),
    ("kda_convolution", r"kda_conv_"),
    (
        "kda_fla",
        r"intra_backward|chunk_gla|chunk_kda|kda|chunk_gated_delta|chunk_fwd|chunk_bwd|chunk_local|solve_tril|wy_|l2norm",
    ),
    ("loss_softmax_gradient", r"softmax_gradient"),
    ("matmul", r"gemm|cutlass|nvjet|sm90_xmma|cublas|matmul"),
    ("optimizer", r"multi_tensor|foreach|adam|newton|muon"),
    ("compiled_elementwise", r"triton_(poi|red|per)"),
    ("memory", r"memcpy|memset|Memcpy|Memset|copy_|fill"),
)


def kernel_category(name):
    for category, pattern in KERNEL_CATEGORIES:
        if re.search(pattern, name):
            return category
    return "other"
