"""Modal image for the hybrid family: the torch version locked in uv.lock.

Construct from training/ so local source paths match the installed packages. The
CUDA 13.0 devel base (matching torch 2.14.1+cu130) provides nvcc for the
softmax-gradient kernel, prebuilt here into /opt/hybrid_softmax.
"""

import modal

from workflows.images import add_project

CUDA_BASE = "nvidia/cuda:13.0.0-devel-ubuntu22.04"
PACKAGES = (
    "torch==2.14.1",
    "numpy",
    "einops",
    "ninja",
    "fla-core==0.6.0",
    "packaging",
    "tokenizers==0.22.2",
    "huggingface-hub==1.33.0",
    "safetensors",
)

BUILD_SOFTMAX = (
    'python -c "from torch.utils.cpp_extension import load; '
    "load(name='hybrid_softmax_gradient', "
    "sources=['/opt/hybrid_softmax/softmax_gradient.cu'], "
    "build_directory='/opt/hybrid_softmax', extra_cuda_cflags=['-O3'])\""
)

image = add_project(
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .pip_install(*PACKAGES)
    .env({"PYTHONPATH": "/project", "TORCH_CUDA_ARCH_LIST": "9.0"})
    .add_local_file(
        "hybrid/kernels/softmax_gradient.cu",
        "/opt/hybrid_softmax/softmax_gradient.cu",
        copy=True,
    )
    .run_commands(BUILD_SOFTMAX)
)

# One function call per launch: no retries, and the call refuses to start once 24
# hours have passed since launch (queue time included).
LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
