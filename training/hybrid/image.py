"""Modal image for the hybrid family: the torch version locked in uv.lock.

Construct from training/ so local source paths match the installed packages. The
CUDA 13.0 devel base (matching torch 2.14.1+cu130) provides nvcc for the family's
CUDA kernels, prebuilt here into /opt/hybrid_kernels/<kernel>/.
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

KERNELS = ("softmax_gradient", "kda_convolution", "decode")


def build_kernel(kernel):
    """Prebuild with the name, flags and directory hybrid/cuda_extension.py loads."""
    folder = f"/opt/hybrid_kernels/{kernel}"
    return (
        'python -c "from torch.utils.cpp_extension import load; '
        f"load(name='hybrid_{kernel}', sources=['{folder}/{kernel}.cu'], "
        f"build_directory='{folder}', extra_cuda_cflags=['-O3'])\""
    )


image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .pip_install(*PACKAGES)
    .env({"PYTHONPATH": "/project", "TORCH_CUDA_ARCH_LIST": "9.0"})
)
for kernel in KERNELS:
    image = image.add_local_file(
        f"hybrid/kernels/{kernel}.cu", f"/opt/hybrid_kernels/{kernel}/{kernel}.cu", copy=True
    ).run_commands(build_kernel(kernel))
image = add_project(image)

# One function call per launch: no retries, and the call refuses to start once 24
# hours have passed since launch (queue time included).
LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
