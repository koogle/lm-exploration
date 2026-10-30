"""Shared Modal image for subword training, qualification and benchmarks.

Construct from training/ so local source paths match the installed packages.
The CUDA devel base provides nvcc for the cross-entropy kernel
(trainer/kernels/stable_ce.cu), prebuilt here into /opt/stable_ce.
"""

import modal

from workflows.images import add_project

CUDA_BASE = "nvidia/cuda:13.0.0-devel-ubuntu22.04"  # Matches torch 2.14.1+cu130 (uv.lock).
BASE_PACKAGES = ("torch==2.14.1", "numpy", "pydantic>=2.11,<3", "tokenizers==0.22.2", "ninja")
BUILD_KERNEL = (
    'python -c "from torch.utils.cpp_extension import load; '
    "load(name='stable_ce', sources=['/opt/stable_ce/stable_ce.cu'], "
    "build_directory='/opt/stable_ce', extra_cuda_cflags=['-O3'])\""
)


def project_image(*packages, env=None):
    """Extra pip packages and environment must precede local sources in a Modal image."""
    return add_project(
        modal.Image.from_registry(CUDA_BASE, add_python="3.12")
        .pip_install(*BASE_PACKAGES, *packages)
        .env({"PYTHONPATH": "/project", "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0", **(env or {})})
        .add_local_file("trainer/kernels/stable_ce.cu", "/opt/stable_ce/stable_ce.cu", copy=True)
        .run_commands(BUILD_KERNEL)
    )


image = project_image()
