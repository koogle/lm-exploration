"""Shared Modal image for subword training, qualification and benchmarks.

Construct from training/ so local source paths match the installed packages.
The CUDA devel base provides nvcc for the cross-entropy kernel
(trainer/kernels/stable_ce.cu), prebuilt here into /opt/stable_ce.
"""

import modal

CUDA_BASE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"  # Matches torch 2.6.0+cu124.
BASE_PACKAGES = ("torch==2.6.0", "numpy", "pydantic>=2.11,<3", "tokenizers==0.22.2", "ninja")
BUILD_KERNEL = (
    'python -c "from torch.utils.cpp_extension import load; '
    "load(name='stable_ce', sources=['/opt/stable_ce/stable_ce.cu'], "
    "build_directory='/opt/stable_ce', extra_cuda_cflags=['-O3'])\""
)


def project_image(*packages, env=None):
    """Extra pip packages and environment must precede local sources in a Modal image."""
    return (
        modal.Image.from_registry(CUDA_BASE, add_python="3.12")
        .pip_install(*BASE_PACKAGES, *packages)
        .env({"PYTHONPATH": "/project", "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0", **(env or {})})
        .add_local_file("trainer/kernels/stable_ce.cu", "/opt/stable_ce/stable_ce.cu", copy=True)
        .run_commands(BUILD_KERNEL)
        .add_local_dir("100m", "/project/model_100m")
        .add_local_dir("10m", "/project/model_10m")
        .add_local_dir("trainer", "/project/trainer")
        .add_local_dir("distributed", "/project/distributed")
        .add_local_dir("workflows", "/project/workflows")
        .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
        .add_local_dir("tokenizer", "/project/tokenizer")
        .add_local_dir("tests", "/project/tests")
    )


image = project_image()
