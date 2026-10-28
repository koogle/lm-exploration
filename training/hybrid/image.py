"""Modal image for the hybrid family: the torch version locked in uv.lock.

Construct from training/ so local source paths match the installed packages. The
CUDA 13.0 devel base matches torch 2.14.1+cu130.
"""

import modal

CUDA_BASE = "nvidia/cuda:13.0.0-devel-ubuntu22.04"
PACKAGES = ("torch==2.14.1", "numpy", "einops", "ninja")

image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .pip_install(*PACKAGES)
    .env({"PYTHONPATH": "/project", "TORCH_CUDA_ARCH_LIST": "9.0"})
    .add_local_dir("10m", "/project/model_10m", ignore=["__pycache__"])
    .add_local_dir("hybrid", "/project/model_hybrid", ignore=["__pycache__"])
    .add_local_dir("tests", "/project/tests", ignore=["__pycache__", "fixtures"])
)

# One function call per launch: no retries, and the call refuses to start once 24
# hours have passed since launch (queue time included).
LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
