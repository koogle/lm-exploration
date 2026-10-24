"""Modal image with the CUDA toolkit, so custom kernels compile with nvcc at import."""

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.6.0", "numpy", "ninja", "pydantic>=2.11,<3")
    .env({"PYTHONPATH": "/project", "TORCH_CUDA_ARCH_LIST": "9.0"})
    .add_local_dir("100m", "/project/model_100m", ignore=["__pycache__", "images"])
    .add_local_dir("10m", "/project/model_10m", ignore=["__pycache__"])
    .add_local_dir("kernels", "/project/kernels", ignore=["__pycache__"])
)

# Each launch runs one function call on one H100: a 30-minute timeout, no retries, and
# the call refuses to start once 24 hours have passed since launch (queue time included).
LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
limits = {"gpu": "H100", "cpu": 8, "memory": 32768, "timeout": 30 * 60, "retries": 0}
