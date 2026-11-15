"""The project sources every Modal image carries, whatever model family it runs.

Each image adds the same packages under /project (PYTHONPATH), so any workflow,
trainer or family module imports in any image. Construct from training/.
"""

SOURCES = (
    ("10m", "model_10m"),
    ("100m", "model_100m"),
    ("hybrid", "model_hybrid"),
    ("trainer", "trainer"),
    ("kernels", "kernels"),
    ("distributed", "distributed"),
    ("workflows", "workflows"),
    ("data", "training_data"),
    ("analysis/evaluation", "evaluation"),
    ("tokenizer", "tokenizer"),
    ("modal_scripts", "modal_scripts"),
    ("tests", "tests"),
)
IGNORE = ["raw", "cache", "__pycache__"]


def add_project(image):
    """`image` with every project package added under /project."""
    for local, package in SOURCES:
        image = image.add_local_dir(local, f"/project/{package}", ignore=IGNORE)
    return image


def vllm_base(env=None):
    """CUDA 13.0 devel image with vLLM 0.31 and the Hugging Face cache at /hf. vLLM's
    FlashInfer kernels compile at load time, so the image carries nvcc; `env` adds to
    the environment. Callers add the project with `add_project`."""
    import modal

    return (
        modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
        .pip_install("vllm==0.31.0", "huggingface-hub[hf_transfer]", "pyarrow==26.0.0")
        .env(
            {
                "HF_HOME": "/hf",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PYTHONPATH": "/project",
                "CUDA_HOME": "/usr/local/cuda",
                **(env or {}),
            }
        )
    )
