"""Shared Modal image for subword training, qualification and benchmarks.

Construct from training/ so local source paths match the installed packages.
"""

import modal

BASE_PACKAGES = ("torch==2.6.0", "numpy", "pydantic>=2.11,<3", "tokenizers==0.22.2")


def project_image(*packages, env=None):
    """Extra pip packages and environment must precede local sources in a Modal image."""
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*BASE_PACKAGES, *packages)
        .env({"PYTHONPATH": "/project", **(env or {})})
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
