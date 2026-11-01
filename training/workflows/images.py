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
