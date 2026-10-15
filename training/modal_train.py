"""Launch a training recipe (architecture + dataset + settings) on Modal.

Run from training/ with the Modal CLI; see recipes/README.md. No worker starts on
import. The recipe is validated and the budget reserved locally before launch.
"""

import json
from pathlib import Path

import modal

from trainer.recipe import load_recipe

LEDGER = Path("runs/modal-budget.json")

app = modal.App("lm-exploration-training")
volume = modal.Volume.from_name("lm-exploration-training", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "numpy>=1.26,<3", "tqdm==4.70.1", "safetensors==0.8.0")
    # The whole uv project, so new architectures and fetchers need no image edits.
    .add_local_dir(
        ".",
        "/project",
        copy=True,
        ignore=[
            ".venv",
            "runs",
            "results",
            "designs",
            "data/raw",
            "data/cache",
            "**/__pycache__",
            "*.egg-info",
            "build",
            "dist",
        ],
    )
    .run_commands("pip install --no-deps /project")
)


@app.function(image=image, gpu="L4", retries=0, max_containers=1, volumes={"/runs": volume})
def worker(recipe, run_id: str):
    import torch

    from trainer.recipe import run_recipe

    torch.set_num_threads(recipe.backend.cpu)
    try:
        return run_recipe(recipe, Path("/runs") / run_id)
    finally:
        volume.commit()


def reserve(run_id, timeout_seconds):
    # Reservation accounting at a conservative rate, not an actual bill.
    if not LEDGER.exists():
        raise ValueError(
            f"{LEDGER} is missing: restore it, or create it with an approved cap, e.g. "
            '{"cap_usd": 10.0, "conservative_gpu_hour_usd": 5.0, "reservations": []}'
        )
    ledger = json.loads(LEDGER.read_text())
    amount = timeout_seconds / 3600 * ledger["conservative_gpu_hour_usd"]
    if any(row["run_id"] == run_id for row in ledger["reservations"]):
        raise ValueError("run ID already reserved; inspect its existing execution")
    if sum(row["usd"] for row in ledger["reservations"]) + amount > ledger["cap_usd"]:
        raise ValueError("training budget allocation exhausted")
    ledger["reservations"].append(
        {"run_id": run_id, "timeout_seconds": timeout_seconds, "usd": amount}
    )
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")


@app.local_entrypoint()
def main(recipe: str, run_id: str, smoke: bool = False):
    # The run ID becomes a folder on the volume and a results/ file name.
    plain = run_id.replace("-", "").replace("_", "")
    if not (plain.isascii() and plain.isalnum()) or len(run_id) > 80:
        raise ValueError("run ID: up to 80 letters, digits, _ or -")
    loaded = load_recipe(recipe)
    if smoke:
        loaded = loaded.as_smoke()
    backend = loaded.backend
    reserve(run_id, backend.timeout_seconds)
    configured = worker.with_options(
        gpu=backend.gpu,
        cpu=backend.cpu,
        memory=backend.memory_mib,
        timeout=backend.timeout_seconds,
    )
    report = configured.remote(loaded, run_id)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{run_id}.json").write_text(json.dumps(report, indent=2) + "\n")
