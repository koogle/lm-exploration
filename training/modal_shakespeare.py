"""Bounded from-scratch 10M Shakespeare training on a dedicated Modal volume.

Run from training/ with the Modal CLI. No worker starts on import.
"""

import json
import re
from pathlib import Path

import modal

app = modal.App("lm-exploration-shakespeare-10m")
volume = modal.Volume.from_name("lm-exploration-training", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "numpy>=1.26,<3", "tqdm==4.70.1", "safetensors==0.8.0")
    .add_local_file("pyproject.toml", "/project/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/project/uv.lock", copy=True)
    .add_local_dir("100m", "/project/100m", copy=True)
    .add_local_dir("10m", "/project/10m", copy=True)
    .add_local_dir("trainer", "/project/trainer", copy=True)
    .add_local_dir("data", "/project/data", copy=True, ignore=["raw", "cache", "__pycache__"])
    .add_local_file("cli.py", "/project/cli.py", copy=True)
    .add_local_file("model_checks.py", "/project/model_checks.py", copy=True)
    .add_local_file("ui.py", "/project/ui.py", copy=True)
    .run_commands("pip install --no-deps /project")
)


def execute(run_id, smoke):
    import shutil
    import time

    import torch
    from model_10m import model as architecture
    from training_data.fetching.fetch_tinyshakespeare import main as fetch
    from training_data.loading import load_data

    from trainer.config import TrainConfig
    from trainer.loop import train

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("invalid run ID")
    torch.set_num_threads(4)
    fetch()
    root = Path("/runs") / run_id
    data = Path(__import__("training_data").__file__).parent / "raw/tinyshakespeare.txt"
    tokenizer, _, _, _ = load_data(data, 256)
    cfg = architecture.ModelConfig(vocab_size=len(tokenizer.characters))
    args = TrainConfig(
        data=str(data),
        out_dir=str(root),
        device="cuda",
        steps=5 if smoke else 4000,
        batch_size=2 if smoke else 32,
        lr=3e-4,
        seed=1337,
        eval_interval=5 if smoke else 200,
        log_interval=5 if smoke else 200,
        eval_batches=2 if smoke else 20,
        checkpoint_interval=5 if smoke else 200,
    )
    started = time.perf_counter()
    try:
        train(
            args,
            architecture_module=architecture,
            model_config=cfg,
            max_seconds=60 if smoke else 1200,
        )
        config = json.loads((root / "config.json").read_text())
        metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        candidates = [
            row
            for row in metrics
            if "val_loss" in row and (root / "checkpoints" / f"step-{row['step']:06d}.pt").is_file()
        ]
        if not candidates:
            raise ValueError("no evaluated checkpoint available")
        best = min(candidates, key=lambda row: row["val_loss"])
        shutil.copyfile(root / "checkpoints" / f"step-{best['step']:06d}.pt", root / "best.pt")
        report = {
            "run_id": run_id,
            "smoke": smoke,
            "gpu": torch.cuda.get_device_name(),
            "config": config,
            "selected": best,
            "final": metrics[-1],
            "worker_elapsed_seconds": time.perf_counter() - started,
            "sample": (root / "sample.txt").read_text(),
            "selection": "lowest measured validation loss among saved checkpoints",
            "limits": "one seed; contiguous 90/10 split; no sealed test or broad capability claims",
        }
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        volume.commit()


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=120,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def smoke_worker(run_id: str):
    return execute(run_id, True)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=1500,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def train_worker(run_id: str):
    return execute(run_id, False)


@app.local_entrypoint()
def main(run_id: str, smoke: bool = False):
    ledger_path = Path("runs/shakespeare-budget.json")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {
            "cap_usd": 3.0,
            "conservative_gpu_hour_usd": 5.0,
            "reservations": [],
            "authorization": "user requested Modal training and public Hugging Face publication",
        }
    )
    timeout = 120 if smoke else 1500
    amount = timeout / 3600 * ledger["conservative_gpu_hour_usd"]
    if any(row["run_id"] == run_id for row in ledger["reservations"]):
        raise ValueError("run ID already reserved; inspect its existing execution")
    if sum(row["usd"] for row in ledger["reservations"]) + amount > ledger["cap_usd"]:
        raise ValueError("bounded training allocation exhausted")
    ledger["reservations"].append({"run_id": run_id, "timeout_seconds": timeout, "usd": amount})
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    report = (smoke_worker if smoke else train_worker).remote(run_id)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{run_id}.json").write_text(json.dumps(report, indent=2) + "\n")
