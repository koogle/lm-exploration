"""Bounded two/three GPU data-parallel benchmarks; never starts full pretraining."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import modal

from modal_scripts.modal_scholarly_pilot import image

app = modal.App("lm-exploration-multigpu-comparison")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
results_volume = modal.Volume.from_name("lm-exploration-training")


@app.function(
    image=image,
    gpu="H100:2",
    cpu=8,
    memory=32768,
    timeout=1800,
    retries=0,
    max_containers=4,
    volumes={"/data": data_volume, "/runs": results_volume},
)
def worker(gpu, world, run_id, provenance, sustained_steps):
    import torch

    output = Path("/runs") / run_id / f"{gpu}x{world}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = "/data/scholarly-code/scholarly-124m-20261016/snapshots/pilot-001/manifest.json"
    rows = []
    start = time.monotonic()

    def run(batch, steps, name):
        path = output / f"{name}.json"
        result = subprocess.run(
            [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={world}",
                "-m",
                "trainer.ddp_benchmark",
                "--manifest",
                manifest,
                "--microbatch",
                str(batch),
                "--steps",
                str(steps),
                "--output",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        (output / f"{name}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            if "out of memory" in (result.stdout + result.stderr).lower():
                return {"status": "out_of_memory", "microbatch_per_gpu": batch}
            raise RuntimeError(
                f"DDP trial failed; preserved log at {output}/{name}.log: " + result.stderr[-3000:]
            )
        return {"status": "ok", **json.loads(path.read_text())}

    try:
        for batch in (2, 4, 8, 12, 24) if world == 2 else (2, 4, 8, 16):
            row = run(batch, 6, f"batch-{batch}")
            rows.append(row)
            print(
                json.dumps(
                    {
                        "gpu": gpu,
                        "gpus": world,
                        **{k: v for k, v in row.items() if k not in ("measurements", "data_end")},
                    }
                ),
                flush=True,
            )
            results_volume.commit()
            if row["status"] == "out_of_memory":
                break
        selected = max(
            (r for r in rows if r["status"] == "ok"), key=lambda r: r["loss_tokens_per_second"]
        )
        sustained = run(selected["microbatch_per_gpu"], sustained_steps, "sustained")
        if sustained["status"] != "ok":
            raise RuntimeError("Selected batch did not pass sustained validation")
        report = {
            "requested_gpu": f"{gpu}:{world}",
            "world_size": world,
            "torch_version": str(torch.__version__),
            "precision": "BF16 autocast / FP32 parameters",
            "attention": "PyTorch causal SDPA",
            "optimizer": "fused AdamW",
            "context": 2048,
            "effective_windows": 48,
            "cpu": 4 * world,
            "host_memory_mib": 32768,
            "provenance": provenance,
            "sweep": rows,
            "sustained": sustained,
            "elapsed_seconds": time.monotonic() - start,
            "notes": [
                "NCCL one-host DDP on one shared model, not independent model replicas.",
                "Same 48 global windows per update and data prefix as single-GPU trials.",
                "Timings include loader, scatter, H2D, gradient synchronization and optimizer.",
                "Unequal loss-token counts and optimizer updates checked against single-model reference.",
                "Checkpoint/evaluation/preparation/startup excluded from timed steps.",
            ],
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        results_volume.commit()


@app.local_entrypoint()
def main(
    run_id: str = "scholarly-124m-gpu-comparison-20261017-01",
    gpus: str = "H100",
    sustained_steps: int = 30,
):
    import re

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    gpu_types = gpus.split(",")
    if not gpu_types or any(gpu not in {"H100", "L40S", "A10G"} for gpu in gpu_types):
        raise ValueError("GPUs must be a comma-separated subset of H100,L40S,A10G")
    if not 1 <= sustained_steps <= 100:
        raise ValueError("Sustained steps must be in 1..100 for the bounded pilot buffer")
    Path("results").mkdir(exist_ok=True)
    files = [
        Path("modal_scripts/modal_multi_gpu_benchmark.py"),
        Path("modal_scripts/modal_scholarly_pilot.py"),
        Path("trainer/ddp_benchmark.py"),
        Path("100m/model.py"),
        Path("10m/model.py"),
        Path("10m/legacy.py"),
        *Path("data").glob("*.py"),
    ]
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    calls = {}
    for gpu in gpu_types:
        for world in (2, 3):
            calls[gpu, world] = worker.with_options(gpu=f"{gpu}:{world}", cpu=world * 4).spawn(
                gpu, world, run_id, provenance, sustained_steps
            )
    errors = []
    for (gpu, world), call in calls.items():
        try:
            result = call.get()
            Path(f"results/{run_id}-{gpu}x{world}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
        except Exception as exc:  # noqa: BLE001 -- retain results from independent remote trials
            errors.append(f"{gpu} x {world}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))
