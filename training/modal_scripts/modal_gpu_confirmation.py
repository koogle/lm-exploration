"""Longer H100 scaling confirmation with bounded replay of the benchmark buffer.

This is throughput measurement only: no checkpoint or reusable model is saved.
It does not relax the full training run's two-pass policy.
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import modal

from distributed.image import image

app = modal.App("lm-exploration-h100-confirmation")
data = modal.Volume.from_name("lm-exploration-mixture-data")
results = modal.Volume.from_name("lm-exploration-training")


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    timeout=900,
    retries=0,
    max_containers=3,
    volumes={"/data": data, "/runs": results},
)
def worker(world, run_id, provenance):
    output = Path("/runs") / run_id / f"H100x{world}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = (
        "/data/scholarly-code/scholarly-124m-20261016/snapshots/pilot-001-seekable/manifest.json"
    )
    started = time.monotonic()
    try:
        if world == 1:
            from trainer.gpu_benchmark import benchmark

            report = benchmark(
                manifest,
                "H100",
                provenance,
                microbatches=(24,),
                sustained_steps=500,
                benchmark_replay=True,
            )
        else:
            batch = 24 if world == 2 else 16
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
                    "500",
                    "--output",
                    str(output / "sustained.json"),
                    "--benchmark-replay",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
            (output / "trial.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(result.stderr[-4000:])
            report = {
                "requested_gpu": f"H100:{world}",
                "world_size": world,
                "provenance": provenance,
                "sustained": json.loads((output / "sustained.json").read_text()),
                "benchmark_replay": True,
                "elapsed_seconds": time.monotonic() - started,
            }
        report["cpu"] = 4 * world
        report["host_memory_mib"] = 32768
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {"gpus": world, "tokens_per_second": report["sustained"]["loss_tokens_per_second"]}
            ),
            flush=True,
        )
        return report
    finally:
        results.commit()


@app.local_entrypoint()
def main(run_id: str = "simple-llm-124m-h100-confirmation-20261017-01"):
    import re

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    Path("artifacts/results").mkdir(parents=True, exist_ok=True)
    files = [
        Path("modal_scripts/modal_gpu_confirmation.py"),
        *Path("distributed").glob("*.py"),
        Path("trainer/ddp_benchmark.py"),
        Path("trainer/gpu_benchmark.py"),
        Path("100m/model.py"),
        Path("10m/model.py"),
        *Path("data").glob("*.py"),
    ]
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    calls = {
        world: worker.with_options(gpu=f"H100:{world}", cpu=4 * world).spawn(
            world, run_id, provenance
        )
        for world in (1, 2, 3)
    }
    for world, call in calls.items():
        report = call.get()
        Path(f"artifacts/results/{run_id}-H100x{world}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
