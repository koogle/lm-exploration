"""Bounded L40S/H100/A10G comparison. Does not launch the full training run."""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

from modal_scholarly_pilot import image

image = image.add_local_file("modal_scholarly_pilot.py", "/project/modal_scholarly_pilot.py")

app = modal.App("lm-exploration-gpu-comparison")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
results_volume = modal.Volume.from_name("lm-exploration-training")


@app.function(image=image, cpu=2, memory=4096, timeout=300, retries=0, max_containers=1)
def qualify():
    for filename in ("test_100m_modern.py", "test_tied_embeddings.py", "test_scholarly.py"):
        subprocess.run(
            ["python", "-m", "unittest", "discover", "-s", "/project/tests", "-p", filename, "-v"],
            check=True,
            timeout=120,
        )
    return "passed"


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=24576,
    timeout=900,
    retries=0,
    max_containers=3,
    volumes={"/data": data_volume, "/runs": results_volume},
)
def worker(gpu, run_id, provenance):
    from trainer.gpu_benchmark import benchmark

    output = Path("/runs") / run_id / gpu
    if output.exists():
        raise ValueError("Preserve existing results; use a fresh run ID")
    output.mkdir(parents=True)
    try:
        result = benchmark(
            Path("/data/scholarly-code/scholarly-124m-20261016/snapshots/pilot-001/manifest.json"),
            gpu,
            provenance,
        )
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except Exception as exc:
        (output / "error.json").write_text(
            json.dumps({"type": type(exc).__name__, "error": str(exc)})
        )
        raise
    finally:
        results_volume.commit()


@app.local_entrypoint()
def main(run_id: str = "scholarly-124m-gpu-comparison-20261017-01"):
    import re

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    files = [
        Path("modal_gpu_benchmark.py"),
        Path("modal_scholarly_pilot.py"),
        Path("100m/model.py"),
        Path("10m/model.py"),
        Path("trainer/gpu_benchmark.py"),
        *Path("data").glob("*.py"),
        *Path("tests").glob("test_100m*.py"),
    ]
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    qualify.remote()
    calls = {
        gpu: worker.with_options(gpu=gpu).spawn(gpu, run_id, provenance)
        for gpu in ("L40S", "H100", "A10G")
    }
    Path("results").mkdir(exist_ok=True)
    # Keep individual results even if another device fails or is unavailable.
    errors = []
    for gpu, call in calls.items():
        try:
            result = call.get()
            Path(f"results/{run_id}-{gpu}.json").write_text(json.dumps(result, indent=2) + "\n")
        except Exception as exc:  # noqa: BLE001 -- collect independent remote failures
            errors.append(f"{gpu}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))
