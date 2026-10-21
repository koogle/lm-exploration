"""Train or explicitly resume the scholarly model on one/two/three/eight Modal GPUs."""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

import modal

from modal_scholarly_pilot import image

image = image.add_local_file("modal_scholarly_pilot.py", "/project/modal_scholarly_pilot.py")
app = modal.App("lm-exploration-distributed-training")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
run_volume = modal.Volume.from_name("lm-exploration-training")


def validate_paths(run_id, snapshot, resume):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/snapshots/[a-zA-Z0-9_-]+", snapshot):
        raise ValueError("Expected version/snapshots/name")
    if resume and not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}/step-[0-9]{9}\.pt", resume):
        raise ValueError("Resume must name a trusted run-id/step-NNNNNNNNN.pt on our run volume")
    if resume and resume.split("/")[0] == run_id:
        raise ValueError("Resume uses a fresh run ID to preserve the previous job")


@app.function(
    image=image,
    gpu="H100:3",
    cpu=12,
    memory=32768,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def worker(run_id, snapshot, settings, provenance, resume="", max_seconds=3000, max_steps=None):
    from trainer.distributed_config import DistributedConfig
    from trainer.run_validation import validate_run

    validate_paths(run_id, snapshot, resume)
    config = DistributedConfig(**settings)
    validate_run(config)
    if not 1 <= max_seconds <= 82800 or (max_steps is not None and max_steps < 1):
        raise ValueError("Invalid session limits")
    output = Path("/runs") / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = Path("/data/scholarly-code") / snapshot / "manifest.json"
    try:
        for folder in ("model_100m", "model_10m", "trainer", "training_data"):
            shutil.copytree(
                Path("/project") / folder,
                output / "source" / folder,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        shutil.copytree("/project/tokenizer/structured-code-v1", output / "tokenizer")
        shutil.copyfile(manifest, output / "data-manifest.json")
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        request = output / "request.json"
        request.write_text(json.dumps(settings, indent=2) + "\n")
        command = [
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={config.world_size}",
            "-m",
            "trainer.distributed_train",
            "--config",
            str(request),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--max-seconds",
            str(max_seconds),
            "--volume-name",
            "lm-exploration-training",
        ]
        if resume:
            command.extend(["--resume", str(Path("/runs") / resume)])
        if max_steps is not None:
            command.extend(["--max-steps", str(max_steps)])
        print(
            json.dumps({"run_id": run_id, "world_size": config.world_size, "resume": resume}),
            flush=True,
        )
        with (output / "torchrun.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=max_seconds + 480)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        if code:
            raise RuntimeError(
                f"Training failed; inspect {output}/torchrun.log: "
                + (output / "torchrun.log").read_text()[-3000:]
            )
        return json.loads((output / "report.json").read_text())
    finally:
        run_volume.commit()


@app.local_entrypoint()
def main(
    config: str,
    run_id: str,
    snapshot: str,
    gpu: str = "H100",
    resume: str = "",
    max_seconds: int = 3000,
    max_steps: int = 0,
):
    from trainer.distributed_config import DistributedConfig
    from trainer.run_validation import validate_run

    validate_paths(run_id, snapshot, resume)
    settings = DistributedConfig(**json.loads(Path(config).read_text()))
    validate_run(settings)
    if gpu not in {"H100", "L40S", "A10G"}:
        raise ValueError("Supported GPU families: H100, L40S, A10G")
    if not 1 <= max_seconds <= 82800 or max_steps < 0:
        raise ValueError("Invalid session limits")
    files = [
        Path(__file__),
        *Path("trainer").glob("*.py"),
        *Path("data").glob("*.py"),
        Path("100m/model.py"),
        Path("10m/model.py"),
    ]
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "gpu": gpu,
    }
    report = worker.with_options(
        gpu=f"{gpu}:{settings.world_size}", cpu=4 * settings.world_size
    ).remote(
        run_id, snapshot, settings.as_dict(), provenance, resume, max_seconds, max_steps or None
    )
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{run_id}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
