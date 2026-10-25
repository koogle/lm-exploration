"""Train or explicitly resume the Simple LLM model on one/two/three/eight Modal GPUs."""

import json
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

import modal
from training_data.growing import STATE_ENV

from distributed.config import DistributedConfig
from distributed.image import image
from workflows.checks import validate_paths

app = modal.App("lm-exploration-distributed-training")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
run_volume = modal.Volume.from_name("lm-exploration-training")


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
def worker(
    run_id,
    snapshot,
    settings,
    provenance,
    resume="",
    max_seconds=3000,
    max_steps=None,
    *,
    dataset_root,
):
    validate_paths(run_id, snapshot, resume)
    if not isinstance(dataset_root, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", dataset_root):
        raise ValueError("Invalid dataset root")
    config = DistributedConfig(**settings)
    if not 1 <= max_seconds <= 82800 or (max_steps is not None and max_steps < 1):
        raise ValueError("Invalid session limits")
    output = Path("/runs") / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = Path("/data") / dataset_root / snapshot / "manifest.json"
    try:
        for folder in (
            "model_100m",
            "model_10m",
            "trainer",
            "distributed",
            "workflows",
            "training_data",
        ):
            shutil.copytree(
                Path("/project") / folder,
                output / "source" / folder,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        from training_data.subword import DEFAULT_FOLDER

        shutil.copytree(DEFAULT_FOLDER, output / "tokenizer")
        shutil.copyfile(manifest, output / "data-manifest.json")
        if STATE_ENV in os.environ:
            # A refresher reloads the data volume while it grows; read the run's copy.
            manifest = output / "data-manifest.json"
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
            "trainer.loop",
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
