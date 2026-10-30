"""Hybrid tools on Modal H100s; training runs through the workflows (workflows/README.md).

Run from training/:
    modal run -m model_hybrid.modal_tools::bench --gpus 1 [--microbatch 4]

Each benchmark is one call with a 30-minute timeout and no retries.
"""

import json
import subprocess
import time
from pathlib import Path

import modal
from model_hybrid.image import image

app = modal.App("lm-exploration-hybrid-tools")
data_volume = modal.Volume.from_name("lm-exploration-hybrid-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("lm-exploration-hybrid-runs", create_if_missing=True)
DEFAULT_CONFIG = "hybrid/configs/hybrid-720m-40b.json"
VOLUMES = {"/data": data_volume, "/runs": runs_volume}


def run_torchrun(module, args, gpus, timeout):
    """torchrun's exit code and output lines (printed as they arrive)."""
    command = ["torchrun", "--standalone", f"--nproc_per_node={gpus}", "-m", module, *args]
    lines = []
    process = subprocess.Popen(
        command, cwd="/project", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    start = time.monotonic()
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
        if time.monotonic() - start > timeout:
            process.terminate()
    return process.wait(), lines


@app.function(
    image=image, gpu="H100:8", cpu=32, memory=262144, timeout=1800, retries=0, volumes=VOLUMES
)
def benchmark(config, gpus, steps, microbatch, global_rows, document_tokens, deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    from model_hybrid.benchmark import summarize

    path = Path("/tmp/run-config.json")
    path.write_text(json.dumps(config))
    args = ["--config", str(path), "--steps", str(steps), "--document-tokens", str(document_tokens)]
    if microbatch:
        args += ["--microbatch", str(microbatch)]
    if global_rows:
        args += ["--global-rows", str(global_rows)]
    code, lines = run_torchrun("model_hybrid.benchmark", args, gpus, 1500)
    if code:
        raise RuntimeError("".join(lines[-60:]))
    import torch

    return {
        "gpus": gpus,
        "microbatch": microbatch or config["microbatch"],
        "torch": torch.__version__,
        **summarize(lines),
    }


def load_config(path):
    return json.loads(Path(path).read_text())


@app.local_entrypoint()
def bench(
    config: str = DEFAULT_CONFIG,
    gpus: int = 1,
    steps: int = 20,
    microbatch: int = 0,
    global_rows: int = 0,
    document_tokens: int = 1000,
):
    """One bounded throughput benchmark; with one GPU the global batch shrinks to fit."""
    from workflows.deadline import MAX_LAUNCH_SECONDS

    settings = load_config(config)
    rows = global_rows or (
        settings["global_rows"] if gpus == 8 else 8 * (microbatch or settings["microbatch"]) * gpus
    )
    result = benchmark.with_options(gpu=f"H100:{gpus}").remote(
        settings, gpus, steps, microbatch, rows, document_tokens, time.time() + MAX_LAUNCH_SECONDS
    )
    print(json.dumps(result, indent=2))
