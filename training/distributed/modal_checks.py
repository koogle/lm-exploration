"""Bounded CUDA/NCCL correctness and restart checks for the distributed trainer."""

import json
import subprocess
from pathlib import Path

import modal

from distributed.image import image

app = modal.App("lm-exploration-distributed-checks")
volume = modal.Volume.from_name("lm-exploration-training")


@app.function(
    image=image,
    gpu="H100:3",
    cpu=12,
    memory=32768,
    timeout=900,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def check(run_id):
    import runpy

    try:
        for filename in (
            "test_distributed_config.py",
            "test_distributed_single_gpu.py",
            "test_100m_modern.py",
            "test_tied_embeddings.py",
            "test_simple_llm.py",
            "test_token_shards.py",
        ):
            subprocess.run(
                [
                    "python",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "/project/tests",
                    "-p",
                    filename,
                    "-v",
                ],
                check=True,
                timeout=120,
            )
        checks = runpy.run_path("/project/tests/check_distributed_cuda.py")["checks"]
        return checks(Path("/runs") / run_id)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(run_id: str):
    import re

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    result = check.remote(run_id)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{run_id}.json").write_text(json.dumps(result, indent=2) + "\n")
