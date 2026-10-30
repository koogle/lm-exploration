"""One-H100 gate for document-masked attention in the shared training image.

Run from training/:
    modal run -m modal_scripts.modal_document_check [--output <result.json>]

One H100, 30-minute timeout, no retries. The launcher records a deadline 24 hours
after launch, and the worker refuses to start after it. The worker runs
tests/check_documents_cuda.py (varlen attention against explicit per-document
attention, the compiled model against eager, attention timing) and the trainer's
single-rank CUDA gate. Results are structured JSON in the ignored artifacts/.
"""

import json
import subprocess
import time
from pathlib import Path

import modal

from distributed.image import image

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60

app = modal.App("lm-exploration-document-check")


def run(command, timeout):
    result = subprocess.run(
        command, check=False, cwd="/project", capture_output=True, text=True, timeout=timeout
    )
    print(result.stdout, result.stderr)
    if result.returncode:
        raise RuntimeError(f"Failed: {' '.join(command)}")
    return result


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=1800, retries=0)
def check(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import torch

    documents = run(["python", "tests/check_documents_cuda.py"], 900)
    run(["torchrun", "--nproc_per_node=1", "tests/check_optimization_cuda.py"], 900)
    return {
        "torch": torch.__version__,
        "documents": json.loads(documents.stdout[documents.stdout.index("{") :]),
        "trainer_gate": "passed",
    }


@app.local_entrypoint()
def main(output: str = ""):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    result = check.remote(deadline)
    path = Path(output or f"artifacts/results/document-check-{int(time.time())}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
