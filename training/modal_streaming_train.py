"""Start GPU training from a bounded live upstream prefix, then prepare ahead.

This launcher does not scan or tokenize the whole corpus before requesting GPUs.
All launched runs require an explicit config/run ID. No jobs launch on import.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import modal

from distributed.image import project_image

image = project_image(
    "requests==2.34.2",
    "pyarrow==26.0.0",
    "huggingface-hub==1.33.0",
    env={"RAYON_NUM_THREADS": "8"},
)
app = modal.App("lm-exploration-overlapped-training")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
run_volume = modal.Volume.from_name("lm-exploration-training")


@app.function(
    image=image, cpu=2, memory=4096, timeout=120, retries=0, volumes={"/data": data_volume}
)
def initialize(run_id):
    from training_data.simple_llm import SPLIT_VERSION, load_recipe
    from training_data.subword import TOKENIZER_SHA256

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", run_id):
        raise ValueError("Invalid run ID")
    snapshot = f"{run_id}/snapshots/live"
    folder = Path("/data/scholarly-code") / snapshot
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format": "scholarly-live-v1",
        "recipe": load_recipe("/project/tokenizer/recipe-v1.json"),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "split_policy": SPLIT_VERSION,
        "max_passes": 2,
        "max_seen_documents": 5_000_000,
        "training_ready": False,
        "pending": [
            "full corpus capacity",
            "global near deduplication",
            "eligibility and contamination audits",
        ],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    data_volume.commit()
    return snapshot


@app.function(
    image=image,
    gpu="L40S:3",
    cpu=16,
    memory=32768,
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def worker(run_id, snapshot, settings, provenance, resume="", max_steps=None, max_seconds=3000):
    from distributed.modal_train import worker as existing_worker

    return existing_worker.local(
        run_id,
        snapshot,
        settings,
        provenance,
        resume,
        max_seconds,
        max_steps,
        dataset_root="scholarly-code",
    )


@app.local_entrypoint()
def main(
    config: str,
    run_id: str,
    snapshot: str = "",
    gpu: str = "H100",
    resume: str = "",
    max_seconds: int = 3000,
    max_steps: int = 0,
):
    from distributed.config import DistributedConfig

    cfg = DistributedConfig(**json.loads(Path(config).read_text()))
    if not cfg.prefetch_batches:
        raise ValueError("Set prefetch_batches > 0")
    if gpu not in ("H100", "L40S", "A10G"):
        raise ValueError("Unsupported GPU family")
    if not 1 <= max_seconds <= 3000:
        raise ValueError("This review launcher uses bounded sessions <=3000s")
    if resume and not snapshot:
        raise ValueError("Resume must pin its original live manifest")
    files = [
        Path(__file__),
        *Path("data").glob("*.py"),
        *Path("trainer").glob("*.py"),
        *Path("distributed").glob("*.py"),
        Path("100m/model.py"),
        Path("10m/model.py"),
    ]
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "mode": "bounded producer/consumer; checkpoint only acknowledged batches",
    }
    snapshot = snapshot or initialize.remote(run_id)
    report = worker.with_options(gpu=f"{gpu}:{cfg.world_size}", cpu=4 * cfg.world_size).remote(
        run_id, snapshot, cfg.as_dict(), provenance, resume, max_steps or None, max_seconds
    )
    Path("artifacts/results").mkdir(parents=True, exist_ok=True)
    Path(f"artifacts/results/{run_id}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
