"""Publish an n-gram hint pilot's outputs to the HF results dataset.

Run from training/:
    modal run -m model_hybrid.publish_hint_pilot --run-id hint-pilot-01

Copies /runs/hint-pilot/<run>/ from the `lm-exploration-training` volume to
`jfrick/lm-exploration-results` under `pr-109/<run>/`. Checkpoints are converted
to safetensors weights plus a JSON configuration; metrics, logs and hint tables
(numpy .npz of plain arrays) are copied as they are.
"""

import json
import shutil
import time
from pathlib import Path

import modal
from model_hybrid.image import LAUNCH_LIMIT_SECONDS, hybrid_image

image = hybrid_image("huggingface-hub==1.33.0", "safetensors>=0.4").add_local_dir(
    "data", "/project/training_data", ignore=["raw", "cache", "__pycache__"]
)
app = modal.App("lm-exploration-hybrid-hint-publish")
run_volume = modal.Volume.from_name("lm-exploration-training")
REPO = "jfrick/lm-exploration-results"
PREFIX = "pr-109"
COPIED = ("*.json", "*.jsonl", "*.npz")


@app.function(
    image=image,
    cpu=4,
    memory=32768,
    timeout=3600,
    retries=0,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": run_volume},
)
def publish(run_id, deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import torch
    from huggingface_hub import HfApi
    from model_hybrid.hint_pilot import model_config
    from safetensors.torch import save_file

    source = Path("/runs/hint-pilot") / run_id
    staging = Path("/tmp/publish") / run_id
    shutil.rmtree(staging, ignore_errors=True)
    files = []
    for pattern in COPIED:
        for path in source.rglob(pattern):
            target = staging / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files.append(str(path.relative_to(source)))
    for checkpoint in source.rglob("model.pt"):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        folder = staging / checkpoint.parent.relative_to(source)
        folder.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.contiguous() for k, v in saved["model"].items()}, folder / "model.safetensors"
        )
        config = model_config(saved["settings"], saved["hinted"])
        description = {
            "model_config": {k: getattr(config, k) for k in config.__dataclass_fields__},
            "settings": saved["settings"],
            "hinted": saved["hinted"],
            "steps": saved["steps"],
            "dtype": "bfloat16",
        }
        (folder / "model-config.json").write_text(json.dumps(description, indent=2) + "\n")
        files += [
            str(folder.relative_to(staging) / n) for n in ("model.safetensors", "model-config.json")
        ]
    commit = HfApi().upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=str(staging),
        path_in_repo=f"{PREFIX}/{run_id}",
        commit_message=f"N-gram hint pilot {run_id} (koogle/lm-exploration PR 109)",
    )
    return {
        "repo": REPO,
        "path": f"{PREFIX}/{run_id}",
        "commit": commit.oid,
        "files": sorted(files),
    }


@app.local_entrypoint()
def main(run_id: str):
    result = publish.remote(run_id, time.time() + LAUNCH_LIMIT_SECONDS)
    print(json.dumps(result, indent=2))
