"""Prepare a chat SFT recipe (hybrid/recipes/chat-sft-*.json) on Modal CPUs.

Run from training/:
    modal run -m model_hybrid.modal_sft_data [--recipe hybrid/recipes/chat-sft-v1.json]

One call per parquet file, all in parallel: download it at the recipe's pinned
revision, convert rows to OpenAI messages, render them in GLM's chat format with
loss masks, drop conversations longer than the context or sharing a 13-gram with
the pretraining recipe's benchmark screen, hold out a hash-selected fraction, and
write the subset to the `lm-exploration-hybrid-data` volume under `<recipe name>/`
(see sft_data.py). Each call has `retries=0`, a 1-hour timeout and refuses to start
24 hours after launch; finished subsets are skipped. `summary.json` collects the
per-subset counts.
"""

import io
import json
import time
from pathlib import Path

import modal

app = modal.App("lm-exploration-hybrid-sft-data")
volume = modal.Volume.from_name("lm-exploration-hybrid-data", create_if_missing=True)
secret = modal.Secret.from_name("simple-llm-hf")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy", "pyarrow==26.0.0", "tokenizers==0.22.2", "huggingface-hub==1.33.0", "jinja2==3.1.6"
    )
    .env({"PYTHONPATH": "/project", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_dir("hybrid", "/project/model_hybrid", ignore=["__pycache__", "kernels"])
)
ROOT = Path("/data")
DEFAULT_RECIPE = "hybrid/recipes/chat-sft-v1.json"
LAUNCH_LIMIT_SECONDS = 24 * 60 * 60


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3600,
    retries=0,
    secrets=[secret],
    volumes={"/data": volume},
)
def prepare(recipe, file, deadline):
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from model_hybrid import sft_data, tokenizer

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    folder = ROOT / recipe["name"]
    name = Path(file).stem
    if (folder / f"{name}.json").exists():
        return json.loads((folder / f"{name}.json").read_text())
    path = hf_hub_download(
        recipe["repo"], file, revision=recipe["revision"], repo_type="dataset", local_dir="/tmp/sft"
    )
    rows = pq.read_table(path, columns=["messages", "chat_template_kwargs"]).to_pylist()
    screen = np.load(ROOT / recipe["screen"])
    glm = tokenizer.load(tokenizer.tokenizer_file("/tmp/tokenizer"))
    meta = sft_data.prepare_subset(recipe, name, rows, folder, glm, screen)
    volume.commit()
    return meta


@app.local_entrypoint()
def main(recipe: str = DEFAULT_RECIPE):
    settings = json.loads(Path(recipe).read_text())
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    metas = list(prepare.starmap([(settings, file, deadline) for file in settings["files"]]))
    keys = ("conversations", "tokens", "target_tokens", "heldout_conversations")
    summary = {
        "recipe": settings["name"],
        "subsets": metas,
        "totals": {key: sum(m[key] for m in metas) for key in keys},
        "dropped": {key: sum(m["dropped"][key] for m in metas) for key in metas[0]["dropped"]},
    }
    with volume.batch_upload(force=True) as upload:
        upload.put_file(
            io.BytesIO(json.dumps(summary, indent=2).encode()),
            f"{settings['name']}/summary.json",
        )
    print(json.dumps(summary["totals"] | {"dropped": summary["dropped"]}, indent=2))
