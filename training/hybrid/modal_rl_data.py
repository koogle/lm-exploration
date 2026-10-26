"""Prepare an RL prompt recipe (hybrid/recipes/rl-*.json) on a Modal CPU container.

Run from training/:
    modal run -m model_hybrid.modal_rl_data [--recipe hybrid/recipes/rl-v1.json]

Downloads each source at its pinned revision, keeps only code problems whose own
reference solution passes its tests in this container's sandbox (so every reward
is attainable), builds the prompts (rl_data.prepare: benchmark screen, held-out
split) and writes them to the `lm-exploration-hybrid-data` volume under
`<recipe name>/`. One call, `retries=0`, a 1-hour timeout, refusing work 24 hours
after launch. Reference code runs only inside the container.
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("lm-exploration-hybrid-rl-data")
volume = modal.Volume.from_name("lm-exploration-hybrid-data")
secret = modal.Secret.from_name("simple-llm-hf")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy",
        "pyarrow==26.0.0",
        "tokenizers==0.22.2",
        "huggingface-hub==1.33.0",
        "math-verify==0.9.0",
        "pytest==8.4.2",
    )
    .env({"PYTHONPATH": "/project", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_dir("hybrid", "/project/model_hybrid", ignore=["__pycache__", "kernels"])
)
DEFAULT_RECIPE = "hybrid/recipes/rl-v1.json"
LAUNCH_LIMIT_SECONDS = 24 * 60 * 60


@app.function(
    image=image,
    cpu=32,
    memory=32768,
    timeout=3600,
    retries=0,
    secrets=[secret],
    volumes={"/data": volume},
)
def prepare(recipe, deadline):
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from model_hybrid import rewards, rl_data, tokenizer

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    rows, verified = {}, {}
    for name, source in recipe["sources"].items():
        path = hf_hub_download(
            source["repo"], source["file"], revision=source["revision"], repo_type="dataset"
        )
        rows[name] = pq.read_table(path).to_pylist()
        if source["kind"] == "code":
            checks = [("code", f"```python\n{r['solution']}\n```", r["test"]) for r in rows[name]]
            passed = rewards.score(checks, workers=32)
            verified[name] = {"rows": len(rows[name]), "reference_passes": int(sum(passed))}
            rows[name] = [r for r, ok in zip(rows[name], passed, strict=True) if ok]
    screen = np.load(Path("/data") / recipe["screen"])
    glm = tokenizer.load(tokenizer.tokenizer_file("/tmp/tokenizer"))
    summary = rl_data.prepare(recipe, rows, "/data", glm, screen)
    summary["verified_code"] = verified
    (Path("/data") / recipe["name"] / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    volume.commit()
    return summary


@app.local_entrypoint()
def main(recipe: str = DEFAULT_RECIPE):
    settings = json.loads(Path(recipe).read_text())
    print(json.dumps(prepare.remote(settings, time.time() + LAUNCH_LIMIT_SECONDS), indent=2))
