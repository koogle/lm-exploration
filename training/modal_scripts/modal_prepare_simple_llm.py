"""CPU-only bounded preparation on the existing mixture data volume.

No app launches on import. Uses bounded CPU resources for authorized jobs.
"""

import json
import re
from pathlib import Path

import modal

app = modal.App("lm-exploration-simple-llm-preparation")
volume = modal.Volume.from_name("lm-exploration-mixture-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "tokenizers==0.22.2", "requests==2.34.2", "pyarrow==26.0.0", "huggingface-hub==1.33.0"
    )
    .env({"PYTHONPATH": "/project"})
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer", "/project/tokenizer")
)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/data": volume},
)
def prepare(version: str, snapshot: str, max_records: int, recipe: dict):
    from training_data.simple_llm import Stager, validate_recipe

    recipe = validate_recipe(recipe)
    stager = Stager(Path("/data/scholarly-code") / version, recipe)
    try:
        for source in recipe["sources"]:
            count = stager.prepare(source["name"], max_records)
            volume.commit()
            print(json.dumps({"source": source["name"], "scanned_this_call": count}), flush=True)
        return stager.export(Path("/data/scholarly-code") / version / "snapshots" / snapshot)
    finally:
        stager.close()
        volume.commit()


@app.local_entrypoint()
def main(version: str, snapshot: str, recipe: str, max_records: int = 100):
    for value in (version, snapshot):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
            raise ValueError("Version and snapshot must be simple unique names")
    if not 1 <= max_records <= 100_000:
        raise ValueError("max_records must be between 1 and 100000 per source")
    report = prepare.remote(version, snapshot, max_records, json.loads(Path(recipe).read_text()))
    Path("artifacts/results").mkdir(parents=True, exist_ok=True)
    Path(f"artifacts/results/prep-{version}-{snapshot}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
