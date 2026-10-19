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
def prepare(version: str, snapshot: str, max_records: int):
    from training_data.simple_llm import Stager, load_recipe

    recipe = load_recipe("/project/tokenizer/recipe-v1.json")
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
def main(version: str, snapshot: str, max_records: int = 100, convert_from: str = ""):
    for value in (version, snapshot, *([convert_from] if convert_from else [])):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
            raise ValueError("Version and snapshot must be simple unique names")
    if not 1 <= max_records <= 100_000:
        raise ValueError("max_records must be between 1 and 100000 per source")
    report = (
        convert_worker.remote(version, convert_from, snapshot)
        if convert_from
        else prepare.remote(version, snapshot, max_records)
    )
    Path("results").mkdir(exist_ok=True)
    Path(f"results/prep-{version}-{snapshot}.json").write_text(json.dumps(report, indent=2) + "\n")


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/data": volume},
)
def convert_worker(version: str, snapshot: str, destination: str):
    from training_data.token_shards import convert_snapshot

    for value in (version, snapshot, destination):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
            raise ValueError("Version and snapshot names must be simple unique names")
    root = Path("/data/scholarly-code") / version / "snapshots"
    try:
        report = convert_snapshot(root / snapshot / "manifest.json", root / destination)
        return {"snapshot": f"{version}/snapshots/{destination}", "schema": report["schema"]}
    finally:
        volume.commit()
