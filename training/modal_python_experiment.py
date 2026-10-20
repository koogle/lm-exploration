"""Detached, bounded Python corpus preparation; preparation is not training."""

import json
from pathlib import Path

import modal

app = modal.App("simple-llm-python-124m-5b")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
FOLDER = "/data/python-124m-5b-20261020-v1"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("tokenizers==0.22.2", "huggingface_hub==0.36.0")
    .env({"PYTHONPATH": "/project", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_XET": "1"})
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer/structured-code-v1", "/project/tokenizer/structured-code-v1")
)


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=14400,
    retries=0,
    max_containers=1,
    volumes={
        "/data": data_volume,
        "/curated": modal.Volume.from_name("curated-oss-code-repos"),
        "/collected": modal.Volume.from_name("mit-ai-code-collector-repos"),
    },
)
def prepare():
    from training_data.python_experiment import Corpus

    corpus = Corpus(FOLDER)
    try:
        return corpus.repositories(["/curated", "/collected"], data_volume.commit)
    except Exception:
        corpus.report("repository_preparation_failed")
        raise
    finally:
        data_volume.commit()


@app.function(image=image, cpu=2, memory=8192, timeout=900, volumes={"/data": data_volume})
def inventory():
    import sqlite3

    result = {}
    base = Path("/data/scholarly-code")
    for relative in [
        "readiness-20261020-v2/multilingual/raw.sqlite",
        "code-heavy-fresh-20261019/staging/raw.sqlite",
        "readiness-20261020-v2/original-refresh/raw.sqlite",
    ]:
        p = base / relative
        if not p.exists():
            continue
        db = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        result[relative] = {
            "bytes": p.stat().st_size,
        }
        for source in ["code", "code_python", "fineweb_edu", "finemath_4plus", "gutenberg"]:
            row = db.execute(
                "SELECT provenance FROM documents WHERE source=? LIMIT 1", (source,)
            ).fetchone()
            if row:
                result[relative][source + "_provenance"] = json.loads(row[0])
        db.close()
    return result


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=4200,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume},
)
def supplement(database: str):
    from training_data.python_experiment import Corpus

    allowed = {
        "readiness-20261020-v2/multilingual/raw.sqlite",
        "code-heavy-fresh-20261019/staging/raw.sqlite",
        "readiness-20261020-v2/original-refresh/raw.sqlite",
    }
    if database not in allowed:
        raise ValueError("Unexpected input snapshot")
    corpus = Corpus(FOLDER)
    try:
        return corpus.supplement(Path("/data/scholarly-code") / database, data_volume.commit)
    finally:
        data_volume.commit()


@app.local_entrypoint()
def main(action: str = "prepare", database: str = "readiness-20261020-v2/multilingual/raw.sqlite"):
    if action == "inventory":
        result = inventory.remote()
        Path("/tmp/python-data-inventory.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    elif action == "prepare":
        print(json.dumps(prepare.remote(), indent=2))
    elif action == "supplement":
        print(json.dumps(supplement.remote(database), indent=2))
    else:
        raise ValueError("Expected prepare, supplement or inventory")
