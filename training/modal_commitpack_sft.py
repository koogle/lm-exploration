"""CPU-only immutable CommitPackFT preparation for the expanded SFT run."""

import json
import re
from pathlib import Path

import modal

app = modal.App("simple-llm-commitpack-sft-data")
volume = modal.Volume.from_name("lm-exploration-training")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tokenizers==0.22.2", "huggingface-hub==1.33.0")
    .env({"PYTHONPATH": "/project"})
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer/structured-code-v1", "/project/tokenizer/structured-code-v1")
)


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0, volumes={"/runs": volume})
def prepare(data_id, excluded_codes):
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download
    from training_data.commitpack_sft import LANGUAGES, REPO, REVISION, build

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", data_id):
        raise ValueError("Invalid data ID")
    volume.reload()

    def download(lang):
        return lang, hf_hub_download(
            REPO, f"data/{lang}/data.jsonl", repo_type="dataset", revision=REVISION, token=False
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(download, LANGUAGES))
    report = build(files, Path("/runs") / data_id, excluded_codes=excluded_codes)
    volume.commit()
    return report


@app.local_entrypoint()
def main(data_id: str, old_dataset: str, report_path: str):
    old = [json.loads(line) for line in Path(old_dataset).read_text().splitlines()]
    excluded = [
        row[field] for row in old if row["split"] == "validation" for field in ("broken", "fixed")
    ]
    report = prepare.remote(data_id, excluded)
    Path(report_path).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
