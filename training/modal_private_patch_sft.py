"""Prepare private HF patches on Modal CPU; training requires an explicit launch."""

import json
import re
from pathlib import Path

import modal

app = modal.App("simple-llm-private-patch-data")
volume = modal.Volume.from_name("lm-exploration-training")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tokenizers==0.22.2", "huggingface-hub==1.33.0")
    .env({"PYTHONPATH": "/project"})
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer/structured-code-v1", "/project/tokenizer/structured-code-v1")
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=10800,
    retries=0,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": volume},
)
def prepare(data_id, sources):
    from training_data.download_private_patches import download
    from training_data.private_patch_sft import build

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", data_id):
        raise ValueError("Invalid data ID")
    volume.reload()
    output = Path("/runs") / data_id
    if output.exists():
        raise ValueError("Use a fresh data ID")
    source_path = Path("/tmp/private-patch-sources.json")
    source_path.write_text(json.dumps(sources))
    sample = Path("/tmp/private-patch-downloads")
    download(source_path, sample)
    records = [json.loads(line) for line in (sample / "downloaded.jsonl").read_text().splitlines()]
    failures = [r for r in records if "error" in r]
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(records)} source shards failed; training was not launched"
        )
    previous = Path("/runs/repair-commitpack-20261020-v1/examples.jsonl")
    report = build(
        sample / "downloaded.jsonl",
        output,
        reserved_paths=[previous, "/runs/repair-sft-multilang-20261020-r01/examples.jsonl"],
        replay_path=previous,
    )
    (output / "selection.json").write_bytes((sample / "selection.json").read_bytes())
    volume.commit()
    return {k: v for k, v in report.items() if k != "sources"}


@app.local_entrypoint()
def main(sources: str, data_id: str, report_path: str):
    report = prepare.remote(data_id, json.loads(Path(sources).read_text()))
    Path(report_path).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


@app.local_entrypoint()
def launch(data_id: str, run_id: str, report_path: str):
    # The scale worker independently checks the actual volume dataset hash.
    report = json.loads(Path(report_path).read_text())
    if report["total"]["train"] < 60000 or len(report["counts"]["train"]) < 9:
        raise RuntimeError("Corpus did not meet the larger/multilingual launch threshold")
    worker = modal.Function.from_name("simple-llm-repair-sft-scale", "worker")
    settings = {
        "steps": 20000,
        "batch_size": 8,
        "context": 2048,
        "lr": 1e-5,
        "evaluate_every": 500,
        "save_every": 1000,
        "max_seconds": 14400,
    }
    call = worker.spawn(run_id, data_id, run_settings=settings)
    record = {
        "run_id": run_id,
        "data_id": data_id,
        "call_id": call.object_id,
        "status": "submitted",
        "settings": settings,
    }
    Path(report_path).with_suffix(".launch.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
