"""Execute-verify our original curriculum on CPU before scheduling any training."""

import json
from pathlib import Path

import modal

app = modal.App("simple-llm-repair-data")
runtime_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "gcc", "g++", "rustc", "golang-go", "default-jdk-headless", "nodejs", "node-typescript"
    )
    .env({"PYTHONPATH": "/project"})
)

image = runtime_image.add_local_dir(
    "data", "/project/training_data", ignore=["raw", "cache", "__pycache__"]
)


@app.function(image=image, cpu=2, memory=4096, timeout=1800, retries=0, max_containers=9)
def verify(language):
    import tempfile

    from training_data.repair_curriculum import execute, rows

    records = rows(language)
    for row in records:
        results = {}
        for variant in ("broken", "fixed"):
            with tempfile.TemporaryDirectory() as folder:
                results[variant] = execute(language, row[variant], row["cases"], folder)
        if results["broken"]["passed"] or not results["fixed"]["passed"]:
            raise ValueError(f"Oracle qualification failed: {row['id']}: {results}")
        row["verification"] = results
    return records


@app.local_entrypoint()
def main(output: str):
    import tempfile

    from training_data.repair_curriculum import LANGUAGES, rows, write_dataset
    from training_data.repair_sft import prepare
    from training_data.subword import digest, load_frozen

    with tempfile.TemporaryDirectory() as folder:
        candidate = Path(folder) / "candidate.jsonl"
        write_dataset(candidate, [row for lang in LANGUAGES for row in rows(lang)])
        prepare(candidate, load_frozen(), 1024)
    records = [row for result in verify.map(LANGUAGES) for row in result]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_dataset(destination, records)
    splits, data_hash = prepare(destination, load_frozen(), 1024)
    report = {
        "sha256": data_hash,
        "generator_sha256": digest("data/repair_curriculum.py"),
        "languages": list(LANGUAGES),
        "counts": {k: len(v) for k, v in splits.items()},
        "verified_pairs": len(records),
        "synthetic": True,
        "limitation": "Small arithmetic repair curriculum; not repository repair capability.",
    }
    destination.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
