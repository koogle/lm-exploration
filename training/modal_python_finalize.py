"""Continue data preparation after deployed workers finish, then freeze if sufficient."""

import json
from pathlib import Path

import modal

app = modal.App("simple-llm-python-finalize")
volume = modal.Volume.from_name("lm-exploration-mixture-data")
FOLDER = "/data/python-124m-5b-20261020-v1"
DOCUMENTATION = "/data/python-documentation-20261021-v1"
SNAPSHOT = "/data/python-124m-5b-20261021-snapshot-v1"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tokenizers==0.22.2")
    .env({"PYTHONPATH": "/project", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer/structured-code-v1", "/project/tokenizer/structured-code-v1")
    .add_local_dir("evaluation", "/project/evaluation")
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=14400,
    retries=0,
    max_containers=1,
    volumes={"/data": volume},
)
def finish():
    import shutil
    import sqlite3
    import tempfile
    import time

    from training_data.python_experiment import Corpus
    from training_data.python_snapshot import (
        exclude_existing_scaffolds,
        freeze,
        merge_code,
        merge_documentation,
    )
    from training_data.subword import digest

    volume.reload()
    folder, snapshot = Path(FOLDER), Path(SNAPSHOT)

    def complete(result):
        pending = folder / "finalization.json.pending"
        pending.write_text(json.dumps(result, indent=2))
        pending.replace(folder / "finalization.json")
        volume.commit()
        return result

    if snapshot.exists():
        manifest = json.loads((snapshot / "manifest.json").read_text())
        for artifact in (manifest["database"], manifest["heldout"]):
            if digest(snapshot / artifact["path"]) != artifact["sha256"]:
                raise ValueError("Existing immutable snapshot failed verification")
        return complete({"stage": "frozen", "snapshot": SNAPSHOT, "manifest": manifest})

    with tempfile.TemporaryDirectory(prefix="python-finalize-") as scratch:
        local = Path(scratch)
        # SQLite backup recovers any interrupted volume transaction before copying.
        source = sqlite3.connect(folder / "raw.sqlite")
        target = sqlite3.connect(local / "raw.sqlite")
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        last_saved = time.monotonic()

        def persist(force=False):
            nonlocal last_saved
            capacity = json.loads((local / "capacity.json").read_text())
            print(
                json.dumps(
                    {
                        "stage": capacity["stage"],
                        "categories": {
                            name: row["unique_train_targets"]
                            for name, row in capacity["categories"].items()
                        },
                    }
                ),
                flush=True,
            )
            if not force and time.monotonic() - last_saved < 120:
                return
            # Callers commit their SQLite transaction before invoking this callback.
            for name in ("raw.sqlite", "capacity.json", "capacity.md"):
                pending = folder / (name + ".pending")
                shutil.copyfile(local / name, pending)
                pending.replace(folder / name)
            volume.commit()
            last_saved = time.monotonic()

        corpus = Corpus(local)
        heldout = json.loads(Path("/project/evaluation/python-tasks.json").read_text())
        with corpus.db:
            corpus.db.executemany(
                "INSERT OR IGNORE INTO excluded_hashes VALUES(?)",
                [(key,) for key in heldout["reference_hashes"]],
            )
            corpus.db.execute(
                "UPDATE documents SET conflict=1 WHERE split='train' AND conflict=0 "
                "AND hash IN (SELECT hash FROM excluded_hashes)"
            )
        corpus.db.close()
        exclude_existing_scaffolds(local, persist)
        merge_code(local, "/data/python-code-supplement-20261021-v1", persist)
        capacity = merge_documentation(local, DOCUMENTATION, persist)
        persist(force=True)
        if not capacity["capacity_sufficient"]:
            return complete({"stage": "more_data_required", "capacity": capacity})
        manifest = freeze(local, local / "snapshot")
        staging = snapshot.with_name(snapshot.name + ".partial")
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(local / "snapshot", staging)
        staging.rename(snapshot)
        return complete({"stage": "frozen", "snapshot": SNAPSHOT, "manifest": manifest})


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/data": volume},
)
def continue_preparation(code_call_id: str, documentation_call_id: str, full_code_call_id: str):
    for call_id in [code_call_id, documentation_call_id, full_code_call_id]:
        modal.FunctionCall.from_id(call_id).get()
    return finish.remote()
