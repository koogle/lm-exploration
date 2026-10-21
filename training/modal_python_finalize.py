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
    from training_data.python_snapshot import (
        exclude_existing_scaffolds,
        freeze,
        merge_code,
        merge_documentation,
    )

    volume.reload()
    from training_data.python_experiment import Corpus

    corpus = Corpus(FOLDER)
    heldout = json.loads(Path("/project/evaluation/python-tasks.json").read_text())
    with corpus.db:
        corpus.db.executemany(
            "INSERT OR IGNORE INTO excluded_hashes VALUES(?)",
            [(key,) for key in heldout["reference_hashes"]],
        )
        corpus.db.execute(
            "UPDATE documents SET conflict=1 WHERE split='train' AND hash IN (SELECT hash FROM excluded_hashes)"
        )
    corpus.db.close()
    volume.commit()
    exclude_existing_scaffolds(FOLDER, volume.commit)
    merge_code(FOLDER, "/data/python-code-supplement-20261021-v1", volume.commit)
    capacity = merge_documentation(FOLDER, DOCUMENTATION, volume.commit)
    if not capacity["capacity_sufficient"]:
        result = {"stage": "more_data_required", "capacity": capacity}
    else:
        result = {"stage": "frozen", "snapshot": SNAPSHOT, "manifest": freeze(FOLDER, SNAPSHOT)}
    Path(FOLDER, "finalization.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=900,
    volumes={"/data": volume},
)
def settle_existing_selection(full_code_call_id: str):
    """Close a bounded old-pool selection after its writer has been stopped.

    The separately prepared full Python split supplies code capacity without
    exhausting every older supplemental pool. Preserve accepted rows and holdouts.
    """
    from training_data.python_experiment import TARGETS, Corpus

    prepared = modal.FunctionCall.from_id(full_code_call_id).get()
    if prepared.get("completed_shards") != 53:
        raise ValueError("The bounded full Python split must be complete first")
    volume.reload()
    corpus = Corpus(FOLDER)
    try:
        for source in ("original-refresh", "multilingual"):
            origin = f"/data/scholarly-code/readiness-20261020-v2/{source}/raw.sqlite"
            if not corpus.db.execute(
                "SELECT 1 FROM completed WHERE name=?", ("exclusions:" + origin,)
            ).fetchone():
                raise ValueError("Prior-source heldout ledger is incomplete")
        result = corpus.report("bounded_existing_selection")
        for name in ("cs_math", "general_education", "books"):
            if result["categories"][name]["unique_train_targets"] < TARGETS[name]:
                raise ValueError("Supporting-text selection is incomplete")
        result["note"] = (
            "Existing-pool selection stopped at committed cursors; not an upstream "
            "exhaustion claim. Complete collected snapshots are retained. "
            "The separately pinned full Python split supplies remaining code capacity."
        )
        Path(FOLDER, "existing-selection.json").write_text(json.dumps(result, indent=2))
    finally:
        corpus.db.close()
    volume.commit()
    return result


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
