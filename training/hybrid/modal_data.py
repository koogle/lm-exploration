"""Prepare the hybrid recipe's token shards on Modal CPUs, in training order.

Run from training/:
    modal run --detach -m model_hybrid.modal_data --run-id <id> [--recipe <json>]

The launch fixes one deadline 24 hours ahead (less a supervisor margin). An
independent supervisor cancels every recorded call at that deadline. The
coordinator builds the contamination screen once, then prepares shards (held-out
shards first, then by when training will read them) with at most `parallel`
workers. Every call has `retries=0` and a timeout bounded by the deadline; shards
that already exist are skipped, so a later launch continues the preparation.
Shards land on the `lm-exploration-hybrid-data` volume under `<recipe name>/`,
with `screen.json` and `capacity.json` reports beside them.
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("lm-exploration-hybrid-data")
volume = modal.Volume.from_name("lm-exploration-hybrid-data", create_if_missing=True)
secret = modal.Secret.from_name("simple-llm-hf")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyarrow==26.0.0", "tokenizers==0.22.2", "huggingface-hub==1.33.0")
    .env({"PYTHONPATH": "/project", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_file("workflows/__init__.py", "/project/workflows/__init__.py")
    .add_local_file("workflows/deadline.py", "/project/workflows/deadline.py")
    .add_local_dir("hybrid", "/project/model_hybrid", ignore=["__pycache__", "kernels"])
)
ROOT = Path("/data")
DEFAULT_RECIPE = "hybrid/recipes/hybrid-40b-v1.json"


def download(repo, file, revision):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo, file, revision=revision, repo_type="dataset", local_dir="/tmp/upstream"
    )


def calls_folder(run_id):
    return ROOT / ".calls" / run_id


def status(run_id, **fields):
    path = ROOT / ".runs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"updated_unix": time.time(), **fields}, indent=2) + "\n")
    volume.commit()


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3600,
    retries=0,
    secrets=[secret],
    volumes={"/data": volume},
)
def build_screen(recipe, deadline):
    import numpy as np
    from model_hybrid import data, tokenizer

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    folder = ROOT / recipe["name"]
    folder.mkdir(parents=True, exist_ok=True)
    glm = tokenizer.load(tokenizer.tokenizer_file("/tmp/tokenizer"))
    screen, report = data.build_screen(recipe, glm, download)
    np.save(folder / "screen.npy", screen)
    (folder / "screen.json").write_text(
        json.dumps(
            {
                "ngram": recipe["contamination"]["ngram"],
                "unique": len(screen),
                "benchmarks": report,
            },
            indent=2,
        )
        + "\n"
    )
    volume.commit()
    return len(screen)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3600,
    retries=0,
    secrets=[secret],
    volumes={"/data": volume},
)
def prepare(recipe, name, index, deadline):
    import os
    import shutil

    import numpy as np
    from model_hybrid import data, tokenizer

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    folder = ROOT / recipe["name"]
    meta_path = data.shard_paths(folder, name, index)[2]
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    os.environ["RAYON_NUM_THREADS"] = "8"
    glm = tokenizer.load(tokenizer.tokenizer_file("/tmp/tokenizer"))
    screen = np.load(folder / "screen.npy")
    meta = data.prepare_shard(recipe, name, index, folder, glm, screen, download, tokenizer.EOS_ID)
    shutil.rmtree("/tmp/upstream", ignore_errors=True)
    volume.commit()
    return meta


def schedule(recipe):
    """(source, index) in the order they are needed: held-out shards, then by expected use."""
    from model_hybrid import data

    jobs = [(-1.0, name, data.heldout_index(s)) for name, s in recipe["sources"].items()]
    for name, source in recipe["sources"].items():
        scale = source["estimated_tokens_per_shard"] / source["weight"]
        jobs += [(p * scale, name, i) for p, i in enumerate(data.training_order(recipe, name))]
    return [(name, index) for _, name, index in sorted(jobs)]


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=86400,
    retries=0,
    secrets=[secret],
    volumes={"/data": volume},
)
def coordinate(run_id, recipe, deadline, parallel):
    from workflows.deadline import bounded_timeout, track

    track(calls_folder(run_id), modal.current_function_call_id(), "coordinate")
    folder = ROOT / recipe["name"]
    if not (folder / "screen.npy").exists():
        call = build_screen.with_options(timeout=bounded_timeout(deadline, 3600)).spawn(
            recipe, deadline
        )
        track(calls_folder(run_id), call.object_id, "build_screen")
        volume.commit()
        call.get()
    volume.reload()
    pending, running, done, failed = schedule(recipe), {}, {}, []
    while pending or running:
        while pending and len(running) < parallel:
            name, index = pending.pop(0)
            call = prepare.with_options(timeout=bounded_timeout(deadline, 3600)).spawn(
                recipe, name, index, deadline
            )
            track(calls_folder(run_id), call.object_id, f"prepare {name} {index}")
            running[call.object_id] = (name, index, call)
        volume.commit()
        for call_id, (name, index, call) in list(running.items()):
            try:
                meta = call.get(timeout=0)
            except (TimeoutError, modal.exception.TimeoutError):
                continue
            except Exception as error:  # noqa: BLE001 - recorded; other shards continue.
                failed.append({"source": name, "index": index, "error": str(error)[:500]})
                meta = None
            del running[call_id]
            if meta:
                done[f"{name}/{index}"] = meta
        status(
            run_id,
            stage="preparing",
            done=len(done),
            running=len(running),
            pending=len(pending),
            failed=failed,
        )
        time.sleep(10)
    capacity = {}
    for name in recipe["sources"]:
        metas = [m for m in done.values() if m["source"] == name]
        train = [m for m in metas if not m["heldout"]]
        capacity[name] = {
            "train_shards": len(train),
            "train_tokens": sum(m["tokens"] for m in train),
            "heldout_tokens": sum(m["tokens"] for m in metas if m["heldout"]),
            "documents": sum(m["documents"] for m in metas),
            "dropped": {
                k: sum(m["dropped"][k] for m in metas) for k in ("license", "empty", "contaminated")
            },
            "needed_tokens": round(recipe["target_tokens"] * recipe["sources"][name]["weight"]),
        }
    (folder / "capacity.json").write_text(json.dumps(capacity, indent=2) + "\n")
    status(
        run_id,
        stage="completed" if not failed else "completed_with_failures",
        done=len(done),
        failed=failed,
        capacity=capacity,
    )
    return capacity


@app.function(
    image=image, cpu=0.25, memory=512, timeout=86400, retries=0, volumes={"/data": volume}
)
def enforce_deadline(run_id, deadline):
    """Independent supervisor: cancel every recorded call of this launch at its deadline."""
    from workflows.deadline import tracked_calls

    path = ROOT / ".runs" / f"{run_id}.json"
    while (remaining := deadline - time.time()) > 0:
        time.sleep(min(remaining, 300))
        volume.reload()
        if path.exists() and json.loads(path.read_text())["stage"].startswith("completed"):
            return {"stage": "completed"}
    volume.reload()
    cancelled = []
    for call_id in tracked_calls(calls_folder(run_id)):
        modal.FunctionCall.from_id(call_id).cancel()
        cancelled.append(call_id)
    return {"stage": "deadline_cancelled", "cancelled": cancelled}


@app.local_entrypoint()
def main(run_id: str, recipe: str = DEFAULT_RECIPE, parallel: int = 48):
    import re

    from model_hybrid.data import load_recipe

    from workflows.deadline import new_deadline

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run ID")
    resolved = load_recipe(recipe)
    deadline = new_deadline()
    supervisor = enforce_deadline.spawn(run_id, deadline)
    call = coordinate.with_options(timeout=int(deadline - time.time())).spawn(
        run_id, resolved, deadline, parallel
    )
    print(
        json.dumps(
            {
                "deadline_unix": deadline,
                "supervisor": supervisor.object_id,
                "coordinator": call.object_id,
            }
        )
    )


@app.local_entrypoint()
def smoke(recipe: str = DEFAULT_RECIPE):
    """Build the screen and prepare each source's held-out shard (one bounded call each)."""
    from model_hybrid.data import heldout_index, load_recipe

    from workflows.deadline import new_deadline

    resolved = load_recipe(recipe)
    deadline = new_deadline()
    print("screen n-grams", build_screen.remote(resolved, deadline))
    calls = [
        prepare.spawn(resolved, name, heldout_index(source), deadline)
        for name, source in resolved["sources"].items()
    ]
    for call in calls:
        print(json.dumps(call.get(timeout=3600)))
