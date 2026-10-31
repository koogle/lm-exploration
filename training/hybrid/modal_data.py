"""Prepare a hybrid recipe's token shards on Modal CPUs, in training order: the hybrid
family's data stage (`workflows/stages/data.py`).

A workflow's data stage runs `coordinate` under the workflow launch's deadline and
supervisor (`workflows/modal_app.py`), which records every call it starts. The
coordinator builds the contamination screen once (or uses the one the recipe's
`screen` names), then prepares shards (held-out shards first, then by when training
will read them) with at most `parallel` workers. Every call has `retries=0` and a
timeout bounded by the deadline; shards
that already exist are skipped, so a later launch continues the preparation.
Shards land on the `lm-exploration-hybrid-data` volume under `<recipe name>/`,
with `screen.json` and `capacity.json` reports beside them.

    modal run -m model_hybrid.modal_data::smoke [--recipe <json>]

builds the screen and each source's held-out shard only.
"""

import json
import time
from pathlib import Path

import modal

from workflows.modal_app import app, hf_secret, run_volume, track_call
from workflows.modal_app import hybrid_data as volume

secret = hf_secret
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyarrow==26.0.0", "tokenizers==0.22.2", "huggingface-hub==1.33.0")
    # Downloads land in /tmp/upstream, which each shard call clears when it finishes.
    .env(
        {
            "PYTHONPATH": "/project",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_CACHE": "/tmp/upstream",
        }
    )
    .add_local_dir("workflows", "/project/workflows", ignore=["__pycache__"])
    .add_local_dir("data", "/project/training_data", ignore=["__pycache__"])
    .add_local_dir("hybrid", "/project/model_hybrid", ignore=["__pycache__", "kernels"])
)
ROOT = Path("/data")
DEFAULT_RECIPE = "hybrid/recipes/hybrid-40b-v1.json"


def screen_path(recipe):
    """The recipe's contamination screen: its own, or another recipe's named by `screen`."""
    return ROOT / recipe.get("screen", f"{recipe['name']}/screen.npy")


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
    from training_data import packed_shards as data

    from tokenizer import glm as tokenizer

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    folder = ROOT / recipe["name"]
    folder.mkdir(parents=True, exist_ok=True)
    glm = tokenizer.load(tokenizer.tokenizer_file("/tmp/tokenizer"))
    screen, report = data.build_screen(recipe, glm)
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
    from training_data import packed_shards as data

    from tokenizer import glm as tokenizer

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    folder = ROOT / recipe["name"]
    meta_path = data.shard_paths(folder, name, index)[2]
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    os.environ["RAYON_NUM_THREADS"] = "8"
    glm = tokenizer.load(tokenizer.tokenizer_file("/tmp/tokenizer"))
    screen = np.load(screen_path(recipe))
    meta = data.prepare_shard(recipe, name, index, folder, glm, screen, tokenizer.EOS_ID)
    shutil.rmtree("/tmp/upstream", ignore_errors=True)
    volume.commit()
    return meta


def schedule(recipe):
    """(source, index) in the order they are needed: held-out shards, then by expected use."""
    from training_data import packed_shards as data

    # Sources prepared under another recipe (`prepared_in`) are not prepared again.
    own = {n: s for n, s in recipe["sources"].items() if "prepared_in" not in s}
    jobs = [(-1.0, name, data.heldout_index(s)) for name, s in own.items()]
    for name, source in own.items():
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
    volumes={"/data": volume, "/runs": run_volume},
)
def coordinate(launch, recipe, deadline, parallel):
    """Prepare every shard of `recipe` within launch `launch`'s deadline; returns the
    capacity report, or raises if a shard failed."""
    from workflows.deadline import bounded_timeout

    track_call(launch, modal.current_function_call_id(), "coordinate")
    folder = ROOT / recipe["name"]
    if "screen" in recipe and not screen_path(recipe).exists():
        raise FileNotFoundError(f"Screen {recipe['screen']} is not prepared")
    if not screen_path(recipe).exists():
        call = build_screen.with_options(timeout=bounded_timeout(deadline, 3600)).spawn(
            recipe, deadline
        )
        track_call(launch, call.object_id, "build_screen")
        call.get()
    volume.reload()
    pending, running, done, failed = schedule(recipe), {}, {}, []
    while pending or running:
        while pending and len(running) < parallel:
            name, index = pending.pop(0)
            call = prepare.with_options(timeout=bounded_timeout(deadline, 3600)).spawn(
                recipe, name, index, deadline
            )
            track_call(launch, call.object_id, "prepare_shard")
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
        print(
            json.dumps({"done": len(done), "running": len(running), "pending": len(pending)}),
            flush=True,
        )
        time.sleep(10)
    from training_data.packed_shards import source_root

    volume.reload()  # shard workers created the recipe folder after this call started
    capacity = {}
    for name in recipe["sources"]:
        shards = source_root(recipe, folder, name) / name
        metas = [json.loads(p.read_text()) for p in sorted(shards.glob("*.json"))]
        train = [m for m in metas if not m["heldout"]]
        capacity[name] = {
            "train_shards": len(train),
            "train_tokens": sum(m["tokens"] for m in train),
            "heldout_tokens": sum(m["tokens"] for m in metas if m["heldout"]),
            "documents": sum(m["documents"] for m in metas),
            "dropped": {
                k: sum(m["dropped"].get(k, 0) for m in metas)
                for k in ("license", "empty", "short", "contaminated")
            },
            "needed_tokens": round(
                recipe["target_tokens"]
                * (1 - recipe.get("replay", {}).get("fraction", 0))
                * recipe["sources"][name]["weight"]
            ),
        }
    (folder / "capacity.json").write_text(json.dumps(capacity, indent=2) + "\n")
    volume.commit()
    if failed:
        raise RuntimeError(f"{len(failed)} shards failed: {failed[:3]}")
    return capacity


@app.local_entrypoint()
def smoke(recipe: str = DEFAULT_RECIPE):
    """Build the screen and prepare each source's held-out shard (one bounded call each)."""
    from training_data.packed_shards import heldout_index, load_recipe

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
