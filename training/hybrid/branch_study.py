"""Matched ~97M KDA study using the large hybrid branch's prepared GLM shards."""

import json
import time
from pathlib import Path

import modal

from .continuation_study import TOTAL_STEPS, VARIANTS
from .long_study import CONFIG, TOKENS_PER_STEP
from .modal_study import app, image, registry, runs

CONFIG_GLM = dict(
    CONFIG,
    vocab_size=154880,
    eos_id=154820,
    n_embd=448,
    n_head=7,
    n_kv_head=1,
    n_kda_head=7,
    mlp_hidden=1152,
)
RECIPE = json.loads((Path(__file__).parent / "recipes/hybrid-40b-v1.json").read_text())
BRANCH_COMMIT = "24692d1239afbbad1a3acc3dc20d3239a4787f9c"
prepared_volume = modal.Volume.from_name("lm-exploration-hybrid-data")


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=300,
    retries=0,
    volumes={"/data": prepared_volume, "/runs": runs},
)
def preflight(run, deadline):
    from model_hybrid.prepared_data import heldout_index, shard_paths, training_order

    assert time.time() < deadline - 300
    root = Path("/data") / RECIPE["name"]
    result = {"branch_commit": BRANCH_COMMIT, "config": CONFIG_GLM, "recipe": RECIPE, "sources": {}}
    for name, source in RECIPE["sources"].items():
        selected = []
        count = 0
        required = int(TOTAL_STEPS * TOKENS_PER_STEP * source["weight"]) + 16384
        for index in training_order(RECIPE, name):
            bin_path, _, meta_path = shard_paths(root, name, index)
            meta = json.loads(meta_path.read_text())
            assert bin_path.stat().st_size == meta["tokens"] * 4
            assert not meta["heldout"]
            count += meta["tokens"]
            selected.append(index)
            if count >= required:
                break
        assert count >= required, (name, count, required)
        bin_path, _, meta_path = shard_paths(root, name, heldout_index(source))
        heldout = json.loads(meta_path.read_text())
        assert heldout["heldout"] and bin_path.stat().st_size == heldout["tokens"] * 4
        result["sources"][name] = {
            "training_shards": selected,
            "available_tokens": count,
            "required_tokens": required,
            "heldout_tokens": heldout["tokens"],
        }
    folder = Path("/runs") / run
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data.json").write_text(json.dumps(result))
    runs.commit()
    return result


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=32400,
    retries=0,
    max_containers=6,
    volumes={"/runs": runs, "/data": prepared_volume},
)
def train_branch(run, variant, seed, deadline):
    from model_hybrid.long_study import train_impl

    assert variant in VARIANTS or variant == "rope_v_conv"
    return train_impl(
        run,
        variant,
        seed,
        deadline,
        steps=TOTAL_STEPS,
        runtime=31800,
        config=CONFIG_GLM,
        prepared_recipe=RECIPE,
    )


@app.function(image=image, timeout=36000, retries=0, volumes={"/runs": runs})
def coordinate(run, deadline, preparation_call, variants=None):
    workers, outcomes = [], []
    try:
        modal.FunctionCall.from_id(preparation_call).get()
        for seed in [17, 29]:
            for variant in VARIANTS if variants is None else variants:
                assert time.time() < deadline - 300
                call = train_branch.spawn(run, variant, seed, deadline)
                calls = registry[run + "/calls"]
                calls.append(call.object_id)
                registry[run + "/calls"] = calls
                workers.append({"variant": variant, "seed": seed, "call": call.object_id})
                registry[run + "/workers"] = workers
        for worker in workers:
            try:
                result = modal.FunctionCall.from_id(worker["call"]).get()
            except Exception as error:  # noqa: BLE001
                result = {**worker, "status": "failed", "error": repr(error)}
            outcomes.append(result)
        result = {
            "status": "complete"
            if all(x["status"] == "complete" for x in outcomes)
            else "incomplete",
            "workers": workers,
            "outcomes": outcomes,
        }
    except Exception as error:  # noqa: BLE001
        result = {
            "status": "failed",
            "error": repr(error),
            "workers": workers,
            "outcomes": outcomes,
        }
        for worker in workers:
            modal.FunctionCall.from_id(worker["call"]).cancel()
    runs.reload()
    folder = Path("/runs") / run
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "completion.json").write_text(json.dumps(result))
    runs.commit()
    registry[run + "/result"] = result
    registry[run + "/done"] = True
    return result
