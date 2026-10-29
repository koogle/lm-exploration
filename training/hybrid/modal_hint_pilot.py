"""N-gram hint pilot on Modal: two hybrid arms, a corpus index job and evaluation.

Run from training/:
    modal run --detach -m model_hybrid.modal_hint_pilot --run-id hint-pilot-01 [--steps N]

`supervise` (CPU) runs the whole pilot. It spawns the corpus job and both arms
(one H100 each), waits, then evaluates the hinted arm with the corpus tables.
It records the deadline and every call id in /runs/hint-pilot/<run>/supervisor.json
and cancels outstanding calls at the deadline, 24 hours after launch. Every
function has retries disabled, refuses to start after the deadline and a
timeout that ends it before the deadline even if the supervisor is gone.
"""

import json
import time
from pathlib import Path

import modal
from model_hybrid.image import LAUNCH_LIMIT_SECONDS, hybrid_image

image = (
    hybrid_image("tokenizers==0.22.2", "huggingface-hub==1.33.0")
    .add_local_dir("trainer", "/project/trainer", ignore=["__pycache__"])
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer", "/project/tokenizer", ignore=["__pycache__"])
)
app = modal.App("lm-exploration-hybrid-hint-pilot")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
run_volume = modal.Volume.from_name("lm-exploration-training")
secrets = [modal.Secret.from_name("simple-llm-hf")]
MANIFEST = "/data/climbmix-code/climbmix-code-v6/snapshots/climbmix-124m-wsd20b-02/manifest.json"
TRAIN_TIMEOUT = 10 * 3600
CORPUS_TIMEOUT = 8 * 3600
EVALUATE_TIMEOUT = 2 * 3600
SUPERVISE_TIMEOUT = LAUNCH_LIMIT_SECONDS - 600


def refuse_late(deadline, timeout):
    """Start only if this call's timeout ends before the launch deadline."""
    if time.time() + timeout > deadline:
        raise TimeoutError("Too close to the launch deadline; refusing work")


def folder(run_id):
    path = Path("/runs/hint-pilot") / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def logger(path):
    def log(line):
        print(line, flush=True)
        with path.open("a") as handle:
            handle.write(line + "\n")

    return log


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=131072,
    timeout=TRAIN_TIMEOUT,
    retries=0,
    secrets=secrets,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def train(run_id, settings, hinted, deadline):
    refuse_late(deadline, TRAIN_TIMEOUT)
    from model_hybrid.hint_pilot import train_arm

    output = folder(run_id) / ("hinted" if hinted else "baseline")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        # Stop training with time left to evaluate and save inside this call's timeout.
        stop = min(deadline, started + TRAIN_TIMEOUT)
        result = train_arm(
            MANIFEST, output, settings, hinted, stop, log=logger(output / "log.jsonl")
        )
    finally:
        run_volume.commit()
    return {k: v for k, v in result.items() if k != "history"}


@app.function(
    image=image,
    cpu=16,
    memory=65536,
    timeout=CORPUS_TIMEOUT,
    retries=0,
    secrets=secrets,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def corpus(run_id, settings, deadline):
    refuse_late(deadline, CORPUS_TIMEOUT)
    from model_hybrid.hint_pilot import corpus_tables

    output = folder(run_id) / "corpus"
    rows = settings["steps"] * settings["rows_per_step"]
    milestones = [rows] + [t // settings["context"] for t in settings["corpus_tokens"]]
    try:
        stop = min(deadline, time.time() + CORPUS_TIMEOUT - 600)
        return corpus_tables(
            MANIFEST, output, settings, milestones, stop, log=logger(output / "log.jsonl")
        )
    finally:
        run_volume.commit()


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=EVALUATE_TIMEOUT,
    retries=0,
    secrets=secrets,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def evaluate(run_id, settings, deadline):
    refuse_late(deadline, EVALUATE_TIMEOUT)
    from model_hybrid.hint_pilot import evaluate_corpora

    run_volume.reload()
    root = folder(run_id)
    tables = sorted((root / "corpus").glob("corpus-*.npz"), key=lambda p: int(p.stem.split("-")[1]))
    results = evaluate_corpora(MANIFEST, root / "hinted", tables, settings)
    (root / "corpus-evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    run_volume.commit()
    return results


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=SUPERVISE_TIMEOUT,
    retries=0,
    volumes={"/runs": run_volume},
)
def supervise(run_id, settings, deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    root = folder(run_id)
    status_path = root / "supervisor.json"
    status = {"run_id": run_id, "deadline_unix": deadline, "settings": settings, "calls": {}}

    def save(**changes):
        status.update(changes)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        run_volume.commit()

    calls = {
        "corpus": corpus.spawn(run_id, settings, deadline),
        "baseline": train.spawn(run_id, settings, False, deadline),
        "hinted": train.spawn(run_id, settings, True, deadline),
    }
    save(calls={k: c.object_id for k, c in calls.items()}, stage="training")
    results = {}
    try:
        for name, call in calls.items():
            try:
                results[name] = call.get(timeout=max(deadline - time.time(), 0))
            except Exception as error:  # noqa: BLE001 -- keep the other arms' results
                results[name] = {"error": repr(error)}
            save(results=results)
        if "error" not in results["hinted"] and "error" not in results["corpus"]:
            call = evaluate.spawn(run_id, settings, deadline)
            calls["evaluate"] = call
            save(calls={k: c.object_id for k, c in calls.items()}, stage="evaluating")
            try:
                results["evaluate"] = call.get(timeout=max(deadline - time.time(), 0))
            except Exception as error:  # noqa: BLE001 -- keep the training results
                results["evaluate"] = {"error": repr(error)}
        save(results=results, stage="done")
    finally:
        # Children outlive their parent unless cancelled explicitly.
        for call in calls.values():
            try:
                call.cancel()
            except Exception as error:  # noqa: BLE001 -- cancel every other call
                print(f"cancel {call.object_id}: {error!r}", flush=True)
    return results


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=EVALUATE_TIMEOUT,
    retries=0,
    secrets=secrets,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def overlap(run_id, deadline):
    refuse_late(deadline, EVALUATE_TIMEOUT)
    from model_hybrid.hint_pilot import overlap_study

    run_volume.reload()
    root = folder(run_id)
    settings = json.loads((root / "supervisor.json").read_text())["settings"]
    output = root / "overlap"
    output.mkdir(exist_ok=True)
    try:
        summary = overlap_study(MANIFEST, root, settings, output)
        (output / "overlap.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        run_volume.commit()
    return summary


@app.local_entrypoint()
def overlap_main(run_id: str):
    """Loss split by corpus overlap for a finished pilot (one H100, under two hours)."""
    print(json.dumps(overlap.remote(run_id, time.time() + LAUNCH_LIMIT_SECONDS), indent=2))


@app.local_entrypoint()
def main(run_id: str, steps: int = 0, corpus_tokens: str = "", validation_rows: int = 0):
    from model_hybrid.hint_pilot import SETTINGS

    settings = dict(SETTINGS)
    if steps:
        settings["steps"] = steps
        settings["warmup_steps"] = min(settings["warmup_steps"], max(steps // 10, 1))
    if corpus_tokens:
        settings["corpus_tokens"] = [int(t) for t in corpus_tokens.split(",")]
    if validation_rows:
        settings["validation_rows"] = validation_rows
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    call = supervise.spawn(run_id, settings, deadline)
    print(json.dumps({"supervisor_call": call.object_id, "deadline_unix": deadline}))
