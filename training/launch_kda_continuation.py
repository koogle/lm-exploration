"""Launch the bounded two-seed baseline/RoPE continuation to 1B positions."""

import truststore

truststore.inject_into_ssl()
import hashlib
import json
import subprocess
import time
from pathlib import Path

import modal
from huggingface_hub import HfApi

from hybrid.continuation_study import (
    SOURCE,
    TOTAL_STEPS,
    VARIANTS,
    prepare_continuation,
    supervise_continuation,
    train_continuation,
)
from hybrid.long_study import CONFIG, TOKENS_PER_STEP
from hybrid.modal_study import app, registry

run = "kda-1b-" + str(int(time.time()))
deadline = time.time() + 36000
folder = Path("artifacts/results") / run
folder.mkdir(parents=True)
provenance = {
    "run": run,
    "source_run": SOURCE,
    "deadline": deadline,
    "config": CONFIG,
    "steps": TOTAL_STEPS,
    "seeds": [17, 29],
    "variants": VARIANTS,
    "token_positions": TOTAL_STEPS * TOKENS_PER_STEP,
    "base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source_sha256": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("hybrid").rglob("*.py")
    },
}
(folder / "launch.json").write_text(json.dumps(provenance, indent=2))
print("RUN", run, "RESULTS", folder, flush=True)
with modal.enable_output(), app.run(detach=True):
    calls = []
    registry[run + "/calls"] = calls
    registry[run + "/deadline"] = deadline
    supervisor = supervise_continuation.spawn(run, deadline)
    while not registry.get(run + "/ready", False):
        if time.time() > deadline - 300:
            raise TimeoutError("Supervisor not ready before deadline")
        time.sleep(1)

    def tracked(fn, *args):
        if time.time() > deadline - 300:
            raise TimeoutError("No launch budget remaining")
        call = fn.spawn(*args)
        calls.append(call.object_id)
        registry[run + "/calls"] = calls
        provenance["calls"] = calls
        provenance["supervisor"] = supervisor.object_id
        (folder / "launch.json").write_text(json.dumps(provenance, indent=2))
        return call

    try:
        prep = tracked(prepare_continuation, run, deadline)
        (folder / "data.json").write_text(json.dumps(prep.get(), indent=2))
        workers = [
            (variant, seed, tracked(train_continuation, run, variant, seed, deadline))
            for seed in [17, 29]
            for variant in VARIANTS
        ]
        provenance["workers"] = [
            {"variant": v, "seed": s, "call": c.object_id} for v, s, c in workers
        ]
        (folder / "launch.json").write_text(json.dumps(provenance, indent=2))
        pending = list(workers)
        while pending:
            for item in list(pending):
                variant, seed, call = item
                try:
                    result = call.get(timeout=0)
                except TimeoutError:
                    continue
                except Exception as error:  # noqa: BLE001
                    result = {
                        "variant": variant,
                        "seed": seed,
                        "status": "failed",
                        "error": repr(error),
                    }
                (folder / f"{variant}-{seed}.json").write_text(json.dumps(result, indent=2))
                pending.remove(item)
            if pending:
                time.sleep(10)
    finally:
        for cid in calls:
            try:
                modal.FunctionCall.from_id(cid).cancel()
            except Exception as error:  # noqa: BLE001
                print("Cancellation failed:", error, flush=True)
        registry[run + "/done"] = True
        HfApi().upload_folder(
            repo_id="jfrick/lm-exploration-results",
            repo_type="dataset",
            folder_path=str(folder),
            path_in_repo="training/results/" + run,
            commit_message="Archive 1B KDA continuation evidence",
        )
print("RESULTS", folder, flush=True)
