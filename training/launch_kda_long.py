"""Launch and collect the bounded, two-seed 100M-class mixer study."""

import truststore

truststore.inject_into_ssl()
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import modal
from huggingface_hub import HfApi

from hybrid.long_study import (
    CONFIG,
    STEPS,
    TOKENS_PER_STEP,
    VARIANTS,
    copy_prepared,
    qualify_long,
    train_long,
)
from hybrid.modal_study import app, prepare, registry, supervise

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prepared-run")
args = parser.parse_args()
run = "kda-100m-" + str(int(time.time()))
deadline = (
    json.loads((Path("artifacts/results") / args.prepared_run / "launch.json").read_text())[
        "deadline"
    ]
    if args.prepared_run
    else time.time() + 12600
)
folder = Path("artifacts/results") / run
folder.mkdir(parents=True)
provenance = {
    "run": run,
    "prepared_run": args.prepared_run,
    "deadline": deadline,
    "config": CONFIG,
    "steps": STEPS,
    "seeds": [17, 29],
    "variants": VARIANTS,
    "token_positions": STEPS * TOKENS_PER_STEP,
    "base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source_sha256": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("hybrid").rglob("*.py")
    },
}
(folder / "launch.json").write_text(json.dumps(provenance, indent=2))
print("RUN", run, "RESULTS", folder, flush=True)
with modal.enable_output(), app.run():
    calls = []
    registry[run + "/calls"] = calls
    supervisor = supervise.spawn(run, deadline)
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
        prep = (
            tracked(copy_prepared, args.prepared_run, run, deadline)
            if args.prepared_run
            else tracked(prepare, run, deadline, 101000000, 524288)
        )
        qualification = tracked(qualify_long, run, deadline)
        (folder / "qualification.json").write_text(json.dumps(qualification.get(), indent=2))
        (folder / "data.json").write_text(json.dumps(prep.get(), indent=2))
        workers = [
            (variant, seed, tracked(train_long, run, variant, seed, deadline))
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
            commit_message="Archive 100M KDA mixer study evidence",
        )
print("RESULTS", folder, flush=True)
