"""Separate convolution changes from recurrence input precision in the long study."""

import truststore

truststore.inject_into_ssl()
import argparse
import json
import subprocess
import time
from pathlib import Path

import modal
from huggingface_hub import HfApi

from hybrid.long_study import train_long
from hybrid.modal_study import app, registry, supervise

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", required=True)
run = parser.parse_args().run
folder = Path("artifacts/results") / run
deadline = json.loads((folder / "launch.json").read_text())["deadline"]
key = run + "-precision-" + str(int(time.time()))
record = {
    "deadline": deadline,
    "variant": "baseline_bf16",
    "seeds": [17, 29],
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
}
with modal.enable_output(), app.run():
    supervisor = supervise.spawn(key, deadline)
    while not registry.get(key + "/ready", False):
        if time.time() > deadline - 300:
            raise TimeoutError("No launch budget")
        time.sleep(1)
    calls = []
    try:
        for seed in [17, 29]:
            assert time.time() < deadline - 300
            call = train_long.spawn(run, "baseline_bf16", seed, deadline)
            calls.append((seed, call))
            registry[key + "/calls"] = [c.object_id for _, c in calls]
            record["calls"] = [{"seed": s, "call": c.object_id} for s, c in calls]
            record["supervisor"] = supervisor.object_id
            (folder / "precision-launch.json").write_text(json.dumps(record, indent=2))
        for seed, call in calls:
            try:
                result = call.get()
            except Exception as error:  # noqa: BLE001
                result = {
                    "variant": "baseline_bf16",
                    "seed": seed,
                    "status": "failed",
                    "error": repr(error),
                }
            (folder / f"baseline_bf16-{seed}.json").write_text(json.dumps(result, indent=2))
    finally:
        for _, call in calls:
            call.cancel()
        registry[key + "/done"] = True
        HfApi().upload_folder(
            repo_id="jfrick/lm-exploration-results",
            repo_type="dataset",
            folder_path=str(folder),
            path_in_repo="training/results/" + run,
            commit_message="Archive precision-matched KDA baseline controls",
        )
