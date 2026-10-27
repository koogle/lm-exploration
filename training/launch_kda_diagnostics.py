"""Bounded, checkpoint-preserving diagnostics for the completed KDA study."""

import truststore

truststore.inject_into_ssl()
import json
import subprocess
import time
from pathlib import Path

import modal
from huggingface_hub import HfApi

from hybrid.diagnose_kda import diagnose
from hybrid.modal_study import app, registry, supervise

run = "kda-diagnostics-" + str(int(time.time()))
deadline = time.time() + 5400
folder = Path("artifacts/results") / run
folder.mkdir(parents=True)
meta = {
    "run": run,
    "deadline": deadline,
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
}
print("RUN", run, flush=True)
with modal.enable_output(), app.run(detach=True):
    registry[run + "/calls"] = []
    registry[run + "/deadline"] = deadline
    supervisor = supervise.spawn(run, deadline)
    meta["supervisor"] = supervisor.object_id
    while not registry.get(run + "/ready", False):
        assert time.time() < deadline - 300
        time.sleep(1)
    workers = []
    for seed in [17, 29]:
        for variant in ["baseline", "none", "rope_triton"]:
            assert time.time() < deadline - 300
            call = diagnose.spawn(run, variant, seed, deadline)
            workers.append({"variant": variant, "seed": seed, "call": call.object_id})
            registry[run + "/calls"] = [w["call"] for w in workers]
            meta["workers"] = workers
            (folder / "launch.json").write_text(json.dumps(meta, indent=2))
    pending = list(workers)
    while pending:
        for worker in list(pending):
            try:
                result = modal.FunctionCall.from_id(worker["call"]).get(timeout=0)
            except TimeoutError:
                continue
            except Exception as error:  # noqa: BLE001
                result = {**worker, "status": "failed", "error": repr(error)}
            (folder / f"{worker['variant']}-{worker['seed']}.json").write_text(
                json.dumps(result, indent=2)
            )
            print(
                "FINISHED",
                worker["variant"],
                worker["seed"],
                result["status"],
                result.get("error"),
                flush=True,
            )
            pending.remove(worker)
        if pending:
            time.sleep(10)
    registry[run + "/done"] = True
    HfApi().upload_folder(
        repo_id="jfrick/lm-exploration-results",
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo="training/results/" + run,
    )
