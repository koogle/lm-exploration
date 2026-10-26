"""Local launcher with system CA trust and a remote deadline supervisor."""

import truststore

truststore.inject_into_ssl()
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("hybrid").resolve().parent))
import modal

from hybrid.modal_study import app, prepare, registry, supervise, train

run = "kda-mixers-" + str(int(time.time()))
deadline = time.time() + 10800
folder = Path("artifacts/results") / run
folder.mkdir(parents=True)
provenance = {
    "run": run,
    "deadline": deadline,
    "base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source_sha256": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("hybrid").glob("*.py")
    },
}
(folder / "launch.json").write_text(json.dumps(provenance, indent=2))
with modal.enable_output(), app.run():
    calls = []
    registry[run + "/calls"] = calls
    supervisor = supervise.spawn(run, deadline)
    while not registry.get(run + "/ready", False):
        time.sleep(1)

    def tracked(fn, *args):
        call = fn.spawn(*args)
        calls.append(call.object_id)
        registry[run + "/calls"] = calls
        provenance["calls"] = calls
        provenance["supervisor"] = supervisor.object_id
        (folder / "launch.json").write_text(json.dumps(provenance, indent=2))
        return call

    try:
        info = tracked(prepare, run, deadline).get()
        (folder / "data.json").write_text(json.dumps(info))
        workers = [
            (v, tracked(train, run, v, deadline))
            for v in ["baseline", "none", "rope", "middle_rope", "rotating"]
        ]
        for variant, call in workers:
            result = call.get()
            (folder / f"{variant}.json").write_text(json.dumps(result, indent=2))
    finally:
        for cid in calls:
            try:
                modal.FunctionCall.from_id(cid).cancel()
            except Exception as error:  # noqa: BLE001
                print("Cancellation failed:", error, flush=True)
        registry[run + "/done"] = True
print("RESULTS", folder)
