"""Launch one bounded small-model pretraining and concept-probing study."""

import truststore

truststore.inject_into_ssl()
import hashlib
import json
import subprocess
import time
from pathlib import Path

import modal

from hybrid.concept_modal import app, registry, supervisor, worker

run = "kda-concepts-" + str(int(time.time()))
deadline = time.time() + 6 * 3600 - 120
folder = Path("artifacts/results") / run
folder.mkdir(parents=True)
provenance = {
    "run": run,
    "deadline": deadline,
    "main_commit": "6d2f9ab",
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "sha256": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("hybrid").rglob("*.py")
    },
}
print("RUN", run, flush=True)
with modal.enable_output(), app.run(detach=True):
    registry[run + "/calls"] = []
    registry[run + "/deadline"] = deadline
    sup = supervisor.spawn(run, deadline)
    provenance["supervisor"] = sup.object_id
    (folder / "launch.json").write_text(json.dumps(provenance, indent=2))
    ready_deadline = time.time() + 300
    while not registry.get(run + "/ready", False):
        if time.time() > min(ready_deadline, deadline - 600):
            sup.cancel()
            raise TimeoutError("Supervisor not ready")
        time.sleep(1)
    call = worker.spawn(run, deadline, provenance)
    registry[run + "/calls"] = [call.object_id]
    provenance.update(supervisor=sup.object_id, worker=call.object_id)
    (folder / "launch.json").write_text(json.dumps(provenance, indent=2))
    registry[run + "/registered"] = True
    print("WORKER", call.object_id, flush=True)
