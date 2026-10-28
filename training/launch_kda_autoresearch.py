"""Launch the bounded KDA research campaign; all children are remotely supervised."""

import truststore

truststore.inject_into_ssl()

import hashlib
import json
import subprocess
import time
from pathlib import Path

import modal

from hybrid.autoresearch_modal import app, campaign, supervisor
from hybrid.autoresearch_specs import FINAL_STEPS, PROXY, SCREEN_STEPS, VARIANTS
from hybrid.modal_study import registry

run = "kda-autoresearch-" + str(int(time.time()))
deadline = time.time() + 23 * 3600
folder = Path("artifacts/results") / run
folder.mkdir(parents=True)
provenance = {
    "run": run,
    "deadline": deadline,
    "variants": VARIANTS,
    "config": PROXY,
    "seeds": [17, 29],
    "screen_steps": SCREEN_STEPS,
    "confirmation_steps": FINAL_STEPS,
    "base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source_sha256": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("hybrid").rglob("*.py")
    },
}
print("RUN", run, flush=True)
with modal.enable_output(), app.run(detach=True):
    registry[run + "/calls"] = []
    registry[run + "/deadline"] = deadline
    sup = supervisor.spawn(run, deadline)
    provenance["supervisor"] = sup.object_id
    while not registry.get(run + "/ready", False):
        assert time.time() < deadline - 300
        time.sleep(1)
    ctl = campaign.spawn(run, deadline)
    provenance["controller"] = ctl.object_id
    registry[run + "/controller"] = ctl.object_id
    registry[run + "/calls"] = [ctl.object_id]
    (folder / "launch.json").write_text(json.dumps(provenance, indent=2))
    registry[run + "/registered"] = True
    print("CONTROLLER", ctl.object_id, flush=True)
