"""Run bounded recurrence probes against an existing diagnostic sample."""

import truststore

truststore.inject_into_ssl()
import argparse
import json
import time
from pathlib import Path

import modal

from hybrid.modal_study import app, registry, supervise
from hybrid.probe_kda_state import probe_states

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run")
args = parser.parse_args()
folder = Path("artifacts/results") / args.run
key = args.run + "/state-probe"
deadline = time.time() + 2400
with modal.enable_output(), app.run(detach=True):
    registry[key + "/calls"] = []
    registry[key + "/deadline"] = deadline
    watch = supervise.spawn(key, deadline)
    while not registry.get(key + "/ready", False):
        assert time.time() < deadline - 120
        time.sleep(1)
    call = probe_states.spawn(args.run, deadline)
    registry[key + "/calls"] = [call.object_id]
    (folder / "state-probe-launch.json").write_text(
        json.dumps({"deadline": deadline, "supervisor": watch.object_id, "call": call.object_id})
    )
    try:
        result = call.get()
        (folder / "state-probes.json").write_text(json.dumps(result, indent=2))
    finally:
        registry[key + "/done"] = True
