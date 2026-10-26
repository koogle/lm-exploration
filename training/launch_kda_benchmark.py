"""Run a supervised, warmed same-GPU benchmark against explicit study data."""

import truststore

truststore.inject_into_ssl()
import argparse
import json
import time
from pathlib import Path

import modal

from hybrid.modal_study import app, benchmark, registry, supervise

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
run = parser.parse_args().run
key = run + "-benchmark-" + str(int(time.time()))
deadline = time.time() + 3600
folder = Path("artifacts/results") / run
with modal.enable_output(), app.run():
    supervisor = supervise.spawn(key, deadline)
    while not registry.get(key + "/ready", False):
        time.sleep(1)
    call = benchmark.spawn(run, deadline)
    registry[key + "/calls"] = [call.object_id]
    (folder / "benchmark-launch.json").write_text(
        json.dumps(
            {"deadline": deadline, "call": call.object_id, "supervisor": supervisor.object_id}
        )
    )
    try:
        (folder / "benchmark.json").write_text(json.dumps(call.get(), indent=2))
    finally:
        call.cancel()
        registry[key + "/done"] = True
