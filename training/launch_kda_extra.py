"""Additional middle-layer RoPE control using the pilot's identical data."""

import truststore

truststore.inject_into_ssl()
import argparse
import json
import time
from pathlib import Path

import modal

from hybrid.modal_study import app, registry, supervise, train

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", required=True)
run = parser.parse_args().run
if not run.replace("-", "").replace("_", "").isalnum():
    raise ValueError("Invalid run ID")
key = run + "-middle-conv-" + str(int(time.time()))
deadline = time.time() + 10800
folder = Path("artifacts/results") / run
with modal.enable_output(), app.run():
    supervisor = supervise.spawn(key, deadline)
    while not registry.get(key + "/ready", False):
        time.sleep(1)
    call = train.spawn(run, "middle_rope_conv", deadline)
    registry[key + "/calls"] = [call.object_id]
    (folder / "extra-launch.json").write_text(
        json.dumps(
            {"deadline": deadline, "call": call.object_id, "supervisor": supervisor.object_id}
        )
    )
    try:
        result = call.get()
        (folder / "middle_rope_conv.json").write_text(json.dumps(result, indent=2))
    finally:
        call.cancel()
        registry[key + "/done"] = True
