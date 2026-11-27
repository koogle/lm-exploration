"""The playground (app.py) on one Modal H100, serving exports from the runs volume.

Run from training/:
    modal run --detach -m model_hybrid.playground.modal_serve \
        --models hybrid/playground/malama-final.json [--hours 4]

`--models` is a JSON file {"name", "description", "models": [{"label", "run", "export",
"chat"?}]}: each export is `<run>/export/<export>` on lm-exploration-hybrid-runs. One
function call starts the Gradio app, opens a public tunnel to it and prints its URL (also
kept in the Modal Dict `malama-playground` under "url"); anyone with the URL can use
the app. The call ends after `--hours` (at most 24): its timeout is that many seconds
with `retries=0`, so Modal stops the GPU then whether or not a client is connected.
Stop it sooner with `modal app stop`.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal
from model_hybrid.image import LAUNCH_LIMIT_SECONDS, build

app = modal.App("lm-exploration-playground")
runs_volume = modal.Volume.from_name("lm-exploration-hybrid-runs")
image = build(("gradio==6.29.1", "jinja2==3.1.6"))
address = modal.Dict.from_name("malama-playground", create_if_missing=True)
PORT = 7860


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=LAUNCH_LIMIT_SECONDS,
    retries=0,
    max_containers=1,
    volumes={"/runs": runs_volume},
)
def serve(settings, deadline):
    """Run the app until `deadline`; the launch sets this call's timeout to the same time."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    path = Path("/tmp/model-settings.json")
    path.write_text(json.dumps(settings))
    process = subprocess.Popen(
        [sys.executable, "-m", "model_hybrid.playground.app"],
        env={**os.environ, "PLAYGROUND_SETTINGS": str(path)},
    )
    try:
        with modal.forward(PORT) as tunnel:
            address["url"] = tunnel.url
            print(json.dumps({"url": tunnel.url, "deadline_unix": deadline}), flush=True)
            while time.time() < deadline and process.poll() is None:
                time.sleep(5)
    finally:
        process.terminate()
        del address["url"]
    if process.returncode not in (None, 0):
        raise RuntimeError(f"The playground exited with status {process.returncode}")


@app.local_entrypoint()
def main(models: str, hours: float = 4.0):
    if not 0 < hours <= 24:
        raise ValueError("--hours is above 0 and at most 24")
    settings = json.loads(Path(models).read_text())
    for entry in settings["models"]:
        if ".." in Path(entry["run"], entry["export"]).parts:
            raise ValueError("run and export are names on the runs volume")
        entry["path"] = f"/runs/{entry.pop('run')}/export/{entry['export']}"
    seconds = int(hours * 3600)
    call = serve.with_options(timeout=seconds).spawn(settings, time.time() + seconds)
    print(json.dumps({"call": call.object_id, "timeout_seconds": seconds}))
