"""Collect durable Modal continuation metrics and archive progress on HF."""

import truststore

truststore.inject_into_ssl()
import argparse
import json
import time
from pathlib import Path

import modal
from huggingface_hub import HfApi

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run")
args = parser.parse_args()
folder = Path("artifacts/results") / args.run
launch = json.loads((folder / "launch.json").read_text())
registry = modal.Dict.from_name("kda-mixer-study-deadlines")
volume = modal.Volume.from_name("lm-exploration-training")
api = HfApi()
last_upload = 0
while True:
    workers = registry.get(args.run + "/workers", [])
    launch["workers"] = workers
    (folder / "launch.json").write_text(json.dumps(launch, indent=2))
    names = ["data.json", "completion.json"] + [
        f"{w['variant']}-{w['seed']}/metrics.json" for w in workers
    ]
    for name in names:
        target = folder / name
        if name == "data.json" and target.exists():
            continue
        try:
            raw = b"".join(volume.read_file(f"{args.run}/{name}"))
        except FileNotFoundError:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    done = (folder / "completion.json").exists()
    expired = time.time() > launch["deadline"] + 120
    if time.time() - last_upload >= 300 or done or expired:
        api.upload_folder(
            repo_id="jfrick/lm-exploration-results",
            repo_type="dataset",
            folder_path=str(folder),
            path_in_repo="training/results/" + args.run,
            commit_message="Archive KDA 1B continuation progress",
        )
        last_upload = time.time()
        print("Archived", args.run, "workers", len(workers), "complete", done, flush=True)
    if done or expired:
        break
    time.sleep(60)
