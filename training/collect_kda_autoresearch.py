"""Archive campaign decisions, qualification, and training metrics without checkpoints."""

import truststore

truststore.inject_into_ssl()

import argparse
import json
import time
from pathlib import Path

import modal
from huggingface_hub import HfApi

from hybrid.autoresearch_specs import VARIANTS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run")
parser.add_argument("--once", action="store_true")
args = parser.parse_args()
folder = Path("artifacts/results") / args.run
launch = json.loads((folder / "launch.json").read_text())
reg = modal.Dict.from_name("kda-mixer-study-deadlines")
vol = modal.Volume.from_name("lm-exploration-training")
while True:
    workers = reg.get(args.run + "/workers", [])
    names = [
        "data.json",
        "completion.json",
        "selection.json",
        "qualification-progress.json",
        "screen-progress.json",
        "confirm-progress.json",
        "evaluation-screen-progress.json",
        "evaluation-confirm-progress.json",
    ]
    names += [f"qualification/{v}.json" for v in VARIANTS]
    for w in workers:
        names += [
            f"{w['stage']}/{w['variant']}-{w['seed']}/{n}" for n in ("metrics.json", "failure.json")
        ]
        names += [f"evaluation/{w['stage']}-{w['variant']}-{w['seed']}.json"]
    for name in names:
        try:
            raw = b"".join(vol.read_file(f"{args.run}/{name}"))
        except FileNotFoundError:
            continue
        p = folder / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    (folder / "workers.json").write_text(json.dumps(workers))
    print(
        HfApi().upload_folder(
            repo_id="jfrick/lm-exploration-results",
            repo_type="dataset",
            folder_path=str(folder),
            path_in_repo="training/results/" + args.run,
            commit_message="Archive KDA autoresearch evidence",
        ),
        flush=True,
    )
    if args.once or (folder / "completion.json").exists() or time.time() > launch["deadline"] + 120:
        break
    time.sleep(60)
