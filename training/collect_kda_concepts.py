"""Read-only collection and structured summaries of native KDA probes."""

import truststore

truststore.inject_into_ssl()
import argparse
import json
from pathlib import Path
from statistics import mean

import modal
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("run")
args = parser.parse_args()
folder = Path("artifacts/results") / args.run
folder.mkdir(parents=True, exist_ok=True)
vol = modal.Volume.from_name("lm-exploration-training")
for name in [
    "qualification.json",
    "metrics.json",
    "completion.json",
    "untrained/causal.json",
    "untrained/atlas.json",
    "trained/causal.json",
    "trained/atlas.json",
]:
    try:
        raw = b"".join(vol.read_file(args.run + "/" + name))
    except FileNotFoundError:
        continue
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    if name.endswith("causal.json"):
        data = json.loads(raw)
        groups = {}
        for r in data["rows"]:
            key = (r["task"], r["layer"], r["lag"], r["mode"])
            groups.setdefault(key, []).append(r)
        summary = []
        for (task, layer, lag, mode), rows in groups.items():
            summary.append(
                {
                    "task": task,
                    "layer": layer,
                    "lag": lag,
                    "mode": mode,
                    "n": len(rows),
                    "mean_effect": mean(r["oriented_effect"] for r in rows),
                    "mean_clean_gap": mean(r["clean_gap"] for r in rows),
                    "fraction_positive_effect": mean(r["oriented_effect"] > 0 for r in rows),
                    "paired_correct_fraction": mean(
                        r["donor_prefers_correct"] and r["target_prefers_correct"] for r in rows
                    ),
                }
            )
        (path.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    if name in ["qualification.json", "metrics.json", "completion.json"]:
        d = json.loads(raw)
        print(name, {k: v for k, v in d.items() if k not in ["provenance", "curve", "config"]})
        if d.get("curve"):
            print("latest", d["curve"][-1])
print(
    HfApi().upload_folder(
        repo_id="jfrick/lm-exploration-results",
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo="training/results/" + args.run,
        commit_message="Archive native KDA concept probing evidence",
    )
)
