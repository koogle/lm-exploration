"""Archive frozen-checkpoint route probes and summarize dependent observations honestly."""

import truststore

truststore.inject_into_ssl()
import argparse
import json
from pathlib import Path
from statistics import mean

import modal
from huggingface_hub import HfApi

from hybrid.route_analysis import allocations

parser = argparse.ArgumentParser()
parser.add_argument("run")
args = parser.parse_args()
folder = Path("artifacts/results") / args.run
folder.mkdir(parents=True, exist_ok=True)
vol = modal.Volume.from_name("lm-exploration-training")
for name in ["routes.json", "atlas.json", "completion.json"]:
    try:
        raw = b"".join(vol.read_file(args.run + "/" + name))
    except FileNotFoundError:
        continue
    (folder / name).write_bytes(raw)
    d = json.loads(raw)
    print(
        name,
        "status",
        d.get("status"),
        "rows",
        len(d.get("rows", [])),
        "documents",
        d.get("documents"),
    )
    if name == "completion.json":
        print({k: v for k, v in d.items() if k not in ["precision_comparison"]})
    if name != "routes.json":
        continue
    groups = {}
    for r in d["rows"]:
        key = (r["task"], tuple(r["values"]), r["lag"], r["mode"])
        groups.setdefault(key, []).append(r)
    summary = []
    for (task, values, lag, mode), rows in groups.items():
        entities = {}
        for r in rows:
            entities.setdefault(r["entity"], []).append(r["oriented_effect"])
        entity_means = [mean(v) for v in entities.values()]
        summary.append(
            {
                "task": task,
                "values": values,
                "lag": lag,
                "mode": mode,
                "n_rows": len(rows),
                "n_entities": len(entities),
                "mean_effect": mean(r["oriented_effect"] for r in rows),
                "mean_abs_effect": mean(abs(r["oriented_effect"]) for r in rows),
                "positive_fraction": mean(r["oriented_effect"] > 0 for r in rows),
                "mean_clean_gap": mean(r["clean_gap"] for r in rows),
                "paired_correct_fraction": mean(r["paired_correct"] for r in rows),
                "entity_mean_effect_min": min(entity_means),
                "entity_mean_effect_max": max(entity_means),
            }
        )
    (folder / "analysis.json").write_text(
        json.dumps(
            {
                "groups": summary,
                "route_allocations": allocations(d["rows"]),
                "note": "Directions/templates/lags repeat entities; row counts are not independent sample counts.",
            },
            indent=2,
        )
    )
print(
    HfApi().upload_folder(
        repo_id="jfrick/lm-exploration-results",
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo="training/results/" + args.run,
        commit_message="Archive FP32 KDA route and generalization probes",
    )
)
