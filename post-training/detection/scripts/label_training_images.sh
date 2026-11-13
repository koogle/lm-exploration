#!/bin/sh
# Label prepared training images with the teacher until both quotas are met or the
# budget is spent; re-runs `label` as preparation adds images (it skips labelled ones).
# Run from post-training/detection/:
#   sh scripts/label_training_images.sh CONFIG OUT BUDGET_USD
set -u
config=$1
out=$2
budget=$3
while :; do
    .venv/bin/python -m parallel_detection.teacher label --config "$config" \
        --out "$out" --budget-usd "$budget"
    done=$(CONFIG="$config" OUT="$out" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
config = json.loads(Path(os.environ["CONFIG"]).read_text())
labelled = {}
(model,) = config["models"]
path = Path(os.environ["OUT"]) / (model.replace("/", "--") + ".jsonl")
for line in path.read_text().splitlines() if path.exists() else []:
    record = json.loads(line)
    if "error" not in record:
        source = record["image_id"].split("/")[0]
        labelled[source] = labelled.get(source, 0) + 1
spent = sum(json.loads(line)["cost"] for line in path.read_text().splitlines()) if path.exists() else 0
quotas = config["per_source"]
print(int(all(labelled.get(s, 0) >= q for s, q in quotas.items())), round(spent, 2), labelled)
PY
)
    echo "$(date -u +%H:%M) complete-and-spent: $done"
    case "$done" in 1*) exit 0 ;; esac
    spent=$(echo "$done" | cut -d' ' -f2)
    awk "BEGIN{exit !($spent >= $budget - 0.5)}" && exit 0
    sleep 600
done
