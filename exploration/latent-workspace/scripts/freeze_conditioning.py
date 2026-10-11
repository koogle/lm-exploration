"""Freeze all validation-selected checkpoints before the one-time test suite."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
names = [
    f"conditioning-{condition}-01"
    for condition in (
        "final",
        "all",
        "best",
        "selected",
        "attention-all",
        "attention-best",
        "attention-selected",
        "jspace",
        "parallel-all",
    )
]
names += ["qwen3-conditioning-final-01", "qwen3-conditioning-all-01"]
records, hashes = {}, {}
for name in names:
    path = ROOT / "results" / f"{name}.json"
    if path.exists():
        value = json.loads(path.read_text())
    else:
        with gzip.open(str(path) + ".gz", "rt") as stream:
            value = json.load(stream)
    report = value["report"]
    if not report.get("completed"):
        raise ValueError(f"training not complete: {name}")
    curve = report["curves"]["196608"]
    best = min(curve["checkpoints"], key=lambda r: r["macro_validation_nll"])
    if best["step"] != curve["best_step"]:
        raise ValueError("validation checkpoint selection mismatch")
    digest = hashlib.sha256((json.dumps(report, indent=2) + "\n").encode()).hexdigest()
    hashes[f"/runs/{name}"] = digest
    records[name] = {
        "source_report_sha256": digest,
        "checkpoint": "size-196608-best.pt",
        "selected_step": best["step"],
        "validation_nll": best["macro_validation_nll"],
        "model": report["config"]["model"],
        "model_revision": report["config"]["model_revision"],
        "variant": report["config"]["variants"][0],
        "selected_layers": report["config"]["selected_layers"],
    }
protocol = {
    "selection": "Minimum macro validation all-masked NLL within each completed training run.",
    "evaluation": "One fixed test evaluation of all nine Qwen2.5 conditions and both Qwen3 conditions; no retuning.",
    "paired_comparisons": [
        ["conditioning-all-01", "conditioning-final-01"],
        ["conditioning-jspace-01", "conditioning-selected-01"],
        ["conditioning-all-01", "conditioning-parallel-all-01"],
        ["qwen3-conditioning-all-01", "qwen3-conditioning-final-01"],
    ],
    "test_windows": 192,
    "domains": ["prose", "code", "translation"],
    "uncertainty": "Small grouped test set; one decoder seed; multiple exploratory comparisons, not confirmatory significance.",
    "checkpoints": records,
}
(ROOT / "results" / "conditioning-test-protocol.json").write_text(
    json.dumps(protocol, indent=2) + "\n"
)
config = {
    "study_eval_only": True,
    "evaluate_test": True,
    "study_eval_parents": list(hashes),
    "study_eval_hashes": hashes,
    "max_seconds": 3300,
}
(ROOT / "configs" / "conditioning-test.json").write_text(json.dumps(config, indent=2) + "\n")
print(f"Frozen {len(records)} checkpoint selections; test has not been evaluated by this script.")
