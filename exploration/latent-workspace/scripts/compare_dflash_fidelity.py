"""Paired source-group uncertainty for completed, fixed-corpus fidelity studies."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+", type=Path, help="Historical control first")
parser.add_argument("--output", type=Path, default=Path("results/dflash-fidelity-comparison.json"))
args = parser.parse_args()
if args.output.resolve() in [p.resolve() for p in args.reports]:
    raise ValueError("Output would overwrite a source report")
runs = [json.loads(p.read_text()) for p in args.reports]
for run in runs:
    if not run.get("training_completed"):
        raise ValueError("Only completed runs may be compared")
    if run["curve"][-1]["step"] != runs[0]["curve"][-1]["step"]:
        raise ValueError("Unequal final update budgets")
    if (
        run["curve"][-1]["supervised_target_exposures"]
        != runs[0]["curve"][-1]["supervised_target_exposures"]
    ):
        raise ValueError("Unequal supervision budgets")
    if run["split_sequence_sha256"] != runs[0]["split_sequence_sha256"]:
        raise ValueError("Train/validation token identities differ")
examples = [{e["group"]: e for e in r["selected_validation"]["examples"]} for r in runs]
if any(e.keys() != examples[0].keys() for e in examples):
    raise ValueError("Validation source groups differ")
metrics = ("accepted_beyond_anchor", "nll")
comparisons = []
for index, run in enumerate(runs[1:], 1):
    rng = np.random.default_rng(20261011)
    draws = np.zeros((10000, 2))
    point = np.zeros(2)
    for domain in ("prose", "code", "translation"):
        groups = sorted(g for g, e in examples[0].items() if e["domain"] == domain)
        weights = np.array([examples[0][g]["blocks"] for g in groups])
        if any(examples[index][g]["blocks"] != examples[0][g]["blocks"] for g in groups):
            raise ValueError("Fixed anchors differ")
        delta = np.array(
            [[examples[index][g][k] - examples[0][g][k] for k in metrics] for g in groups]
        )
        selected = rng.integers(len(groups), size=(10000, len(groups)))
        w = weights[selected]
        draws += (delta[selected] * w[..., None]).sum(1) / w.sum(1)[:, None] / 3
        point += (delta * weights[:, None]).sum(0) / weights.sum() / 3
    expected = [run["selected_validation"][k] - runs[0]["selected_validation"][k] for k in metrics]
    np.testing.assert_allclose(point, expected, atol=1e-5, rtol=1e-5)
    comparisons.append(
        {
            "run": args.reports[index].name,
            "difference_from_control": {
                k: {
                    "estimate": float(point[i]),
                    "conditional_bootstrap_95_percent": np.quantile(
                        draws[:, i], [0.025, 0.975]
                    ).tolist(),
                }
                for i, k in enumerate(metrics)
            },
        }
    )
summary = {
    "control": args.reports[0].name,
    "reports": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.reports},
    "comparisons": comparisons,
    "note": "10000 paired source-group bootstrap draws stratified by domain, block-weighted within domains. Conditional on validation-selected checkpoints; excludes selection bias, seed variability and multiple-comparison adjustment. Not test results.",
}
args.output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
