"""Paired, within-domain group-bootstrap uncertainty for exploratory size comparisons."""

import json
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
report = json.loads(path.read_text())
report = report.get("report", report)
sizes = sorted(map(int, report["curves"]))
small = report["curves"][str(sizes[0])]["checkpoints"][-1]
large = report["curves"][str(sizes[-1])]["checkpoints"][-1]
if small["step"] != large["step"]:
    raise ValueError("comparison requires matched update budgets")
rng = np.random.default_rng(42)
resampled, domains = [], {}
for domain, (a, b) in report["data"]["domain_ranges"]["validation"].items():
    difference = np.array(large["validation_by_domain"][domain]["example_mean_nll"]) - np.array(
        small["validation_by_domain"][domain]["example_mean_nll"]
    )
    groups = report["data"]["window_groups"]["validation"][a:b]
    unique = sorted(set(groups))
    sums = np.array([sum(d for d, g in zip(difference, groups) if g == group) for group in unique])
    counts = np.array([groups.count(group) for group in unique])
    indices = rng.integers(len(unique), size=(5000, len(unique)))
    sampled = sums[indices].sum(1) / counts[indices].sum(1)
    resampled.append(sampled)
    domains[domain] = {"larger_minus_smaller_nll": float(difference.mean()), "groups": len(unique)}
interval = np.quantile(np.mean(resampled, axis=0), [0.025, 0.975]).tolist()
output = {
    "sizes": sizes,
    "step": small["step"],
    "larger_minus_smaller_nll": large["macro_validation_nll"] - small["macro_validation_nll"],
    "paired_group_bootstrap_95_interval": interval,
    "domains": domains,
    "note": "Negative favors more data. Paired resampling of held-out source groups within domains; one training seed and adaptive exploratory comparisons, not confirmatory inference.",
}
path.with_name(path.stem + "-comparison.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
