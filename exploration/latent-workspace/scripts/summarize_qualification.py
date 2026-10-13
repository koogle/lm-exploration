"""Report a frozen one-time test evaluation with conditional source-group bootstrap."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

path = Path("results/qualification-long-196k-01.json")
r = json.loads(path.read_text())
assert r["completed"] and r["evaluation_split"] == "test" and r["long_rollout_parity_passed"]
rng = np.random.default_rng(42)
samples, intervals = [], {}
for d in ("prose", "code", "translation"):
    clusters = defaultdict(lambda: np.zeros(2))
    for e in r["test_metrics"]["examples"]:
        if e["domain"] == d:
            clusters[e["source_group"]] += [e["accepted_beyond_anchor"] * e["blocks"], e["blocks"]]
    values = np.array(list(clusters.values()))
    chosen = rng.integers(0, len(values), size=(10000, len(values)))
    totals = values[chosen].sum(axis=1)
    scores = totals[:, 0] / totals[:, 1]
    intervals[d] = np.quantile(scores, [0.025, 0.975]).tolist()
    samples.append(scores)
intervals["overall"] = np.quantile(np.mean(samples, axis=0), [0.025, 0.975]).tolist()
rollout = {}
for d in ("prose", "code", "translation"):
    rows = [x for x in r["long_rollouts"] if x["domain"] == d]
    accepted = sum(sum(x["accepted_new_draft_tokens"]) for x in rows)
    calls = sum(x["verification_calls"] for x in rows)
    rollout[d] = {
        "prompts": len(rows),
        "output_tokens": sum(x["output_tokens"] for x in rows),
        "accepted_new_per_verification": accepted / calls if calls else 0,
        "verification_calls": calls,
        "all_outputs_match": all(x["identical_to_target_greedy"] for x in rows),
    }
summary = {
    "checkpoint_step": r["frozen"]["checkpoint_step"],
    "checkpoint_sha256": r["frozen"]["checkpoint_sha256"],
    "test_gate_passed": r["test_gate_passed"],
    "test_NEW": r["test_metrics"]["accepted_beyond_anchor"],
    "test_NLL": r["test_metrics"]["nll"],
    "test_by_domain": r["test_metrics"]["by_domain"],
    "NEW_95pct_CI": intervals,
    "long_rollouts": rollout,
    "usable_test_prompts": r["test_usable_prompts"],
    "reserved_test_prompts": r["test_reserved_prompts"],
    "exclusions": r["test_exclusions"],
    "uncertainty_note": "10000 source-group bootstrap samples, stratified by domain, blocks weighted within domains. Conditional on the frozen selected checkpoint; excludes seed variability and validation-selection uncertainty.",
}
Path("results/qualification-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
labels = ["Prose", "Code", "Translation", "Overall"]
keys = ["prose", "code", "translation", "overall"]
val = [
    r["frozen"]["selected_validation"]["by_domain"][k]["mean_accepted_beyond_anchor"]
    for k in keys[:3]
] + [r["frozen"]["selected_validation"]["accepted_beyond_anchor"]]
test = [r["test_metrics"]["by_domain"][k]["mean_accepted_beyond_anchor"] for k in keys[:3]] + [
    r["test_metrics"]["accepted_beyond_anchor"]
]
x = np.arange(4)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 0.18, val, 0.36, label="Selected validation")
ax.bar(x + 0.18, test, 0.36, label="Sealed test")
ax.errorbar(
    x + 0.18,
    test,
    yerr=[
        [max(0, v - intervals[k][0]) for k, v in zip(keys, test)],
        [max(0, intervals[k][1] - v) for k, v in zip(keys, test)],
    ],
    fmt="none",
    ecolor="black",
    capsize=4,
)
ax.set(
    xticks=x,
    xticklabels=labels,
    ylabel="Consecutive correct NEW tokens / 7",
    ylim=(0, 7),
    title="Frozen 79M drafter: selected validation vs. one-time sealed test",
)
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.text(
    0.5,
    0.01,
    "Error bars: conditional 95% source-group bootstrap. Same selected checkpoint; no test-based retuning.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("results/qualification-test.png", dpi=150)
print(json.dumps(summary, indent=2))
