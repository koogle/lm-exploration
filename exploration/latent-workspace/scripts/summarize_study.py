"""Render measured study curves without opening test tensors."""

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text())
report = report.get("report", report)
lines = [
    f"# {path.stem}: initial data-budget study",
    "",
    f"Completed: {report.get('completed', False)}. {report['test_status']}.",
    "",
    "All rows use the same backbone, architecture and seed. NLL is all-masked",
    "conditional token loss, not iterative sequence likelihood. Single-seed",
    "screening with 64 validation windows/domain; no significance claim.",
    "",
    "| Unique blocks | Updates | Supervised masked tokens | Train NLL | Validation NLL | Prose | Code | Translation |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for size, curve in report["curves"].items():
    for row in curve["checkpoints"]:
        train = sum(row["training_sample_nll_by_position"]) / len(
            row["training_sample_nll_by_position"]
        )
        domains = row["validation_by_domain"]
        values = " | ".join(f"{domains[d]['nll']:.3f}" for d in ("prose", "code", "translation"))
        lines.append(
            f"| {size} | {row['step']} | {row['masked_supervised_tokens_seen']} | {train:.3f} | {row['macro_validation_nll']:.3f} | {values} |"
        )
    lines.extend(["", f"## Validation continuations: {size} windows", ""])
    for domain, sample in curve.get("validation_samples", {}).items():
        lines.extend([f"**{domain}**", "", "```text", sample["generated"], "```", ""])
    # Keep the table continuous; samples go below it.
table = [line for line in lines if line.startswith("|")]
other = [line for line in lines if not line.startswith("|")]
insert = (
    next(i for i, line in enumerate(other) if line.startswith("##"))
    if any(line.startswith("##") for line in other)
    else len(other)
)
other[insert:insert] = table + [""]
path.with_suffix(".md").write_text("\n".join(other) + "\n")
print(path.with_suffix(".md"))
