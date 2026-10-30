"""SFT rows of several tasks in one file (`mixture-sft-v1`).

Each row names its `task` and otherwise is a row of that task's dataset, which decides
its prompt, answer and checks. The trainer samples rows uniformly, so each task's share
of training is its share of rows. Validation buckets are `task/bucket`.
"""

from training_data.code_sft import CODE
from training_data.math_sft import MATH
from training_data.sft import COMMON_FIELDS, SFTDataset


class MixtureSFT(SFTDataset):
    FORMAT = "mixture-sft-v1"
    FIELDS = ("task",)

    def __init__(self, tasks):
        self.tasks = tasks

    def task(self, row):
        if row["task"] not in self.tasks:
            raise ValueError(f"Unknown task {row['task']!r} in {row['id']}")
        return self.tasks[row["task"]]

    def prompt(self, row):
        return self.task(row).prompt(row)

    def answer(self, row):
        return self.task(row).answer(row)

    def bucket(self, row):
        return f"{row['task']}/{self.task(row).bucket(row)}"

    def fingerprints(self, row):
        return self.task(row).fingerprints(row)

    def check(self, row):
        task = self.task(row)
        for key in (*COMMON_FIELDS, *task.FIELDS):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing nonempty string: {key}")
        task.check(row)


MIXTURE = MixtureSFT({"math": MATH, "code": CODE})


def combine(datasets, output):
    """Write the rows of each prepared `{task: examples.jsonl}` dataset, tagged with their
    task, to `output/examples.jsonl`, and `report.json` with its SHA256 and counts."""
    import hashlib
    import json
    from pathlib import Path

    output = Path(output)
    lines, counts, sources = [], {}, {}
    for task, path in sorted(datasets.items()):
        if task not in MIXTURE.tasks:
            raise ValueError(f"Unknown task {task!r}")
        raw = Path(path).read_bytes()
        sources[task] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        for line in raw.decode().splitlines():
            if line.strip():
                lines.append(json.dumps({"task": task, **json.loads(line)}) + "\n")
                counts[task] = counts.get(task, 0) + 1
    output.mkdir(parents=True, exist_ok=False)
    path = output / "examples.jsonl"
    path.write_text("".join(lines))
    report = {
        "format": MixtureSFT.FORMAT,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": sources,
        "rows": counts,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
