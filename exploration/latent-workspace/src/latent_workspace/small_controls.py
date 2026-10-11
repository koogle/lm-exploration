"""One bounded exploratory suite; shared frozen corpus and no test metrics."""

import time
from dataclasses import replace
from pathlib import Path

from .study import run_study
from .train import write_json


def run_controls(cfg, output, on_checkpoint=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + cfg.max_seconds
    report = {
        "config": cfg.to_dict(),
        "arms": {},
        "completed": False,
        "test_status": "Previously evaluated test is not used in this exploratory suite.",
    }
    for name, dropout, null in (
        ("baseline", 0.0, False),
        ("dropout", 0.25, False),
        ("null", 0.0, True),
        ("dropout-null", 0.25, True),
    ):
        remaining = int(deadline - time.monotonic())
        if remaining < 120:
            break
        arm = replace(
            cfg,
            study_layer_controls=False,
            layer_dropout=dropout,
            null_attention=null,
            max_seconds=remaining,
        )
        report["arms"][name] = run_study(arm, output / name, on_checkpoint)
        write_json(output / "controls.json", report)
        if on_checkpoint:
            on_checkpoint()
    report["completed"] = len(report["arms"]) == 4 and all(
        r.get("completed", False) for r in report["arms"].values()
    )
    write_json(output / "controls.json", report)
    return report
