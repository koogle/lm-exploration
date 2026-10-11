"""Prepare a fixed-schedule continuation from a finished, time-bounded worker report."""

import argparse
import json
import re
from pathlib import Path

from latent_workspace.dflash import DraftConfig

parser = argparse.ArgumentParser()
parser.add_argument("report", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
report = json.loads(args.report.read_text())
if "training_completed" not in report or report["training_completed"]:
    raise ValueError("Requires a finalized incomplete worker report, not a live or completed fit")
if report.get("stopped") != "bounded time limit; training incomplete":
    raise ValueError("Inspect unexpected failures before continuing")
step = report["curve"][-1]["step"]
goal = report["training_goal_step"]
phase = report["curve"][-1]["phase"]
config = dict(report["config"])
if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", phase):
    raise ValueError("Unsafe parent run identifier")
if not 0 < step < goal <= config.get("schedule_steps", 0):
    raise ValueError("Invalid remaining budget or no fixed global schedule")
config.update(continue_from=f"/runs/{phase}", steps=goal - step)
DraftConfig(**config).validate()
with args.output.open("x") as output:
    output.write(json.dumps(config, indent=2) + "\n")
print(
    f"Prepared {goal - step} remaining updates, steps {step} to {goal}, same global schedule. No job submitted."
)
