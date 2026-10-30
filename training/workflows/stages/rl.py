"""The RL stage's own inputs (`trainer/rl/` trains it).

- `pool`: an `rl_pool` data ID on the run volume (`training_data.rl_tasks`): the prompts
  of the `sampler` and `reuse` sources and the validation problems.
- `gpus`: H100s for the run (1 to 8, default 1), one rank each (`trainer/rl/ranks.py`).
"""

from dataclasses import asdict

from workflows.checks import run_id

FIELDS = {"pool", "gpus"}


def inputs(spec):
    gpus = spec.get("gpus", 1)
    if type(gpus) is not int or not 1 <= gpus <= 8:
        raise ValueError("gpus must be an integer from 1 to 8")
    return {"pool": run_id(spec.get("pool"), "pool"), "gpus": gpus}


def settings(values):
    from trainer.rl.settings import Settings

    resolved = Settings.from_dict(values)
    if resolved.trajectories.source == "file":
        run_id(resolved.trajectories.data, "trajectories.data")
    return asdict(resolved)
