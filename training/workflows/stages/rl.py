"""The RL stage's own inputs (`trainer/rl/` trains it).

- `pool`: an `rl_pool` data ID on the run volume (`training_data.rl_tasks`): the prompts
  of the `sampler` and `reuse` sources and the validation problems.
"""

from dataclasses import asdict

from workflows.checks import run_id

FIELDS = {"pool"}


def inputs(spec):
    return {"pool": run_id(spec.get("pool"), "pool")}


def settings(values):
    from trainer.rl.settings import Settings

    resolved = Settings.from_dict(values)
    if resolved.trajectories.source == "file":
        run_id(resolved.trajectories.data, "trajectories.data")
    return asdict(resolved)
