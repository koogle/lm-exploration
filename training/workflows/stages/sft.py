"""The SFT stage's own inputs (`trainer/sft.py` trains it).

- `data`: `{"local": path}` (validated locally, uploaded, at most 20 MiB) or
  `{"volume": data-id}` (a prepared folder on the run volume).
- `resume`: an SFT checkpoint to continue with identical identity; `extend: true` also
  permits a larger step target and new cadence or time limit.
- `gpu`: `L40S` (default) or `H100`.
"""

from dataclasses import asdict

from workflows.checks import checkpoint, run_id

FIELDS = {"data", "resume", "extend", "gpu"}
GPUS = ("L40S", "H100")


def inputs(spec):
    data = spec.get("data")
    if not isinstance(data, dict) or len(data) != 1 or not set(data) <= {"local", "volume"}:
        raise ValueError('data must be {"local": path} or {"volume": data-id}')
    if "volume" in data:
        run_id(data["volume"], "data.volume")
    elif not isinstance(data["local"], str) or not data["local"]:
        raise ValueError("data.local must be a path")
    resume = spec.get("resume") or ""
    extend = spec.get("extend", False)
    if type(extend) is not bool:
        raise ValueError("extend must be true or false")
    if extend and not resume:
        raise ValueError("extend requires resume")
    gpu = spec.get("gpu", "L40S")
    if gpu not in GPUS:
        raise ValueError(f"gpu must be one of {GPUS}")
    return {
        "data": dict(data),
        "resume": checkpoint(resume, "resume") if resume else "",
        "extend": extend,
        "gpu": gpu,
    }


def settings(values):
    from trainer.sft import Settings

    return asdict(Settings(**values))
