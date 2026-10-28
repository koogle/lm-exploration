"""Launch two matched local value convolution + learned timescale training runs."""

import truststore

truststore.inject_into_ssl()
import hashlib
import json
import subprocess
import time
from pathlib import Path

import modal

from hybrid.branch_study import BRANCH_COMMIT, CONFIG_GLM, coordinate
from hybrid.continuation_study import TOTAL_STEPS, supervise_continuation
from hybrid.value_timescale_study import qualify_value_timescales

VARIANTS = ["rope_v_timescales"]
from hybrid.long_study import TOKENS_PER_STEP
from hybrid.modal_study import app, registry

run = "kda-value-timescales-1b-" + str(int(time.time()))
deadline = time.time() + 36000
folder = Path("artifacts/results") / run
folder.mkdir(parents=True)
provenance = {
    "run": run,
    "deadline": deadline,
    "config": CONFIG_GLM,
    "steps": TOTAL_STEPS,
    "seeds": [17, 29],
    "variants": VARIANTS,
    "token_positions": TOTAL_STEPS * TOKENS_PER_STEP,
    "initialization": "from_scratch",
    "comparison_runs": ["kda-glm-1b-1793107241", "kda-rope-v-1b-1793165441"],
    "data_branch_commit": BRANCH_COMMIT,
    "base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source_sha256": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("hybrid").rglob("*.py")
    },
}
print("RUN", run, flush=True)
with modal.enable_output(), app.run(detach=True):
    registry[run + "/calls"] = []
    registry[run + "/deadline"] = deadline
    supervisor = supervise_continuation.spawn(run, deadline)
    provenance["supervisor"] = supervisor.object_id
    while not registry.get(run + "/ready", False):
        assert time.time() < deadline - 300
        time.sleep(1)
    for name, function, args in [
        ("preparation", qualify_value_timescales, (run, deadline)),
        ("controller", coordinate, None),
    ]:
        assert time.time() < deadline - 300
        if args is None:
            args = (run, deadline, provenance["preparation"], VARIANTS)
        call = function.spawn(*args)
        calls = registry[run + "/calls"]
        calls.append(call.object_id)
        registry[run + "/calls"] = calls
        provenance[name] = call.object_id
        provenance["calls"] = calls
        (folder / "launch.json").write_text(json.dumps(provenance, indent=2))
    print("REMOTE CONTROLLER", provenance["controller"], flush=True)
