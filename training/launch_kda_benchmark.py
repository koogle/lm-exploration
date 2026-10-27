"""Run a supervised, warmed same-GPU benchmark against explicit study data."""

import truststore

truststore.inject_into_ssl()
import argparse
import json
import subprocess
import time
from pathlib import Path

import modal

from hybrid.modal_study import app, benchmark, registry, supervise

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--long", action="store_true")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--inspect-dtypes", action="store_true")
mode.add_argument("--precision-control", action="store_true")
mode.add_argument("--rope-kernel", action="store_true")
args = parser.parse_args()
run = args.run
config = variants = None
if args.long or args.inspect_dtypes or args.precision_control or args.rope_kernel:
    from hybrid.long_study import CONFIG, VARIANTS

    config, variants = CONFIG, VARIANTS
    if args.precision_control:
        variants = [*variants, "baseline_bf16"]
kind = "dtypes" if args.inspect_dtypes else "benchmark"
if args.precision_control:
    kind = "benchmark-precision"
if args.rope_kernel:
    kind = "rope-kernel"
key = run + "-" + kind + "-" + str(int(time.time()))
deadline = time.time() + 3600
folder = Path("artifacts/results") / run
with modal.enable_output(), app.run():
    supervisor = supervise.spawn(key, deadline)
    while not registry.get(key + "/ready", False):
        time.sleep(1)
    if args.rope_kernel:
        from hybrid.long_study import rope_kernel_study

        call = rope_kernel_study.spawn(run, deadline)
    elif args.inspect_dtypes:
        from hybrid.long_study import inspect_inputs

        call = inspect_inputs.spawn(deadline)
    else:
        call = benchmark.spawn(run, deadline, config, variants)
    registry[key + "/calls"] = [call.object_id]
    (folder / f"{kind}-launch.json").write_text(
        json.dumps(
            {
                "deadline": deadline,
                "call": call.object_id,
                "supervisor": supervisor.object_id,
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "config": config,
                "variants": variants,
            }
        )
    )
    try:
        (folder / f"{kind}.json").write_text(json.dumps(call.get(), indent=2))
    finally:
        call.cancel()
        registry[key + "/done"] = True
