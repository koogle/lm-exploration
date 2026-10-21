"""Bounded 3-GPU equivalence/resume checks for the new data pipeline."""

import json
from pathlib import Path

import modal

from modal_streaming_train import image

image = image.add_local_file("modal_streaming_train.py", "/project/modal_streaming_train.py")
app = modal.App("lm-exploration-overlap-checks")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
run_volume = modal.Volume.from_name("lm-exploration-training")


@app.function(
    image=image,
    gpu="L40S:3",
    cpu=16,
    memory=32768,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def check(world_size=3, tag="three01"):
    import torch

    from distributed_training.config import DistributedConfig
    from distributed_training.modal_train import worker

    base = DistributedConfig(
        world_size=world_size,
        global_windows=2 * world_size,
        microbatch=2,
        pilot=True,
        allow_unreviewed_data=True,
        target_tokens=5_000_000,
        warmup_tokens=100000,
        n_layer=8,
        n_head=3,
        n_embd=192,
        checkpoint_every=64,
        evaluate_every=64,
    ).as_dict()
    snapshot = "raw-gpu-qualification-20261017/snapshots/fixture"
    provenance = {"qualification": "frozen synthetic raw data; no completed-run weights modified"}

    def run(name, settings, steps, resume=""):
        name = name.replace("20261018-01", f"20261018-{tag}")
        return worker.local(name, snapshot, settings, provenance, resume, 900, steps)

    sync = run("overlap-sync-20261018-01", base, 128)
    settings = {**base, "prefetch_batches": 8}
    uninterrupted = run("overlap-prefetch-20261018-01", settings, 128)
    first = run("overlap-interrupted-20261018-01", settings, 64)
    resumed = run(
        "overlap-resumed-20261018-01",
        settings,
        64,
        str(Path(first["final_checkpoint"]).relative_to("/runs")),
    )
    states = [
        torch.load(r["final_checkpoint"], map_location="cpu", weights_only=False)
        for r in (sync, uninterrupted, resumed)
    ]
    delta_sync = max(
        float((states[0]["model"][k] - states[1]["model"][k]).abs().max())
        for k in states[0]["model"]
    )
    delta_resume = max(
        float((states[1]["model"][k] - states[2]["model"][k]).abs().max())
        for k in states[1]["model"]
    )
    assert delta_sync < 2e-6 and delta_resume < 2e-6
    assert states[0]["data"] == states[1]["data"] == states[2]["data"]
    assert first["final_validation"] == resumed["initial_validation"]
    result = {
        "world_size": world_size,
        "gpu": torch.cuda.get_device_name(),
        "sync_vs_prefetch_max_weight_delta": delta_sync,
        "uninterrupted_vs_resume_max_weight_delta": delta_resume,
        "data_cursor_exact": True,
        "resume_validation_exact": True,
        "reports": {"sync": sync, "prefetch": uninterrupted, "resumed": resumed},
    }
    (Path("/runs") / f"overlap-checks-20261018-{tag}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    run_volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=8192,
    timeout=1200,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def check_one_gpu():
    return check.local(world_size=1, tag="one01")


@app.local_entrypoint()
def main(single: bool = False):
    report = check_one_gpu.remote() if single else check.remote()
    suffix = "one01" if single else "three01"
    Path(f"results/overlap-checks-20261018-{suffix}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "reports"}))
