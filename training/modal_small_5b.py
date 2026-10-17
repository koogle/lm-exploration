"""Deployed, durable prepare -> qualify -> resume-to-5B workflow.

No job launches on import/deployment. Start prepare_and_launch.spawn explicitly.
"""

import json
import time
from pathlib import Path

import modal

from modal_prepare_scholarly import image as cpu_image
from modal_scholarly_pilot import image as gpu_image

# All top-level imports must also be available when Modal imports this module remotely.
imports = ("modal_prepare_scholarly.py", "modal_scholarly_pilot.py", "modal_distributed_train.py")
for name in imports:
    cpu_image = cpu_image.add_local_file(name, f"/project/{name}")
    gpu_image = gpu_image.add_local_file(name, f"/project/{name}")
cpu_image = cpu_image.add_local_dir("trainer", "/project/trainer")
app = modal.App("lm-exploration-modern-10m-5b")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
run_volume = modal.Volume.from_name("lm-exploration-training")


def record(run_id, stage, **details):
    folder = Path("/runs") / run_id
    folder.mkdir(parents=True, exist_ok=True)
    status = {"run_id": run_id, "stage": stage, "updated_unix": time.time(), **details}
    (folder / "workflow.json").write_text(json.dumps(status, indent=2) + "\n")
    run_volume.commit()
    print(json.dumps(status), flush=True)


@app.function(
    image=gpu_image,
    gpu="H100:3",
    cpu=12,
    memory=32768,
    timeout=25200,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def train_segment(run_id, snapshot, settings, provenance, resume="", max_steps=None):
    from modal_distributed_train import worker

    return worker.local(run_id, snapshot, settings, provenance, resume, 21600, max_steps)


@app.function(
    image=cpu_image,
    cpu=1,
    memory=2048,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/runs": run_volume},
)
def continue_training(run_id, snapshot, settings, provenance, previous=None, first_segment=0):
    from trainer.workflow import check_progress, check_resume

    resume = str(Path(previous["final_checkpoint"]).relative_to("/runs")) if previous else ""
    target = settings["target_tokens"]
    try:
        # Two-step start + two-step resume qualify the real raw data path. These
        # use the final schedule from initialization and count toward the 5B run.
        for segment in range(first_segment, min(first_segment + 3, 20)):
            segment_id = f"{run_id}-s{segment:02d}"
            record(
                run_id,
                "training",
                segment=segment_id,
                target_tokens=target,
                completed_tokens=previous["total_loss_tokens"] if previous else 0,
                pending_data_audits=True,
            )
            report = train_segment.remote(
                segment_id, snapshot, settings, provenance, resume, 2 if segment < 2 else None
            )
            if previous:
                check_resume(previous, report)
            done = check_progress(
                report,
                previous["total_loss_tokens"] if previous else 0,
                target,
                settings["global_windows"] * settings["context"],
            )
            run_volume.reload()
            (Path("/runs") / run_id / f"segment-{segment:02d}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            if done:
                record(
                    run_id,
                    "completed",
                    trained_tokens=report["total_loss_tokens"],
                    target_tokens=target,
                    final_checkpoint=report["final_checkpoint"],
                    pending_data_audits=True,
                )
                return report
            previous = report
            resume = str(Path(report["final_checkpoint"]).relative_to("/runs"))
        if segment + 1 >= 20:
            raise RuntimeError("Training reached the bounded session limit without completing")
        record(run_id, "continuing", completed_tokens=previous["total_loss_tokens"], resume=resume)
        call = continue_training.spawn(
            run_id, snapshot, settings, provenance, previous, segment + 1
        )
        return {"continuation_call_id": call.object_id, "resume": resume}
    except Exception as exc:
        record(run_id, "failed", error=f"{type(exc).__name__}: {exc}", resume=resume)
        raise


@app.function(
    image=cpu_image,
    cpu=4,
    memory=8192,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def prepare_and_launch(run_id, version, settings, provenance, resume_preparation=False):
    from training_data.export_raw import export_raw
    from training_data.persistent_stream import PersistentStream
    from training_data.prepared_snapshot import find_prepared_snapshot
    from training_data.raw_corpus import RawCorpus
    from training_data.scholarly import load_recipe

    from trainer.distributed_config import DistributedConfig
    from trainer.workflow import check_capacity

    DistributedConfig(**settings)
    folder = Path("/data/scholarly-code") / version
    corpus = None
    stream = PersistentStream()
    status_path = Path("/runs") / run_id / "workflow.json"
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        if not resume_preparation or prior.get("stage") != "preparing":
            raise ValueError("Only stopped preparation can reuse its workflow run ID")
    elif resume_preparation:
        raise ValueError("No preparation exists to resume")
    try:
        record(run_id, "preparing", target_tokens=settings["target_tokens"], provenance=provenance)
        recipe = load_recipe("/project/tokenizer/recipe-v1.json")
        target = settings["target_tokens"]
        cached = find_prepared_snapshot(folder, recipe, target)
        if cached is not None:
            snapshot = str(cached.parent.relative_to("/data/scholarly-code"))
            manifest = json.loads(cached.read_text())
            record(
                run_id,
                "data_reused",
                snapshot=snapshot,
                target_tokens=target,
                pending_data_audits=manifest.get("pending", []),
            )
            call = continue_training.spawn(run_id, snapshot, settings, provenance)
            return {"training_call_id": call.object_id, "snapshot": snapshot, "data_reused": True}
        corpus = RawCorpus(folder, recipe)
        for _ in range(3):  # Recheck earlier sources after cross-source conflicts.
            for source in sorted(recipe["sources"], key=lambda s: len(s["files"])):
                name = source["name"]
                while True:
                    report = corpus.report(target)
                    status = report["sources"][name]
                    if (
                        status["train_loss_tokens"] >= status["minimum_unique_train_tokens"] * 1.05
                        and status["validation_loss_tokens"] > 0
                    ):
                        break
                    if status["upstream_exhausted"]:
                        if not status["capacity_sufficient"]:
                            raise RuntimeError(
                                f"{name} exhausted before reaching its required share: {status}"
                            )
                        break
                    corpus.prepare(name, reader=stream)
                    report = corpus.report(target)
                    (folder / "capacity.json").write_text(json.dumps(report, indent=2) + "\n")
                    data_volume.commit()
                    record(
                        run_id,
                        "preparing",
                        source=name,
                        source_status=report["sources"][name],
                        supported_mixture_exposures=report["supported_mixture_exposures"],
                        target_tokens=target,
                    )
            report = corpus.report(target)
            try:
                check_capacity(report, target)
                break
            except ValueError:
                pass
        check_capacity(report, target)
        corpus.close()
        corpus = None
        snapshot = f"{version}/snapshots/{run_id}"
        manifest = export_raw(folder, Path("/data/scholarly-code") / snapshot)
        # Preserve pending audit flags. The explicitly scoped experimental_data
        # configuration permits this research run, without claiming audited data.
        manifest["pending"] = [
            "tokenizer split reconciliation",
            "global near deduplication",
            "cross-source parent reconciliation",
            "eligibility and contamination review",
        ]
        manifest["capacity_verified_for_tokens"] = target
        (Path("/data/scholarly-code") / snapshot / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        data_volume.commit()
        record(
            run_id,
            "data_prepared",
            snapshot=snapshot,
            capacity=report,
            pending_data_audits=manifest["pending"],
        )
        call = continue_training.spawn(run_id, snapshot, settings, provenance)
        return {"training_call_id": call.object_id, "snapshot": snapshot}
    except Exception as exc:
        record(run_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stream.close()
        if corpus is not None:
            corpus.close()
        data_volume.commit()
