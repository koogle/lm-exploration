"""Generic prepare/reuse -> qualify -> continue training workflow.

No job launches on import/deployment. Start prepare_and_launch.spawn explicitly.
"""

import json
import time
from pathlib import Path

import modal

from distributed.image import image as gpu_image
from modal_scripts.modal_prepare_simple_llm import image as cpu_image

# Both images must be able to resolve this module's canonical package imports.
cpu_image = (
    cpu_image.add_local_dir("distributed", "/project/distributed")
    .add_local_dir("workflows", "/project/workflows")
    .add_local_dir("trainer", "/project/trainer")
    .add_local_dir("modal_scripts", "/project/modal_scripts")
)
gpu_image = gpu_image.add_local_dir("modal_scripts", "/project/modal_scripts")
app = modal.App("lm-exploration-training-workflows")
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
    gpu="H100:1",
    cpu=4,
    memory=32768,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def train_segment(run_id, snapshot, bundle, provenance, resume="", max_steps=None):
    from distributed.modal_train import worker
    from workflows.config import resolve_payload

    policy, dataset, config = resolve_payload(bundle)
    settings = config.as_dict()
    return worker.local(
        run_id,
        snapshot,
        settings,
        provenance,
        resume,
        policy.segment_seconds,
        max_steps,
        dataset_root=dataset.root,
    )


@app.function(
    image=cpu_image,
    cpu=1,
    memory=2048,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/runs": run_volume},
)
def continue_training(
    run_id,
    snapshot,
    bundle,
    provenance,
    previous=None,
    first_segment=0,
    initial_resume="",
):
    from workflows.checks import check_progress, check_resume, validate_paths
    from workflows.config import resolve_payload

    policy, dataset, config = resolve_payload(bundle)
    settings = config.as_dict()
    if type(first_segment) is not int or not 0 <= first_segment < policy.max_segments:
        raise ValueError("Invalid first segment")
    validate_paths(f"{run_id}-s{first_segment:02d}", snapshot, initial_resume)
    if dataset.mode == "snapshot" and snapshot != dataset.snapshot:
        raise ValueError("Snapshot differs from the selected dataset")
    resume = (
        str(Path(previous["final_checkpoint"]).relative_to("/runs")) if previous else initial_resume
    )
    target = settings["target_tokens"]
    try:
        # Qualification uses the final schedule and counts toward the target.
        for segment in range(
            first_segment, min(first_segment + policy.segments_per_call, policy.max_segments)
        ):
            segment_id = f"{run_id}-s{segment:02d}"
            record(
                run_id,
                "training",
                segment=segment_id,
                target_tokens=target,
                completed_tokens=previous["total_loss_tokens"] if previous else 0,
                pending_data_audits=provenance.get("pending_data_audits", []),
            )
            report = train_segment.with_options(
                gpu=f"{policy.gpu}:{policy.world_size}",
                cpu=4 * policy.world_size,
                timeout=policy.worker_timeout_seconds,
            ).remote(
                segment_id,
                snapshot,
                bundle,
                provenance,
                resume,
                policy.qualification_steps if segment < policy.qualification_segments else None,
            )
            if previous:
                check_resume(previous, report)
            previous_tokens = previous["total_loss_tokens"] if previous else 0
            if initial_resume and previous is None:
                previous_tokens = report["total_loss_tokens"] - report["session_loss_tokens"]
            done = check_progress(
                report,
                previous_tokens,
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
                    pending_data_audits=provenance.get("pending_data_audits", []),
                )
                return report
            previous = report
            resume = str(Path(report["final_checkpoint"]).relative_to("/runs"))
        if segment + 1 >= policy.max_segments:
            raise RuntimeError("Training reached the bounded session limit without completing")
        record(run_id, "continuing", completed_tokens=previous["total_loss_tokens"], resume=resume)
        call = continue_training.spawn(run_id, snapshot, bundle, provenance, previous, segment + 1)
        return {"continuation_call_id": call.object_id, "resume": resume}
    except Exception as exc:
        record(run_id, "failed", error=f"{type(exc).__name__}: {exc}", resume=resume)
        raise


@app.function(
    image=cpu_image,
    cpu=16,
    memory=16384,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def prepare_and_launch(run_id, version, bundle, provenance, resume_preparation=False):
    import os

    from training_data.cached_shards import CachedShards
    from training_data.export_raw import export_raw
    from training_data.fast_raw_corpus import FastRawCorpus
    from training_data.prepared_snapshot import find_prepared_snapshot

    from workflows.checks import check_capacity, validate_paths
    from workflows.config import resolve_payload

    policy, dataset, config = resolve_payload(bundle)
    settings = config.as_dict()
    validate_paths(run_id + "-s00", f"{version}/snapshots/{run_id}", "")
    if dataset.mode != "prepare_raw":
        raise ValueError("Preparation requires a prepare_raw dataset config")
    data_root = Path("/data") / dataset.root
    folder = data_root / version
    corpus = None
    os.environ["RAYON_NUM_THREADS"] = str(policy.tokenizer_threads)
    run_folder = Path("/runs") / run_id
    status_path = run_folder / "workflow.json"
    selection_path = run_folder / "resolved-workflow.json"
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        if json.loads(selection_path.read_text()) != bundle:
            raise ValueError("Preparation resume requires the identical resolved workflow")
        preparation_failed = (
            prior.get("stage") == "failed"
            and not (folder / "snapshots" / run_id / "manifest.json").exists()
            and not (Path("/runs") / f"{run_id}-s00").exists()
        )
        if not resume_preparation or not (prior.get("stage") == "preparing" or preparation_failed):
            raise ValueError("Only stopped preparation can reuse its workflow run ID")
    elif resume_preparation:
        raise ValueError("No preparation exists to resume")
    try:
        provenance = {**provenance, "resolved_workflow": bundle}
        recipe = bundle["preparation_recipe"]
        target = settings["target_tokens"]
        cached = find_prepared_snapshot(folder, recipe, target)
        # New snapshots retain pending audits. Reject before upstream ingestion
        # or GPU scheduling unless the caller explicitly accepts that status.
        training_ready = json.loads(cached.read_text())["training_ready"] if cached else False
        if not training_ready and not config.allow_unreviewed_data:
            raise ValueError("Dataset has pending audits; allow_unreviewed_data required")
        if not resume_preparation:
            run_folder.mkdir(parents=True, exist_ok=False)
            selection_path.write_text(json.dumps(bundle, indent=2) + "\n")
        record(run_id, "preparing", target_tokens=target, provenance=provenance)
        if cached is not None:
            snapshot = str(cached.parent.relative_to(data_root))
            manifest = json.loads(cached.read_text())
            record(
                run_id,
                "data_reused",
                snapshot=snapshot,
                target_tokens=target,
                pending_data_audits=manifest.get("pending", []),
            )
            provenance = {**provenance, "pending_data_audits": manifest.get("pending", [])}
            call = continue_training.spawn(run_id, snapshot, bundle, provenance)
            return {"training_call_id": call.object_id, "snapshot": snapshot, "data_reused": True}
        corpus = FastRawCorpus(
            folder,
            recipe,
            workers=policy.source_workers,
            batch_size=policy.database_batch_size,
            reader=CachedShards(),
        )
        while True:
            progress = corpus.prepare_parallel(target, max_seconds=policy.preparation_chunk_seconds)
            report = corpus.report(target)
            (folder / "capacity.json").write_text(json.dumps(report, indent=2) + "\n")
            data_volume.commit()
            record(
                run_id,
                "preparing",
                target_tokens=target,
                sources=report["sources"],
                supported_mixture_exposures=report["supported_mixture_exposures"],
                preparation_progress=progress,
                source_workers=policy.source_workers,
                tokenizer_threads=policy.tokenizer_threads,
                database_batch_size=policy.database_batch_size,
            )
            if progress["ready"]:
                break
        check_capacity(report, target)
        corpus.close()
        corpus = None
        snapshot = f"{version}/snapshots/{run_id}"
        manifest = export_raw(folder, data_root / snapshot)
        # Preserve pending audit flags. The explicit allow_unreviewed_data
        # opt-in permits this research run, without claiming audited data.
        manifest["pending"] = [
            "tokenizer split reconciliation",
            "global near deduplication",
            "cross-source parent reconciliation",
            "eligibility and contamination review",
        ]
        manifest["capacity_verified_for_tokens"] = target
        (data_root / snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        data_volume.commit()
        record(
            run_id,
            "data_prepared",
            snapshot=snapshot,
            capacity=report,
            pending_data_audits=manifest["pending"],
        )
        provenance = {**provenance, "pending_data_audits": manifest["pending"]}
        call = continue_training.spawn(run_id, snapshot, bundle, provenance)
        return {"training_call_id": call.object_id, "snapshot": snapshot}
    except Exception as exc:
        record(run_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if corpus is not None:
            corpus.close()
        data_volume.commit()


@app.function(
    image=gpu_image,
    cpu=4,
    memory=32768,
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def seed_phase(parent, seed, manifest, bundle, note=""):
    """Write a continued-pretraining seed in the training image (same PyTorch build)."""
    from trainer.continued_pretraining import migrate_phase
    from workflows.config import resolve_payload

    _, _, config = resolve_payload(bundle)
    try:
        return migrate_phase(
            Path("/runs") / parent, Path("/runs") / seed, Path(manifest), config, note=note
        )
    finally:
        run_volume.commit()


@app.function(
    image=cpu_image,
    cpu=1,
    memory=2048,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def launch_snapshot(run_id, bundle, provenance, resume="", phase_from=""):
    from training_data.mixture import open_mixture

    from workflows.checks import validate_paths
    from workflows.config import resolve_payload

    _, dataset, config = resolve_payload(bundle)
    if dataset.mode != "snapshot":
        raise ValueError("Snapshot launch requires a snapshot dataset config")
    validate_paths(run_id + "-s00", dataset.snapshot, resume or phase_from)
    if resume and phase_from:
        raise ValueError("Choose either an exact resume or a new phase, not both")
    manifest_path = Path("/data") / dataset.root / dataset.snapshot / "manifest.json"
    # Validate format, tokenizer, data review and capacity before any GPU scheduling.
    mixture = open_mixture(
        manifest_path,
        loader=dataset.loader,
        context=config.context,
        allow_unreviewed=config.allow_unreviewed_data,
    )
    try:
        if not config.pilot:
            capacity = min(
                mixture.manifest["max_passes"]
                * s["splits"]["train"]["loss_tokens"]
                * 100
                / s["weight_percent"]
                for s in mixture.sources.values()
            )
            if config.target_tokens > capacity:
                raise ValueError("Snapshot cannot support target mixture within two passes")
    finally:
        mixture.close()
    validation = open_mixture(
        manifest_path,
        loader=dataset.loader,
        context=config.context,
        split="validation",
        allow_unreviewed=config.allow_unreviewed_data,
    )
    validation.close()
    manifest = json.loads(manifest_path.read_text())
    folder = Path("/runs") / run_id
    folder.mkdir(parents=True, exist_ok=False)
    provenance = {
        **provenance,
        "resolved_workflow": bundle,
        "pending_data_audits": manifest.get("pending", []),
    }
    record(run_id, "data_reused", snapshot=dataset.snapshot, provenance=provenance)
    if phase_from:
        # A fresh seed run ID keeps the parent run untouched and passes resume path checks.
        resume = f"{run_id}-seed/step-000000000.pt"
        validate_paths(run_id + "-s00", dataset.snapshot, resume)
        lineage = seed_phase.remote(
            phase_from, resume, str(manifest_path), bundle, note=f"workflow run {run_id}"
        )
        record(run_id, "phase_seeded", seed=resume, parent=phase_from, lineage=lineage)
    call = continue_training.spawn(
        run_id,
        dataset.snapshot,
        bundle,
        provenance,
        initial_resume=resume,
    )
    return {"training_call_id": call.object_id, "snapshot": dataset.snapshot}


@app.local_entrypoint()
def main(
    workflow: str,
    run_id: str,
    version: str = "",
    model_config: str = "",
    dataset_config: str = "",
    resume_preparation: bool = False,
    resume: str = "",
    phase_from: str = "",
):
    import subprocess

    from workflows.checks import validate_paths
    from workflows.config import load_workflow, resolve_payload

    bundle = load_workflow(workflow, model_config=model_config, dataset_config=dataset_config)
    _, dataset, _ = resolve_payload(bundle)
    snapshot = (
        f"{version}/snapshots/{run_id}" if dataset.mode == "prepare_raw" else dataset.snapshot
    )
    validate_paths(run_id + "-s00", snapshot, resume or phase_from)
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "resolved_workflow": bundle,
    }
    if dataset.mode == "prepare_raw":
        if resume or phase_from:
            raise ValueError("Checkpoint resume requires a prepared snapshot dataset")
        print(prepare_and_launch.remote(run_id, version, bundle, provenance, resume_preparation))
    else:
        if version or resume_preparation:
            raise ValueError("Snapshot datasets do not use version or resume_preparation")
        print(launch_snapshot.remote(run_id, bundle, provenance, resume, phase_from))
