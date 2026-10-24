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


# Audits a prepared snapshot still needs; allow_unreviewed_data accepts them explicitly.
PENDING_AUDITS = [
    "tokenizer split reconciliation",
    "global near deduplication",
    "cross-source parent reconciliation",
    "eligibility and contamination review",
]


def track_call(run_id, call_id, kind):
    """Record a call so the deadline supervisor can cancel it."""
    from workflows.deadline import track

    track(Path("/runs/.calls") / run_id, call_id, kind)
    run_volume.commit()


def started_chain(run_id, wait):
    """The training chain this launch already started, from the calls it tracks.

    `continue_training` tracks itself as it starts; after a preempted launch, wait a few
    minutes for a chain spawned just before the preemption to do so.
    """
    from workflows.deadline import tracked_calls

    for attempt in range(4 if wait else 1):
        if attempt:
            time.sleep(60)
        run_volume.reload()
        calls = tracked_calls(Path("/runs/.calls") / run_id, kind="continue_training")
        if calls:
            return calls[0]
    return None


def record(run_id, stage, *, status_file="workflow.json", **details):
    folder = Path("/runs") / run_id
    folder.mkdir(parents=True, exist_ok=True)
    status = {"run_id": run_id, "stage": stage, "updated_unix": time.time(), **details}
    (folder / status_file).write_text(json.dumps(status, indent=2) + "\n")
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
    # Fetches the frozen tokenizer from its private HF repo if the image lacks it.
    secrets=[modal.Secret.from_name("simple-llm-hf")],
)
def train_segment(run_id, snapshot, bundle, provenance, resume, max_steps, max_seconds):
    import os

    from training_data.growing import STATE_ENV, Refresher, volume_paths
    from training_data.raw_token_stream import GROWING_FORMAT

    from distributed.modal_train import worker
    from workflows.config import resolve_payload

    _, dataset, config = resolve_payload(bundle)
    settings = config.as_dict()
    folder = Path("/data") / dataset.root / snapshot
    refresher = None
    if json.loads((folder / "manifest.json").read_text()).get("format") == GROWING_FORMAT:
        # Preparation is still appending; copy each persisted database to local disk.
        database, complete = volume_paths(folder)
        refresher = Refresher(database, complete, "/tmp/growing", data_volume.reload)
        os.environ[STATE_ENV] = str(refresher.state_path)
        # Later reloads start after one interval, once the run has copied its manifest.
        if not refresher.refresh():
            refresher.start()
    try:
        return worker.local(
            run_id,
            snapshot,
            settings,
            provenance,
            resume,
            max_seconds,
            max_steps,
            dataset_root=dataset.root,
        )
    finally:
        if refresher is not None:
            refresher.stop.set()


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
    from workflows.deadline import bounded_timeout, segment_seconds

    policy, dataset, config = resolve_payload(bundle)
    settings = config.as_dict()
    deadline = provenance["deadline_unix"]
    track_call(run_id, modal.current_function_call_id(), "continue_training")
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
            timeout = bounded_timeout(deadline, policy.worker_timeout_seconds)
            call = train_segment.with_options(
                gpu=f"{policy.gpu}:{policy.world_size}",
                cpu=4 * policy.world_size,
                timeout=timeout,
            ).spawn(
                segment_id,
                snapshot,
                bundle,
                provenance,
                resume,
                policy.qualification_steps if segment < policy.qualification_segments else None,
                segment_seconds(timeout, policy.segment_seconds),
            )
            track_call(run_id, call.object_id, "train_segment")
            report = call.get()
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
        call = continue_training.with_options(timeout=bounded_timeout(deadline, 86400)).spawn(
            run_id, snapshot, bundle, provenance, previous, segment + 1
        )
        track_call(run_id, call.object_id, "continue_training")
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
    # Private code sources stream with the HF token.
    secrets=[modal.Secret.from_name("simple-llm-hf")],
)
def prepare_and_launch(run_id, version, bundle, provenance, resume_preparation=False):
    import os
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    from training_data.cached_shards import CachedShards
    from training_data.export_raw import complete_growing, export_growing, export_raw
    from training_data.fast_raw_corpus import FastRawCorpus
    from training_data.prepared_snapshot import find_prepared_snapshot

    from workflows.checks import check_capacity, validate_paths
    from workflows.config import resolve_payload
    from workflows.deadline import bounded_timeout

    policy, dataset, config = resolve_payload(bundle)
    settings = config.as_dict()
    validate_paths(run_id + "-s00", f"{version}/snapshots/{run_id}", "")
    deadline = provenance["deadline_unix"]
    bounded_timeout(deadline, 86400)
    track_call(run_id, modal.current_function_call_id(), "prepare_and_launch")
    if dataset.mode != "prepare_raw":
        raise ValueError("Preparation requires a prepare_raw dataset config")
    data_root = Path("/data") / dataset.root
    folder = data_root / version
    corpus = None
    os.environ["RAYON_NUM_THREADS"] = str(policy.tokenizer_threads)
    run_folder = Path("/runs") / run_id
    status_path = run_folder / "workflow.json"
    selection_path = run_folder / "resolved-workflow.json"
    launch_path = run_folder / "launch.json"
    # With overlap_preparation, training starts on a growing snapshot during preparation;
    # training owns workflow.json from then on and preparation reports in its own file.
    training_path = run_folder / "training.json"
    preparation_status = "workflow.json"
    # Modal restarts a preempted container with the same input. That restart carries
    # this launch's deadline and continues preparation from the persisted database.
    if launch_path.exists():
        restarted = json.loads(launch_path.read_text())["deadline_unix"] == deadline
        resume_preparation = resume_preparation or restarted
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        if json.loads(selection_path.read_text()) != bundle:
            raise ValueError("Preparation resume requires the identical resolved workflow")
        launched = training_path.exists()
        if launched:
            preparation_status = "preparation.json"
            path = run_folder / preparation_status
            prior = json.loads(path.read_text()) if path.exists() else {"stage": "preparing"}
        preparation_failed = prior.get("stage") == "failed" and (
            launched
            or not (folder / "snapshots" / run_id / "manifest.json").exists()
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
        launch_path.write_text(json.dumps({"deadline_unix": deadline}) + "\n")
        record(
            run_id,
            "preparing",
            status_file=preparation_status,
            target_tokens=target,
            provenance=provenance,
        )
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
            call = continue_training.with_options(timeout=bounded_timeout(deadline, 86400)).spawn(
                run_id, snapshot, bundle, provenance
            )
            track_call(run_id, call.object_id, "continue_training")
            return {"training_call_id": call.object_id, "snapshot": snapshot, "data_reused": True}
        # SQLite commits on the network volume are slow; stage on local disk and
        # persist a consistent copy to the volume every `persist_seconds` and when ready.
        local = Path("/tmp/staging") / version
        local.mkdir(parents=True, exist_ok=True)
        if (folder / "raw.sqlite").exists() and not (local / "raw.sqlite").exists():
            shutil.copyfile(folder / "raw.sqlite", local / "raw.sqlite")
        corpus = FastRawCorpus(
            local,
            recipe,
            workers=policy.source_workers,
            batch_size=policy.database_batch_size,
            reader=CachedShards(),
        )
        snapshot = f"{version}/snapshots/{run_id}"
        # Validation reads its first windows in insertion order, so they stay fixed.
        prefix_tokens = (settings["validation_windows_per_source"] + 1) * settings["context"]

        # Sources that have enough data; preparation never extends them again.
        finished = set()

        def persist(report, finished):
            corpus.persist(folder / "raw.sqlite", finished)
            (folder / "capacity.json").write_text(json.dumps(report, indent=2) + "\n")
            data_volume.commit()

        def training_started():
            return training_path.exists() and "training_call_id" in json.loads(
                training_path.read_text()
            )

        def start_training():
            # Record the intent before spawning. A restart that finds an intent without a
            # call ID adopts the chain that launch started, if it registered itself.
            restarted = training_path.exists()
            if not restarted:
                training_path.write_text(json.dumps({"snapshot": snapshot}) + "\n")
                run_volume.commit()
            # A preempted restart may already have exported the snapshot.
            if not (data_root / snapshot / "manifest.json").exists():
                export_growing(folder, data_root / snapshot, PENDING_AUDITS)
                data_volume.commit()
            call_id = started_chain(run_id, wait=restarted)
            if call_id is None:
                call = continue_training.with_options(
                    timeout=bounded_timeout(deadline, 86400)
                ).spawn(
                    run_id, snapshot, bundle, {**provenance, "pending_data_audits": PENDING_AUDITS}
                )
                track_call(run_id, call.object_id, "continue_training")
                call_id = call.object_id
            training_path.write_text(
                json.dumps({"training_call_id": call_id, "snapshot": snapshot}) + "\n"
            )
            run_volume.commit()
            return "preparation.json"

        # Copies run on their own thread so preparation keeps writing meanwhile.
        persister = ThreadPoolExecutor(max_workers=1)
        pending = None  # (copy future, whether training starts once it is persisted)
        persisted = time.monotonic()
        try:
            while True:
                progress = corpus.prepare_parallel(
                    target, max_seconds=policy.preparation_chunk_seconds
                )
                report = corpus.report(target)
                finished |= {
                    name for name, s in report["sources"].items() if FastRawCorpus.enough(s)
                }
                if pending is not None and pending[0].done():
                    copy, launch = pending
                    pending = None
                    copy.result()
                    if launch:
                        preparation_status = start_training()
                launch = (
                    policy.overlap_preparation
                    and not training_started()
                    and all(
                        s["train_loss_tokens"] > 0 and s["validation_loss_tokens"] >= prefix_tokens
                        for s in report["sources"].values()
                    )
                )
                due = time.monotonic() - persisted >= policy.persist_seconds
                if not progress["ready"] and pending is None and (launch or due):
                    pending = (persister.submit(persist, report, frozenset(finished)), launch)
                    persisted = time.monotonic()
                record(
                    run_id,
                    "preparing",
                    status_file=preparation_status,
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
                bounded_timeout(deadline, 86400)
            if pending is not None:
                pending[0].result()
        finally:
            persister.shutdown(wait=True)
        persist(report, set(report["sources"]))
        if policy.overlap_preparation and not training_started():
            preparation_status = start_training()
        check_capacity(report, target)
        corpus.close()
        corpus = None
        if training_path.exists():
            # The final database is persisted; let training start its later passes.
            complete_growing(folder, data_root / snapshot, capacity_verified_for_tokens=target)
            data_volume.commit()
            record(
                run_id,
                "data_prepared",
                status_file=preparation_status,
                snapshot=snapshot,
                capacity=report,
            )
            return json.loads(training_path.read_text())
        manifest = export_raw(folder, data_root / snapshot)
        # Preserve pending audit flags. The explicit allow_unreviewed_data
        # opt-in permits this research run, without claiming audited data.
        manifest["pending"] = PENDING_AUDITS
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
        call = continue_training.with_options(timeout=bounded_timeout(deadline, 86400)).spawn(
            run_id, snapshot, bundle, provenance
        )
        track_call(run_id, call.object_id, "continue_training")
        return {"training_call_id": call.object_id, "snapshot": snapshot}
    except Exception as exc:
        record(
            run_id, "failed", status_file=preparation_status, error=f"{type(exc).__name__}: {exc}"
        )
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
    import os
    import tempfile

    from training_data.growing import STATE_ENV, volume_paths, write_json
    from training_data.mixture import open_mixture
    from training_data.raw_token_stream import GROWING_FORMAT

    from workflows.checks import validate_paths
    from workflows.config import resolve_payload
    from workflows.deadline import bounded_timeout

    _, dataset, config = resolve_payload(bundle)
    deadline = provenance["deadline_unix"]
    bounded_timeout(deadline, 86400)
    track_call(run_id, modal.current_function_call_id(), "launch_snapshot")
    if dataset.mode != "snapshot":
        raise ValueError("Snapshot launch requires a snapshot dataset config")
    validate_paths(run_id + "-s00", dataset.snapshot, resume or phase_from)
    if resume and phase_from:
        raise ValueError("Choose either an exact resume or a new phase, not both")
    manifest_path = Path("/data") / dataset.root / dataset.snapshot / "manifest.json"
    if json.loads(manifest_path.read_text()).get("format") == GROWING_FORMAT:
        # A run that trained during preparation continues once that preparation is done;
        # its segments copy the final database (see train_segment).
        _, complete = volume_paths(manifest_path.parent)
        if not complete.exists():
            raise ValueError("Finish the growing snapshot's preparation with --resume-preparation")
        completion = json.loads(complete.read_text())
        database = manifest_path.parent / Path(completion["database"]["path"]).name
        verified = completion["capacity_verified_for_tokens"]
        if not config.pilot and verified < config.target_tokens:
            raise ValueError("Growing snapshot capacity was verified for fewer tokens")
        state = os.path.join(tempfile.mkdtemp(prefix="growing-"), "state.json")
        write_json(state, {"generation": 1, "path": str(database), "complete": True})
        os.environ[STATE_ENV] = state
    # Validate format, tokenizer, data review and capacity before any GPU scheduling.
    mixture = open_mixture(
        manifest_path,
        loader=dataset.loader,
        context=config.context,
        allow_unreviewed=config.allow_unreviewed_data,
    )
    try:
        if not config.pilot:
            from training_data.mixture import check_capacity

            check_capacity(mixture, config.target_tokens)
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
        call = seed_phase.with_options(timeout=bounded_timeout(deadline, 3600)).spawn(
            phase_from, resume, str(manifest_path), bundle, note=f"workflow run {run_id}"
        )
        track_call(run_id, call.object_id, "seed_phase")
        lineage = call.get()
        record(run_id, "phase_seeded", seed=resume, parent=phase_from, lineage=lineage)
    call = continue_training.with_options(timeout=bounded_timeout(deadline, 86400)).spawn(
        run_id,
        dataset.snapshot,
        bundle,
        provenance,
        initial_resume=resume,
    )
    track_call(run_id, call.object_id, "continue_training")
    return {"training_call_id": call.object_id, "snapshot": dataset.snapshot}


@app.function(
    image=cpu_image,
    cpu=0.25,
    memory=512,
    timeout=86400,
    retries=0,
    volumes={"/runs": run_volume},
)
def enforce_deadline(run_id, deadline_unix):
    """Independent supervisor: cancel every recorded call of this launch at its deadline."""
    from workflows.deadline import tracked_calls

    status = Path("/runs") / run_id / "workflow.json"
    preparation = status.with_name("preparation.json")

    def stage(path):
        return json.loads(path.read_text())["stage"] if path.exists() else None

    while (remaining := deadline_unix - time.time()) > 0:
        time.sleep(min(remaining, 300))
        run_volume.reload()
        # Overlapped preparation can outlast training only if it is still preparing.
        if stage(status) == "completed" and stage(preparation) != "preparing":
            return {"stage": "completed"}
    run_volume.reload()
    cancelled = []
    for call_id in tracked_calls(Path("/runs/.calls") / run_id):
        modal.FunctionCall.from_id(call_id).cancel()
        cancelled.append(call_id)
    record(run_id, "deadline_cancelled", cancelled_calls=cancelled)
    return {"stage": "deadline_cancelled", "cancelled_calls": cancelled}


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
    from workflows.deadline import new_deadline

    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "resolved_workflow": bundle,
        # Every call this launch starts stops by this time.
        "deadline_unix": new_deadline(),
    }
    # The supervisor starts first so no work runs unsupervised.
    supervisor = enforce_deadline.spawn(run_id, provenance["deadline_unix"])
    print(
        json.dumps(
            {"deadline_unix": provenance["deadline_unix"], "supervisor": supervisor.object_id}
        )
    )
    if dataset.mode == "prepare_raw":
        if resume or phase_from:
            raise ValueError("Checkpoint resume requires a prepared snapshot dataset")
        call = prepare_and_launch.spawn(run_id, version, bundle, provenance, resume_preparation)
        print(call.get())
    else:
        if version or resume_preparation:
            raise ValueError("Snapshot datasets do not use version or resume_preparation")
        call = launch_snapshot.spawn(run_id, bundle, provenance, resume, phase_from)
        print(call.get())
