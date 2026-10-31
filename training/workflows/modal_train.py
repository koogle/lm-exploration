"""Run workflows (`workflows/plan.py`): their stages in order, from data preparation to
post-training, as one launch with one deadline.

    modal run --detach -m workflows.modal_train::main --workflow <workflow.json> --run-id <id> \\
        [--start <stage>] [--end <stage>] [--checkpoint <run>/step-NNNNNNNNN.pt]
    modal run -m workflows.modal_train::stop --run-id <id>

`run_workflow` runs the selected stages; one deadline supervisor and one stop request
serve the whole launch. Pretraining and mid-training stages of every family share one
session chain (`continue_training`); a family supplies its session runner (`train_segment` for
subword GPTs, `hybrid_segment` for the hybrid, each `torchrun -m <family trainer>`
in a fresh `<run id>-sNN` folder resuming the previous session's final checkpoint)
and its launch preflight. Workflow state (status, session reports, tracked calls)
lives on the `lm-exploration-training` volume for every family; each family's run
folders live on its own runs volume.

No job launches on import/deployment.
"""

import json
import subprocess
import time
from pathlib import Path

import modal

# Registers the hybrid data stage's functions on the app.
from model_hybrid import modal_data
from model_hybrid.image import image as hybrid_image

from distributed.image import image as gpu_image
from workflows import modal_post_train
from workflows.modal_app import (
    STOP_FILE,
    app,
    control,
    cpu_image,
    data_volume,
    enforce_deadline,
    hf_secret,
    hybrid_data,
    hybrid_runs,
    record,
    run_volume,
    track_call,
    watch,
)

# Audits a prepared snapshot still needs; allow_unreviewed_data accepts them explicitly.
PENDING_AUDITS = [
    "tokenizer split reconciliation",
    "global near deduplication",
    "cross-source parent reconciliation",
    "eligibility and contamination review",
]


def spawn_training(run_id, snapshot, bundle, provenance, **options):
    """Start a training chain within this launch's deadline and track it."""
    from workflows.deadline import bounded_timeout

    call = continue_training.with_options(
        timeout=bounded_timeout(provenance["deadline_unix"], 86400)
    ).spawn(run_id, snapshot, bundle, provenance, **options)
    track_call(provenance["workflow_run_id"], call.object_id, f"continue_training:{run_id}")
    return call.object_id


def started_chain(launch, run_id, wait):
    """The training chain this launch already started, from the calls it tracks.

    `continue_training` tracks itself as it starts; after a preempted launch, wait a few
    minutes for a chain spawned just before the preemption to do so.
    """
    from workflows.deadline import tracked_calls

    for attempt in range(4 if wait else 1):
        if attempt:
            time.sleep(60)
        run_volume.reload()
        calls = tracked_calls(Path("/runs/.calls") / launch, kind=f"continue_training:{run_id}")
        if calls:
            return calls[0]
    return None


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
    from training_data.growing import serve

    from distributed.modal_train import worker
    from workflows.config import resolve_payload

    _, dataset, config = resolve_payload(bundle)
    watching = watch(provenance["workflow_run_id"], 30)
    try:
        # While preparation still appends, copy each persisted database to local disk.
        with serve(Path("/data") / dataset.root / snapshot, data_volume.reload):
            return worker.local(
                run_id,
                snapshot,
                config.as_dict(),
                provenance,
                resume,
                max_seconds,
                max_steps,
                dataset_root=dataset.root,
                stop_file=STOP_FILE,
            )
    finally:
        watching.set()


@app.function(
    image=hybrid_image,
    gpu="H100:8",
    cpu=32,
    memory=262144,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/data": hybrid_data, "/runs": hybrid_runs},
    secrets=[hf_secret],
)
def hybrid_segment(run_id, snapshot, bundle, provenance, resume, max_steps, max_seconds):
    """One hybrid session in `/runs/<run_id>` (torchrun -m model_hybrid.train); its report.

    The run config owns the data recipe (prepared on the hybrid data volume) and
    the token target; `resume` names the previous session's final checkpoint.
    """
    from workflows.checks import validate_paths
    from workflows.config import resolve_payload

    policy, _, settings = resolve_payload(bundle)
    validate_paths(run_id, None, resume)
    output = Path("/runs") / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "run-config.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={policy.world_size}",
        "-m",
        "model_hybrid.train",
        "--config",
        str(output / "run-config.json"),
        "--run-dir",
        str(output),
        "--data-root",
        "/data",
        "--stop-unix",
        str(time.time() + max_seconds),
        "--stop-file",
        STOP_FILE,
    ]
    if resume:
        command += ["--resume", str(Path("/runs") / resume)]
    if max_steps:
        command += ["--max-steps", str(max_steps)]
    hybrid_runs.commit()
    watching = watch(provenance["workflow_run_id"], 60, (hybrid_data.reload, hybrid_runs.commit))
    try:
        with (output / "torchrun.log").open("w") as log:
            code = subprocess.run(
                command, cwd="/project", stdout=log, stderr=subprocess.STDOUT, check=False
            ).returncode
    finally:
        watching.set()
        hybrid_runs.commit()
    if code:
        tail = (output / "torchrun.log").read_text()[-3000:]
        raise RuntimeError(f"Training failed; inspect {output}/torchrun.log: {tail}")
    return json.loads((output / "report.json").read_text())


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
    from workflows.config import resolve_payload, step_tokens
    from workflows.deadline import bounded_timeout, segment_seconds

    policy, dataset, config = resolve_payload(bundle)
    settings = config if policy.family == "hybrid" else config.as_dict()
    session = {"subword": train_segment, "hybrid": hybrid_segment}[policy.family]
    # Hybrid evaluation kernels are not bit-reproducible across processes.
    tolerance = 1e-3 if policy.family == "hybrid" else 0.0
    deadline = provenance["deadline_unix"]
    launch = provenance["workflow_run_id"]
    track_call(launch, modal.current_function_call_id(), f"continue_training:{run_id}")
    if type(first_segment) is not int or not 0 <= first_segment < policy.max_segments:
        raise ValueError("Invalid first segment")
    validate_paths(f"{run_id}-s{first_segment:02d}", snapshot, initial_resume)
    if dataset and dataset.mode == "snapshot" and snapshot != dataset.snapshot:
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
                completed_tokens=previous["tokens"] if previous else 0,
                pending_data_audits=provenance.get("pending_data_audits", []),
            )
            timeout = bounded_timeout(deadline, policy.worker_timeout_seconds)
            call = session.with_options(
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
            track_call(launch, call.object_id, "train_segment")
            report = call.get()
            if report["stop_reason"] == "stop_requested":
                record(
                    run_id,
                    "stopped",
                    trained_tokens=report["tokens"],
                    final_checkpoint=report["final_checkpoint"],
                )
                return report
            if previous:
                check_resume(previous, report, tolerance)
            previous_tokens = previous["tokens"] if previous else 0
            if initial_resume and previous is None:
                previous_tokens = report["tokens"] - report["session_tokens"]
            done = check_progress(
                report, previous_tokens, target, step_tokens(policy.family, settings)
            )
            run_volume.reload()
            (Path("/runs") / run_id / f"segment-{segment:02d}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            if done:
                record(
                    run_id,
                    "completed",
                    trained_tokens=report["tokens"],
                    target_tokens=target,
                    final_checkpoint=report["final_checkpoint"],
                    pending_data_audits=provenance.get("pending_data_audits", []),
                )
                return report
            previous = report
            resume = str(Path(report["final_checkpoint"]).relative_to("/runs"))
        if segment + 1 >= policy.max_segments:
            raise RuntimeError("Training reached the bounded session limit without completing")
        record(run_id, "continuing", completed_tokens=previous["tokens"], resume=resume)
        call_id = spawn_training(
            run_id, snapshot, bundle, provenance, previous=previous, first_segment=segment + 1
        )
        return {"continuation_call_id": call_id, "resume": resume}
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
def prepare_and_launch(
    run_id, version, bundle, provenance, resume_preparation=False, launch_training=True
):
    """Prepare the stage's raw snapshot (or reuse a compatible one) and, with
    `launch_training`, start its training chain: on the growing snapshot with
    `overlap_preparation`, else once the data is prepared."""
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
    overlap = policy.overlap_preparation and launch_training
    validate_paths(run_id + "-s00", f"{version}/snapshots/{run_id}", "")
    deadline = provenance["deadline_unix"]
    bounded_timeout(deadline, 86400)
    track_call(
        provenance["workflow_run_id"], modal.current_function_call_id(), "prepare_and_launch"
    )
    if dataset.mode != "prepare_raw":
        raise ValueError("Preparation requires a prepare_raw dataset config")
    data_root = Path("/data") / dataset.root
    folder = data_root / version
    corpus = None
    os.environ["RAYON_NUM_THREADS"] = str(policy.tokenizer_threads)
    run_folder = Path("/runs") / run_id
    selection_path = run_folder / "resolved-workflow.json"
    launch_path = run_folder / "launch.json"
    # Overlapped training owns workflow.json, so that preparation reports in its own file.
    status_file = "preparation.json" if overlap else "workflow.json"
    status_path = run_folder / status_file
    snapshot = f"{version}/snapshots/{run_id}"
    manifest_path = data_root / snapshot / "manifest.json"
    # Modal restarts a preempted container with the same input. That restart carries
    # this launch's deadline and continues preparation from the persisted database.
    if launch_path.exists():
        restarted = json.loads(launch_path.read_text())["deadline_unix"] == deadline
        resume_preparation = resume_preparation or restarted
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        if json.loads(selection_path.read_text()) != bundle:
            raise ValueError("Preparation resume requires the identical resolved workflow")
        # Overlapped training runs on its own chain, so its preparation can always resume.
        preparation_failed = prior.get("stage") == "failed" and (
            overlap or not manifest_path.exists() and not (Path("/runs") / f"{run_id}-s00").exists()
        )
        if not resume_preparation or not (prior.get("stage") == "preparing" or preparation_failed):
            raise ValueError("Only stopped preparation can reuse its workflow run ID")
    elif resume_preparation:
        raise ValueError("No preparation exists to resume")

    def report_status(stage, **details):
        record(run_id, stage, status_file=status_file, **details)

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
        report_status("preparing", target_tokens=target, provenance=provenance)
        if cached is not None:
            reused = str(cached.parent.relative_to(data_root))
            pending = json.loads(cached.read_text()).get("pending", [])
            report_status(
                "data_reused", snapshot=reused, target_tokens=target, pending_data_audits=pending
            )
            if not launch_training:
                return {"snapshot": reused, "data_reused": True}
            call_id = spawn_training(
                run_id, reused, bundle, {**provenance, "pending_data_audits": pending}
            )
            return {"training_call_id": call_id, "snapshot": reused, "data_reused": True}
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
        # Validation reads its first windows in insertion order, so they stay fixed.
        prefix_tokens = (settings["validation_windows_per_source"] + 1) * settings["context"]

        def persist(report, paths=(folder / "raw.sqlite",)):
            corpus.persist(paths)
            (folder / "capacity.json").write_text(json.dumps(report, indent=2) + "\n")
            data_volume.commit()

        def start_training():
            """Adopt or spawn the training chain on the growing snapshot.

            The growing manifest marks the launch. A restart that finds it adopts the chain
            that launch started if it registered itself, and spawns one otherwise.
            """
            restarted = manifest_path.exists()
            if not restarted:
                export_growing(folder, manifest_path.parent, PENDING_AUDITS)
                data_volume.commit()
            return started_chain(
                provenance["workflow_run_id"], run_id, wait=restarted
            ) or spawn_training(
                run_id, snapshot, bundle, {**provenance, "pending_data_audits": PENDING_AUDITS}
            )

        training_call = None
        if overlap and manifest_path.exists():
            training_call = start_training()
        # Copies run on their own thread so preparation keeps writing meanwhile.
        persister = ThreadPoolExecutor(max_workers=1)
        pending = None  # (copy future, whether training starts once it lands)
        persisted = time.monotonic()
        try:
            while True:
                progress = corpus.prepare_parallel(
                    target, max_seconds=policy.preparation_chunk_seconds
                )
                report = corpus.report(target)
                if pending is not None and pending[0].done():
                    copy, launch = pending
                    pending = None
                    copy.result()
                    if launch:
                        training_call = start_training()
                launch = (
                    overlap
                    and training_call is None
                    and all(
                        s["train_loss_tokens"] > 0 and s["validation_loss_tokens"] >= prefix_tokens
                        for s in report["sources"].values()
                    )
                )
                due = time.monotonic() - persisted >= policy.persist_seconds
                if not progress["ready"] and pending is None and (launch or due):
                    pending = (persister.submit(persist, report), launch)
                    persisted = time.monotonic()
                report_status(
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
                bounded_timeout(deadline, 86400)
            if pending is not None:
                pending[0].result()
        finally:
            persister.shutdown(wait=True)
        check_capacity(report, target)
        if overlap:
            # One final copy for the version and one frozen into the snapshot.
            persist(report, (folder / "raw.sqlite", manifest_path.parent / "raw.sqlite"))
            corpus.close()
            corpus = None
            training_call = training_call or start_training()
            # The final database is frozen; training starts its later passes.
            complete_growing(manifest_path.parent, capacity_verified_for_tokens=target)
            pending_audits = PENDING_AUDITS
        else:
            persist(report)
            corpus.close()
            corpus = None
            manifest = export_raw(folder, manifest_path.parent)
            # Preserve pending audit flags. The explicit allow_unreviewed_data
            # opt-in permits this research run, without claiming audited data.
            manifest["pending"] = pending_audits = PENDING_AUDITS
            manifest["capacity_verified_for_tokens"] = target
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        data_volume.commit()
        report_status(
            "data_prepared", snapshot=snapshot, capacity=report, pending_data_audits=pending_audits
        )
        if not launch_training:
            return {"snapshot": snapshot}
        if training_call is None:
            training_call = spawn_training(
                run_id, snapshot, bundle, {**provenance, "pending_data_audits": pending_audits}
            )
        return {"training_call_id": training_call, "snapshot": snapshot}
    except Exception as exc:
        report_status("failed", error=f"{type(exc).__name__}: {exc}")
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
    # Preflight loads the frozen tokenizer from its private HF repo if the image lacks it.
    secrets=[modal.Secret.from_name("simple-llm-hf")],
)
def launch_snapshot(run_id, bundle, provenance, resume="", phase_from=""):
    from training_data.growing import COMPLETE_RECORD, GROWING_FORMAT
    from training_data.mixture import open_mixture

    from workflows.checks import validate_paths
    from workflows.config import resolve_payload
    from workflows.deadline import bounded_timeout

    _, dataset, config = resolve_payload(bundle)
    deadline = provenance["deadline_unix"]
    bounded_timeout(deadline, 86400)
    track_call(provenance["workflow_run_id"], modal.current_function_call_id(), "launch_snapshot")
    if dataset.mode != "snapshot":
        raise ValueError("Snapshot launch requires a snapshot dataset config")
    validate_paths(run_id + "-s00", dataset.snapshot, resume or phase_from)
    if resume and phase_from:
        raise ValueError("Choose either an exact resume or a new phase, not both")
    manifest_path = Path("/data") / dataset.root / dataset.snapshot / "manifest.json"
    if json.loads(manifest_path.read_text()).get("format") == GROWING_FORMAT:
        # A run that trained during preparation continues on its frozen final database.
        completion = manifest_path.parent / COMPLETE_RECORD
        if not completion.exists():
            raise ValueError(
                "Finish the growing snapshot's preparation: launch its workflow run again"
            )
        verified = json.loads(completion.read_text())["capacity_verified_for_tokens"]
        if not config.pilot and verified < config.target_tokens:
            raise ValueError("Growing snapshot capacity was verified for fewer tokens")
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
        track_call(provenance["workflow_run_id"], call.object_id, "seed_phase")
        lineage = call.get()
        record(run_id, "phase_seeded", seed=resume, parent=phase_from, lineage=lineage)
    call_id = spawn_training(run_id, dataset.snapshot, bundle, provenance, initial_resume=resume)
    return {"training_call_id": call_id, "snapshot": dataset.snapshot}


@app.function(
    image=cpu_image,
    cpu=1,
    memory=2048,
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/runs": run_volume, "/hybrid-data": hybrid_data, "/hybrid-runs": hybrid_runs},
)
def launch_hybrid(run_id, bundle, provenance, resume="", init_from=""):
    """Preflight a hybrid stage (recipe prepared, resume checkpoint or `init_from` export
    present), then start its session chain."""
    from workflows.checks import validate_paths
    from workflows.config import resolve_payload
    from workflows.deadline import bounded_timeout

    if init_from:
        bundle = {**bundle, "model": {**bundle["model"], "init_from": f"/runs/{init_from}"}}
    _, _, settings = resolve_payload(bundle)
    bounded_timeout(provenance["deadline_unix"], 3600)
    track_call(provenance["workflow_run_id"], modal.current_function_call_id(), "launch_hybrid")
    validate_paths(run_id + "-s00", None, resume)
    recipe = json.loads(Path(settings["recipe"]).read_text())
    if not (Path("/hybrid-data") / recipe["name"]).is_dir():
        raise ValueError(f"Prepare the {recipe['name']} recipe on the hybrid data volume first")
    if resume and not (Path("/hybrid-runs") / resume).exists():
        raise ValueError(f"No checkpoint {resume} on the hybrid runs volume")
    if init_from and not (Path("/hybrid-runs") / init_from / "model.safetensors").exists():
        raise ValueError(f"No export {init_from} on the hybrid runs volume")
    folder = Path("/runs") / run_id
    folder.mkdir(parents=True, exist_ok=False)
    provenance = {**provenance, "resolved_workflow": bundle}
    record(run_id, "launched", recipe=recipe["name"], resume=resume, provenance=provenance)
    call_id = spawn_training(run_id, None, bundle, provenance, initial_resume=resume)
    return {"training_call_id": call_id, "recipe": recipe["name"]}


# Seconds before the launch deadline at which the orchestrator stops starting stages.
STAGE_RESERVE_SECONDS = 1800


def stage_status(run_id):
    """The status a stage's run records in `/runs/<run_id>/workflow.json`, or None."""
    run_volume.reload()
    path = Path("/runs") / run_id / "workflow.json"
    return json.loads(path.read_text()) if path.exists() else None


def last_checkpoint(run_id):
    """The final checkpoint of the last session a pretraining stage's run reported."""
    reports = sorted((Path("/runs") / run_id).glob("segment-*.json"))
    return json.loads(reports[-1].read_text())["final_checkpoint"] if reports else None


def await_training(run_id, deadline):
    """Wait for a pretraining stage's session chain: its final status, or None when the
    launch deadline comes first (a later launch continues it)."""
    while time.time() < deadline - STAGE_RESERVE_SECONDS:
        status = stage_status(run_id)
        if status and status["stage"] in ("completed", "failed", "stopped", "deadline_cancelled"):
            return status
        time.sleep(60)
    return None


def relative(path):
    return str(Path(path).relative_to("/runs"))


def run_training_stage(stage, plan, state, parent, provenance, launch_data):
    """Run (or continue) a pretraining or mid-training stage; its output, or None if
    the launch ends first. `launch_data` is the data stage to prepare first, if any."""
    from workflows.config import resolve_payload

    deadline = provenance["deadline_unix"]
    bundle, family = stage["spec"], plan["family"]
    record_ = state["stages"].setdefault(stage["name"], {"attempt": 0})
    previous_run = record_.get("run_id")
    # A stage interrupted by an earlier launch resumes its last checkpoint in a new run.
    resume = stage.get("resume", "")
    if previous_run:
        resume = last_checkpoint(previous_run) or resume
        resume = relative(resume) if resume.startswith("/runs") else resume
    record_["attempt"] += 1
    run = stage["run_id"] if record_["attempt"] == 1 else f"{stage['run_id']}-a{record_['attempt']}"
    record_.update(run_id=run, status="running")
    stage_provenance = {**provenance, "stage": stage["name"]}
    if family == "hybrid":
        init_from = ""
        if stage["stage"] == "midtrain" and not resume:
            init_from = parent["export"]
        launch_hybrid.spawn(run, bundle, stage_provenance, resume, init_from).get()
    elif launch_data is not None:
        # Preparation that an earlier launch started continues in that run.
        if previous_run:
            run = previous_run
            record_["attempt"] -= 1
            record_["run_id"] = run
        prepared = prepare_and_launch.spawn(
            run,
            launch_data["spec"]["version"],
            bundle,
            stage_provenance,
            resume_preparation=bool(previous_run),
        ).get()
        state["stages"][launch_data["name"]] = {"status": "completed", "output": prepared}
    else:
        snapshot = (
            state["stages"][stage["data"]]["output"]["snapshot"] if stage.get("data") else None
        )
        if snapshot:
            _, dataset, _ = resolve_payload(bundle)
            bundle = {
                **bundle,
                "dataset": {
                    "mode": "snapshot",
                    "root": dataset.root,
                    "loader": "raw",
                    "snapshot": snapshot,
                },
            }
        phase_from = parent["checkpoint"] if stage["stage"] == "midtrain" and not resume else ""
        launch_snapshot.spawn(run, bundle, stage_provenance, resume, phase_from).get()
    status = await_training(run, deadline)
    # A stage stopped on request continues, like one the deadline ended, at the next launch.
    if status is None or status["stage"] == "stopped":
        return None
    if status["stage"] != "completed":
        raise RuntimeError(
            f"Stage {stage['name']} ended {status['stage']}: {status.get('error', '')}"
        )
    output = {"checkpoint": relative(status["final_checkpoint"])}
    if family == "hybrid":
        output["export"] = str(
            Path(output["checkpoint"]).parent / "export" / f"tokens-{status['trained_tokens']:012d}"
        )
    return output


def run_data_stage(stage, plan, consumer, provenance, resume=False):
    """Prepare a data stage's data on its own (`resume` continues an earlier launch's
    preparation); its output."""
    deadline = provenance["deadline_unix"]
    from workflows.deadline import bounded_timeout

    if plan["family"] == "hybrid":
        from training_data.packed_shards import load_recipe

        recipe = load_recipe(stage["spec"]["recipe"])
        call = modal_data.coordinate.with_options(timeout=bounded_timeout(deadline, 86400)).spawn(
            plan["run_id"], recipe, deadline, stage["spec"]["parallel"]
        )
        track_call(plan["run_id"], call.object_id, "coordinate")
        return {"recipe": stage["spec"]["recipe_name"], "capacity": call.get()}
    if consumer is None:
        raise ValueError(f"Data stage {stage['name']} feeds no pretraining stage of the workflow")
    return prepare_and_launch.spawn(
        stage["run_id"],
        stage["spec"]["version"],
        consumer["spec"],
        {**provenance, "stage": stage["name"]},
        resume_preparation=resume,
        launch_training=False,
    ).get()


@app.function(
    image=cpu_image,
    cpu=1,
    memory=2048,
    timeout=86400,
    retries=0,
    volumes={"/runs": run_volume},
    secrets=[hf_secret],
)
def run_workflow(plan, selected, provenance, checkpoint="", examples=None, qualify=0):
    """Run the selected stages of a workflow in order, skipping those an earlier launch of
    this workflow run finished; records `/runs/<run>/stages.json` and the launch status.
    With `qualify` N, qualify exact resume of the one selected SFT stage over N steps
    instead (`modal_post_train.qualify_sft`)."""
    launch, deadline = plan["run_id"], provenance["deadline_unix"]
    track_call(launch, modal.current_function_call_id(), "run_workflow")
    run_volume.reload()
    state_path = Path("/runs") / launch / "stages.json"
    state = {"plan_sha256": plan["sha256"], "stages": {}}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != plan["sha256"]:
            raise ValueError("This workflow run was started with another plan")

    def save():
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        run_volume.commit()

    stages = {stage["name"]: stage for stage in plan["stages"]}
    order = [stage["name"] for stage in plan["stages"]]

    def parent_of(stage):
        """The output of the stage this one continues, or the given/`from` checkpoint."""
        if stage.get("parent") is None:
            return None
        given = checkpoint or plan["from"]
        if stage["parent"] != "from":
            done = state["stages"].get(stage["parent"], {})
            if done.get("status") == "completed":
                return done["output"]
            if stage["parent"] in selected:
                raise RuntimeError(f"Stage {stage['parent']} did not finish")
            given = checkpoint
        if not given:
            raise ValueError(
                f"Stage {stage['name']} needs --checkpoint: its parent stage has not run"
            )
        return {"checkpoint": given, "export": given}

    def consumer_of(data_stage):
        for name in order[order.index(data_stage) + 1 :]:
            if stages[name].get("data") == data_stage:
                return stages[name]
        return None

    # Post-training calls still running; branches of the last stage run side by side.
    pending = {}

    def finish(name):
        """Record a post-training call's result; an RL run that stopped short of its step
        target is paused, and the next launch resumes it in the same run."""
        result = pending.pop(name).get()
        reason = result.get("stop_reason", "steps")
        state["stages"][name].update(
            status="completed" if reason == "steps" else "paused",
            output={"checkpoint": relative(result["checkpoint"])},
            result=result,
        )
        save()

    record(launch, "running", selected=selected, provenance=provenance)
    try:
        if qualify:
            stage = stages[selected[0]]
            recipe = {
                **stage["spec"],
                "run_id": f"{stage['run_id']}-q",
                "parent": parent_of(stage)["checkpoint"],
            }
            report = modal_post_train.qualify_sft(
                recipe, launch, deadline, qualify, (examples or {}).get(stage["name"], b"")
            )
            record(launch, "completed", qualification=report, provenance=provenance)
            return report
        for name in selected:
            stage = stages[name]
            if state["stages"].get(name, {}).get("status") == "completed":
                continue
            if time.time() >= deadline - STAGE_RESERVE_SECONDS:
                record(launch, "paused", stages=state["stages"], provenance=provenance)
                return state["stages"]
            consumer = consumer_of(name) if stage["stage"] == "data" else None
            if stage["stage"] == "data":
                # A subword corpus prepared together with the selected stage it feeds.
                if plan["family"] == "subword" and consumer and consumer["name"] in selected:
                    continue
                resume = state["stages"].get(name, {}).get("status") == "running"
                state["stages"][name] = {"status": "running"}
                save()
                state["stages"][name] = {
                    "status": "completed",
                    "output": run_data_stage(stage, plan, consumer, provenance, resume),
                }
                save()
                continue
            if stage.get("parent") in pending:
                finish(stage["parent"])
            parent = parent_of(stage)
            if stage["stage"] in ("pretrain", "midtrain"):
                data_stage = stage.get("data")
                launch_data = None
                if (
                    plan["family"] == "subword"
                    and data_stage
                    and state["stages"].get(data_stage, {}).get("status") != "completed"
                ):
                    if data_stage not in selected:
                        raise RuntimeError(f"Data stage {data_stage} has not run in this workflow")
                    launch_data = stages[data_stage]
                save()
                output = run_training_stage(stage, plan, state, parent, provenance, launch_data)
                save()
                if output is None:
                    record(launch, "paused", stages=state["stages"], provenance=provenance)
                    return state["stages"]
                state["stages"][name].update(status="completed", output=output)
                save()
                continue
            # A stage an earlier launch started runs again from its parent in a new run.
            attempt = state["stages"].get(name, {}).get("attempt", 0) + 1
            # RL resumes its own run; other stages restart from their parent in a new run.
            fresh = attempt == 1 or stage["stage"] == "rl"
            run = stage["run_id"] if fresh else f"{stage['run_id']}-a{attempt}"
            recipe = {**stage["spec"], "run_id": run, "parent": parent["checkpoint"]}
            state["stages"][name] = {"status": "running", "run_id": run, "attempt": attempt}
            save()
            pending[name] = modal_post_train.run_stage(
                stage["stage"],
                plan["family"],
                recipe,
                launch,
                deadline,
                (examples or {}).get(name, b""),
            )
        for name in list(pending):
            finish(name)
        if any(state["stages"].get(n, {}).get("status") == "paused" for n in selected):
            record(launch, "paused", stages=state["stages"], provenance=provenance)
            return state["stages"]
    except BaseException as error:
        for done in state["stages"].values():
            if done.get("status") == "running":
                done["status"] = "failed"
        save()
        record(launch, "failed", error=repr(error), stages=state["stages"], provenance=provenance)
        raise
    record(launch, "completed", stages=state["stages"], provenance=provenance)
    return state["stages"]


@app.local_entrypoint()
def main(
    workflow: str,
    run_id: str,
    start: str = "",
    end: str = "",
    checkpoint: str = "",
    qualify: int = 0,
    wait: bool = False,
):
    """Run a workflow, or its stages `--start` to `--end`, as workflow run `--run-id`;
    `--checkpoint` is the parent of the first selected stage if its parent stage has not
    finished in this run. Launching the same workflow and run ID again continues it.
    `--qualify N` qualifies exact resume of one selected SFT stage over N steps.

    The controller is spawned, so a `--detach` launch keeps running after this client
    exits; `--wait` waits for it and prints its result."""
    from workflows.checks import parent_path
    from workflows.deadline import new_deadline
    from workflows.plan import load_workflow, select

    plan = load_workflow(workflow, run_id)
    selected = select(plan, start, end)
    stages = {stage["name"]: stage for stage in plan["stages"]}
    if qualify and (len(selected) != 1 or stages[selected[0]]["stage"] != "sft" or qualify < 2):
        raise ValueError("--qualify N (at least 2) selects one SFT stage with --start/--end")
    if checkpoint:
        parent_path(checkpoint, "checkpoint")
    examples = {
        stage["name"]: modal_post_train.local_examples(stage["spec"])
        for stage in plan["stages"]
        if stage["name"] in selected and stage["stage"] == "sft"
    }
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "workflow_run_id": run_id,
        "workflow": plan.get("name", ""),
        # Every call this launch starts stops by this time.
        "deadline_unix": new_deadline(),
    }
    control[run_id] = "run"
    # The supervisor starts first so no work runs unsupervised.
    supervisor = enforce_deadline.spawn(run_id, provenance["deadline_unix"])
    print(
        json.dumps(
            {
                "deadline_unix": provenance["deadline_unix"],
                "supervisor": supervisor.object_id,
                "stages": selected,
            }
        )
    )
    controller = run_workflow.spawn(plan, selected, provenance, checkpoint, examples, qualify)
    print(json.dumps({"controller": controller.object_id}))
    if wait:
        print(json.dumps(controller.get(), indent=2))


@app.local_entrypoint()
def stop(run_id: str):
    """Ask the run's current session to checkpoint and exit at its next step."""
    control[run_id] = "stop"
    print(json.dumps({"run_id": run_id, "requested": "stop"}))
