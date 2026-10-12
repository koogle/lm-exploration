"""Bounded resumable long-response preparation; shares the authoritative ledger."""

import json
import re
from pathlib import Path

import modal

from latent_workspace.budget import reserve

app = modal.App("latent-workspace-qwen-long-data")
cache = modal.Volume.from_name("latent-workspace-qwen-cache", create_if_missing=True)
runs = modal.Volume.from_name("latent-workspace-qwen-runs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv==0.10.12")
    .add_local_file("pyproject.toml", "/project/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/project/uv.lock", copy=True)
    .run_commands("cd /project && uv sync --frozen --no-install-project")
    .env({"PATH": "/project/.venv/bin:/usr/local/bin:/usr/bin:/bin", "HF_HOME": "/cache/hf"})
    .add_local_python_source("latent_workspace")
)


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=131072,
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/cache": cache, "/runs": runs},
)
def prepare(
    config, run_id, manifest_parent, seed_parent, resume_parent, seal_parent, required_parent
):
    import sys

    sys.path.insert(0, "/project/.venv/lib/python3.12/site-packages")
    from latent_workspace.dflash import DraftConfig
    from latent_workspace.dflash_large_data import prepare_chunk, prepare_manifest, seal_subset

    runs.reload()
    try:
        if seal_parent:
            return seal_subset(
                f"/runs/{seal_parent}", f"/runs/{required_parent}", f"/runs/{run_id}"
            )
        if not manifest_parent:
            return prepare_manifest(
                DraftConfig(**config),
                f"/runs/{run_id}",
                f"/runs/{seed_parent}",
                on_checkpoint=runs.commit,
            )
        return prepare_chunk(
            DraftConfig(**config),
            f"/runs/{run_id}",
            f"/runs/{manifest_parent}",
            f"/runs/{seed_parent}",
            f"/runs/{resume_parent}" if resume_parent else "",
            on_checkpoint=runs.commit,
        )
    finally:
        runs.commit()
        cache.commit()


@app.function(image=image, cpu=2, memory=16384, timeout=900, volumes={"/runs": runs})
def seal_available(parent, required_parent, output, count, domains=None):
    """Seal a completed snapshot's balanced subset while teacher GPU work continues."""
    import sys

    sys.path.insert(0, "/project/.venv/lib/python3.12/site-packages")
    from latent_workspace.dflash_large_data import seal_subset

    try:
        return seal_subset(
            f"/runs/{parent}",
            f"/runs/{required_parent}",
            f"/runs/{output}",
            count=count,
            domains=domains,
        )
    finally:
        runs.commit()


@app.local_entrypoint()
def seal_snapshot(parent: str, run_id: str, count: int = 98304, prose_code_only: bool = False):
    # Reserve through the authoritative coordinator's ledger before invoking this.
    report = seal_available.remote(
        parent,
        "dflash-expansion-49152-01",
        run_id,
        count,
        ["prose", "code"] if prose_code_only else None,
    )
    print(json.dumps(report, indent=2))


@app.local_entrypoint()
def training_check(config: str, run_id: str):
    """Run a separately reserved, bounded real-model curriculum plumbing check."""
    report = grid_train.remote(json.loads(Path(config).read_text()), run_id)
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "completed",
                    "training_completed",
                    "source_sha256",
                    "initial_drafter_sha256",
                    "frozen_target_has_no_gradients",
                    "verification_passed",
                    "best_step",
                )
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def main(
    config: str,
    run_id: str,
    manifest_parent: str = "",
    seed_parent: str = "dflash-data-54k-01",
    resume_parent: str = "",
    seal_parent: str = "",
    required_parent: str = "dflash-expansion-49152-01",
    launch_grid: bool = False,
    resume_controller: str = "",
    available_teacher: str = "",
    prose_code_teacher: str = "",
):
    from latent_workspace.dflash import DraftConfig

    cfg = DraftConfig(**json.loads(Path(config).read_text())).validate()
    if launch_grid:
        import hashlib

        reserve("artifacts/budget.json", run_id, "CPU", timeout_seconds=86400)
        source = hashlib.sha256()
        for path in sorted(Path("src/latent_workspace").glob("*.py")):
            source.update(path.name.encode() + path.read_bytes())
        blueprint = {
            "package_source_sha256": source.hexdigest(),
            "small_teacher": json.loads(Path("configs/long-data-54k-r256.json").read_text()),
            "large_teacher": json.loads(Path("configs/long-data-216k.json").read_text()),
            "conditions": {
                name: json.loads((Path("configs") / f"{name}.json").read_text())
                for name in (
                    "fresh-grid-49152-r64",
                    "fresh-grid-49152-r256",
                    "fresh-grid-98304-r256",
                    "fresh-grid-196608-r64",
                    "fresh-grid-196608-r256",
                )
            },
        }
        if resume_controller:
            budget_parent = Path("artifacts/fresh-grid-remote-state.json")
            blueprint["resume_state"] = json.loads(budget_parent.read_text())
            blueprint["resumed_controller"] = resume_controller
        if available_teacher:
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", available_teacher):
                raise ValueError("invalid available teacher run")
            blueprint["available_teacher"] = available_teacher
        if prose_code_teacher:
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", prose_code_teacher):
                raise ValueError("invalid prose/code teacher run")
            blueprint["prose_code_teacher"] = prose_code_teacher
        call = grid_controller.spawn(
            run_id, blueprint, json.loads(Path("artifacts/budget.json").read_text())
        )
        handle = {
            "run_id": run_id,
            "call_id": call.object_id,
            "remote_budget_path": f"{run_id}/budget.json",
            "remote_state_path": f"{run_id}/state.json",
        }
        Path("artifacts/budget_remote_controller.json").write_text(
            json.dumps(handle, indent=2) + "\n"
        )
        print(json.dumps(handle, indent=2), flush=True)
        return

    for name in (run_id, seed_parent, manifest_parent, resume_parent, seal_parent, required_parent):
        if name and not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", name):
            raise ValueError("invalid run ID")
    if (
        (seal_parent and (manifest_parent or resume_parent))
        or cfg.max_seconds > 3300
        or (resume_parent and not manifest_parent)
    ):
        raise ValueError("invalid bounded preparation")
    amount = reserve("artifacts/budget.json", run_id, "L40S", timeout_seconds=3600)
    print(f"Reserved ${amount:.2f} for bounded preparation.")
    report = prepare.remote(
        cfg.to_dict(),
        run_id,
        manifest_parent,
        seed_parent,
        resume_parent,
        seal_parent,
        required_parent,
    )
    Path(f"results/{run_id}.json").write_text(json.dumps(report, indent=2) + "\n")


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=131072,
    timeout=3600,
    retries=0,
    max_containers=5,
    volumes={"/cache": cache, "/runs": runs},
)
def grid_train(config, run_id):
    import sys

    sys.path.insert(0, "/project/.venv/lib/python3.12/site-packages")
    from latent_workspace.dflash import DraftConfig
    from latent_workspace.dflash_train import run_dflash

    runs.reload()
    try:
        return run_dflash(DraftConfig(**config), f"/runs/{run_id}", on_checkpoint=runs.commit)
    finally:
        runs.commit()
        cache.commit()


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/runs": runs},
)
def grid_controller(run_id, blueprint, ledger):
    """Coordinate bounded GPU jobs on Modal, independently of chat/client lifetime."""
    import time

    from latent_workspace.train import write_json

    directory = Path("/runs") / run_id
    directory.mkdir(exist_ok=True)
    budget_path = directory / "budget.json"
    if not budget_path.exists():
        write_json(budget_path, ledger)
    state = {
        "phase": "extending 54k teacher trajectories",
        "runs": {},
        "package_source_sha256": blueprint["package_source_sha256"],
    }
    if (directory / "state.json").exists():
        state = json.loads((directory / "state.json").read_text())
    elif blueprint.get("resume_state"):
        state = blueprint["resume_state"]
    if state["package_source_sha256"] != blueprint["package_source_sha256"]:
        raise ValueError("controller restart changes captured training source")
    write_json(directory / "blueprint.json", blueprint)
    active = {}
    for name, record in state["runs"].items():
        if record.get("status") == "running":
            active[name] = (
                modal.FunctionCall.from_id(record["call_id"]),
                record["config"],
                record["worker"],
                record["run_id"],
            )
    allocation_blocked = False

    def save():
        write_json(directory / "state.json", state)
        runs.commit()

    def allocate(job_id):
        nonlocal allocation_blocked
        try:
            reserve(budget_path, job_id, "L40S", timeout_seconds=3600)
            save()
            return True
        except ValueError as exc:
            if "allocation cap reached" not in str(exc):
                raise
            allocation_blocked = True
            state["budget_error"] = str(exc)
            save()
            return False

    def submit_training(name, cfg, number=1):
        if name in active or state["runs"].get(name, {}).get("status") in (
            "completed",
            "early stopped",
        ):
            return
        job_id = f"{name}-{number:02d}"
        if not allocate(job_id):
            state["runs"].setdefault(name, {})["status"] = "budget paused"
            save()
            return
        call = grid_train.spawn(cfg, job_id)
        active[name] = (call, cfg, number, job_id)
        state["runs"][name] = {
            "status": "running",
            "run_id": job_id,
            "call_id": call.object_id,
            "worker": number,
            "config": cfg,
        }
        save()

    def poll_training():
        for name, (call, cfg, number, job_id) in list(active.items()):
            try:
                report = call.get(timeout=0)
            except TimeoutError:
                continue
            del active[name]
            if report["source_sha256"] != blueprint["package_source_sha256"]:
                raise RuntimeError("worker package source changed")
            state["runs"][name].update(
                step=report["curve"][-1]["step"], best_step=report["best_step"]
            )
            if report["training_completed"]:
                state["runs"][name].update(
                    status="completed",
                    validation_new=report["selected_validation"]["accepted_beyond_anchor"],
                )
            else:
                if report.get("stopped") != "bounded time limit; training incomplete":
                    raise RuntimeError(f"unexpected training failure: {job_id}")
                remaining = report["training_goal_step"] - report["curve"][-1]["step"]
                if remaining <= 0 or number >= 30:
                    raise RuntimeError("invalid or excessive training continuation")
                resumed = {
                    **cfg,
                    "continue_from": f"/runs/{job_id}",
                    "steps": remaining,
                    "curriculum_switch": False,
                }
                if cfg.get("training_domains") == ["prose", "code"] and state.get(
                    "full_teacher_run"
                ):
                    resumed.update(
                        teacher_parent=f"/runs/{state['full_teacher_run']}",
                        training_domains=["prose", "code", "translation"],
                        train_limit=196608,
                        curriculum_switch=True,
                    )
                submit_training(name, resumed, number + 1)
            save()

    def wait_call(call):
        while True:
            poll_training()
            try:
                return call.get(timeout=15)
            except TimeoutError:
                continue

    def preparation(cfg, manifest, seed, prefix, start=1):
        stages = state.setdefault("preparations", {})
        if prefix not in stages:
            stages[prefix] = {"next_worker": start, "previous": ""}
            if state.get("teacher_run", "").startswith(prefix + "-") and state.get(
                "teacher_call_id"
            ):
                stages[prefix].update(
                    next_worker=int(state["teacher_run"].rsplit("-", 1)[1]),
                    job_id=state["teacher_run"],
                    call_id=state["teacher_call_id"],
                )
        stage = stages[prefix]
        if stage.get("completed_run"):
            return stage["completed_run"]
        while stage["next_worker"] <= 30:
            number = stage["next_worker"]
            job_id = f"{prefix}-{number:02d}"
            if stage.get("call_id"):
                call = modal.FunctionCall.from_id(stage["call_id"])
            else:
                if not allocate(job_id):
                    return None
                call = prepare.spawn(cfg, job_id, manifest, seed, stage["previous"], "", "")
                stage.update(job_id=job_id, call_id=call.object_id)
                state.update(teacher_run=job_id, teacher_call_id=call.object_id)
                save()
            report = wait_call(call)
            state.update(
                teacher_cursor=report["cursor"],
                teacher_counts=report["teacher_audit"]["by_split_domain"]["train"],
            )
            if report["completed"]:
                stage["completed_run"] = job_id
                save()
                return job_id
            stage.update(previous=job_id, next_worker=number + 1)
            stage.pop("call_id", None)
            save()
        raise RuntimeError("teacher preparation reached its worker limit")

    try:
        for name, record in list(state["runs"].items()):
            if record.get("status") == "retry pending":
                submit_training(name, record["config"], record["worker"] + 1)
        for cfg in blueprint["conditions"].values():
            if cfg["warm_start_from"] or cfg["continue_from"] or cfg["steps"] != 100000:
                raise ValueError("grid condition must start fresh with declared exposure")
        save()
        small = preparation(
            blueprint["small_teacher"],
            "dflash-data-54k-01",
            "dflash-data-54k-01",
            "long-teacher-54k-r256",
            start=2,
        )
        if small:
            for size, cap in ((49152, 64), (49152, 256)):
                name = f"fresh-grid-{size}-r{cap}"
                submit_training(
                    name, {**blueprint["conditions"][name], "teacher_parent": f"/runs/{small}"}
                )
            if blueprint.get("available_teacher"):
                name = "fresh-grid-98304-r256"
                submit_training(
                    name,
                    {
                        **blueprint["conditions"][name],
                        "teacher_parent": f"/runs/{blueprint['available_teacher']}",
                    },
                )
            if blueprint.get("prose_code_teacher"):
                for cap in (256, 64):
                    name = f"fresh-grid-196608-r{cap}"
                    submit_training(
                        name,
                        {
                            **blueprint["conditions"][name],
                            "teacher_parent": f"/runs/{blueprint['prose_code_teacher']}",
                            "train_limit": 131072,
                            "training_domains": ["prose", "code"],
                        },
                    )
            state["phase"] = (
                "large prose/code fits; preparing translation"
                if blueprint.get("prose_code_teacher")
                else "98k fit; expanding long-response pool"
                if blueprint.get("available_teacher")
                else "two fresh fits; expanding long-response pool"
            )
            save()
            large = preparation(
                blueprint["large_teacher"], "long-manifest-216k-02", small, "long-teacher-216k"
            )
            if large:
                state["full_teacher_run"] = large
                if not all(
                    state["teacher_counts"][domain]["prompts"] >= 65536
                    for domain in ("prose", "code", "translation")
                ):
                    raise RuntimeError(
                        "completed teacher pool lacks enough balanced usable examples"
                    )
                for size, cap in ((98304, 256), (196608, 64), (196608, 256)):
                    name = f"fresh-grid-{size}-r{cap}"
                    condition = {**blueprint["conditions"][name]}
                    if size == 196608 and blueprint.get("available_teacher"):
                        condition["retain_training_parent"] = (
                            f"/runs/{blueprint['available_teacher']}"
                        )
                    submit_training(name, {**condition, "teacher_parent": f"/runs/{large}"})
                state["phase"] = (
                    "large curriculum fits; translation added at next worker boundary"
                    if blueprint.get("prose_code_teacher")
                    else (
                        "larger-data fits"
                        if blueprint.get("available_teacher")
                        else "five fresh fits"
                    )
                )
                save()
        while active:
            poll_training()
            time.sleep(15)
        state["phase"] = (
            "budget paused; study incomplete"
            if allocation_blocked
            else "training complete; comparison and qualification pending"
        )
        save()
        return state
    except Exception as exc:
        state.update(phase="needs review", error=str(exc))
        save()
        raise
