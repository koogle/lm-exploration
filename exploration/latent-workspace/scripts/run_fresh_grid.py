"""Run the declared fresh grid with bounded Modal chunks and a fail-closed ledger.

Run from this project's directory. Logs/state stay in ignored artifacts. This
orchestrator never raises the spending cap or modifies Python training source.
"""

import fcntl
import hashlib
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path.cwd()
STATE_PATH = ROOT / "artifacts/fresh-grid-progress.json"
LOCK = threading.Lock()
STATE = {"study": "fresh-response-data-grid", "runs": {}, "phase": "waiting for manifest"}
SOURCE = hashlib.sha256()
for path in sorted((ROOT / "src/latent_workspace").glob("*.py")):
    SOURCE.update(path.name.encode() + path.read_bytes())
SOURCE = SOURCE.hexdigest()
STATE["package_source_sha256"] = SOURCE


def update(**values):
    with LOCK:
        STATE.update(values)
        temporary = STATE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(STATE, indent=2) + "\n")
        temporary.replace(STATE_PATH)


def check_source():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/latent_workspace").glob("*.py")):
        digest.update(path.name.encode() + path.read_bytes())
    if digest.hexdigest() != SOURCE:
        raise RuntimeError("training source changed during the grid; refusing another worker")


def invoke(script, run_id, config, **options):
    check_source()
    args = [
        "uv",
        "run",
        "modal",
        "run",
        "--detach",
        script,
        "--config",
        str(config),
        "--run-id",
        run_id,
    ]
    for key, value in options.items():
        args.append("--" + key.replace("_", "-"))
        if value is not True:
            args.append(str(value))
    with (ROOT / f"artifacts/{run_id}.log").open("w") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=4200)
    return json.loads((ROOT / f"results/{run_id}.json").read_text())


def train_condition(size, cap, teacher):
    name = f"fresh-grid-{size}-r{cap}"
    config_path = ROOT / f"configs/{name}.json"
    cfg = json.loads(config_path.read_text())
    if cfg["warm_start_from"] or cfg["continue_from"] or cfg["steps"] != 100000:
        raise RuntimeError("initial grid condition is not the declared fresh 100000-update fit")
    cfg["teacher_parent"] = "/runs/" + teacher
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    for worker in range(1, 31):
        run_id = f"{name}-{worker:02d}"
        with LOCK:
            STATE["runs"][name] = {"run_id": run_id, "status": "running", "worker": worker}
        update()
        report = invoke("modal_dflash.py", run_id, config_path, long_run=True)
        if report["source_sha256"] != SOURCE:
            raise RuntimeError("worker package source does not match declared grid")
        with LOCK:
            STATE["runs"][name].update(
                step=report["curve"][-1]["step"],
                best_step=report["best_step"],
                best_validation_new=report.get("selected_validation", {}).get(
                    "accepted_beyond_anchor"
                ),
            )
        if report["training_completed"]:
            with LOCK:
                STATE["runs"][name]["status"] = "completed"
            update()
            return run_id
        if report.get("stopped") != "bounded time limit; training incomplete":
            raise RuntimeError(f"unexpected training failure in {run_id}")
        config_path = ROOT / f"configs/{name}-{worker + 1:02d}.json"
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/resume_dflash.py",
                f"results/{run_id}.json",
                "--output",
                str(config_path),
            ],
            check=True,
        )
    raise RuntimeError("condition reached 30 bounded workers; review progress before extending")


def main():
    manifest = "long-manifest-216k-02"
    futures = []
    previous = ""
    with ThreadPoolExecutor(max_workers=5) as pool:
        update(phase="extending the existing 54k teacher trajectories to 256 tokens")
        for worker in range(1, 31):
            small_teacher = f"long-teacher-54k-r256-{worker:02d}"
            report = invoke(
                "modal_long_data.py",
                small_teacher,
                "configs/long-data-54k-r256.json",
                manifest_parent="dflash-data-54k-01",
                **({"resume_parent": previous} if previous else {}),
            )
            update(
                teacher_run=small_teacher,
                teacher_cursor=report["cursor"],
                teacher_counts=report["teacher_audit"]["by_split_domain"]["train"],
            )
            if report["completed"]:
                break
            previous = small_teacher
        else:
            raise RuntimeError("54k trajectory extension reached its bounded worker limit")
        for cap in (64, 256):
            futures.append(pool.submit(train_condition, 49152, cap, small_teacher))
        while not (ROOT / f"results/{manifest}.json").exists():
            time.sleep(15)
        manifest_report = json.loads((ROOT / f"results/{manifest}.json").read_text())
        if not manifest_report["completed"]:
            raise RuntimeError(
                "manifest incomplete; resume it explicitly before expanding the pool"
            )
        update(phase="two fresh fits running; expanding long teacher pool", manifest=manifest)
        previous = ""
        for worker in range(1, 31):
            for future in futures:
                if future.done() and future.exception() is not None:
                    raise future.exception()
            current = f"long-teacher-216k-{worker:02d}"
            report = invoke(
                "modal_long_data.py",
                current,
                "configs/long-data-216k.json",
                manifest_parent=manifest,
                seed_parent=small_teacher,
                **({"resume_parent": previous} if previous else {}),
            )
            update(
                teacher_run=current,
                teacher_cursor=report["cursor"],
                teacher_counts=report["teacher_audit"]["by_split_domain"]["train"],
            )
            counts = report["teacher_audit"]["by_split_domain"]["train"]
            if report["completed"]:
                if not all(counts[d]["prompts"] >= 65536 for d in ("prose", "code", "translation")):
                    raise RuntimeError(
                        "completed teacher pool lacks 196608 usable balanced examples"
                    )
                for size, cap in ((98304, 256), (196608, 64), (196608, 256)):
                    futures.append(pool.submit(train_condition, size, cap, current))
                update(phase="five substantive fresh fits running", completed_teacher=current)
                break
            previous = current
        else:
            raise RuntimeError("teacher preparation reached 30 bounded workers; review throughput")
        results = [future.result() for future in futures]
    update(phase="training complete; comparison and qualification pending", final_runs=results)


if __name__ == "__main__":
    lease = (ROOT / "artifacts/fresh-grid-orchestrator.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        update()
        main()
    except Exception as exc:
        update(phase="needs review", error=str(exc))
        raise
