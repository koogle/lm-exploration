"""Modal orchestration for the distillation validation ladder.

``campaign`` is a remote supervisor with an absolute deadline (launch + plan budget,
at most 24 hours). Every child call gets a timeout bounded by the remaining budget,
all call and sandbox IDs are recorded in a Modal Dict and on the volume, and
outstanding work is cancelled at the deadline. Completed stages are cached on the
volume, so a relaunch resumes instead of repeating them. Retries are disabled.

Dry run (prints the plan and limits; launches nothing):
    modal run distillation/validation/modal_app.py::main
Launch (detached; the supervisor survives client disconnection):
    modal run --detach distillation/validation/modal_app.py::main --launch
"""

import json
import random
import sys
import time
from pathlib import Path

import modal

LOCAL = modal.is_local()
HERE = Path(__file__).parent if LOCAL else Path("/validation")
sys.path.insert(0, str(HERE))

from common import (
    CAMPAIGN,
    CONTROL,
    CONTROL_REVISION,
    DATASET,
    MBPP,
    MBPP_REVISION,
    TEACHER,
    TEACHER_REVISION,
    atomic_json,
    file_sha256,
    is_heldout,
    training_order,
)

PLAN = json.loads((HERE / "plan.json").read_text())
MAX_SECONDS = 24 * 60 * 60
VLLM_IMAGE = "vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9"
TIMEOUTS = {
    "prepare_group": 3600,
    "finalize_data": 1800,
    "build_mbpp": 900,
    "teacher_job": 5400,
    "control_job": 4 * 3600,
    "student_job": 6 * 3600,
}

app = modal.App("lm-distillation-validation")
volume = modal.Volume.from_name(
    "lm-exploration-distillation-validation", create_if_missing=True
)
registry = modal.Dict.from_name(
    "lm-distillation-validation-calls", create_if_missing=True
)
hf_secret = modal.Secret.from_name("distillation-huggingface")
IGNORE = ["artifacts", "tests", "__pycache__", "*.pyc"]


MODEL_SOURCE = HERE.parents[1] / "training" / "10m" if LOCAL else HERE / "model_10m"


def with_source(image):
    return image.add_local_dir(HERE, "/validation", ignore=IGNORE).add_local_dir(
        MODEL_SOURCE, "/validation/model_10m", ignore=["*.md", "__pycache__"]
    )


cpu_image = with_source(
    modal.Image.debian_slim(python_version="3.12").pip_install(
        "numpy==2.3.4",
        "huggingface_hub==0.36.0",
        "pyarrow==21.0.0",
        "tokenizers==0.22.1",
    )
)
torch_image = with_source(
    modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch==2.9.1", "numpy==2.3.4", "tokenizers==0.22.1"
    )
)
vllm_image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .entrypoint([])
    # The pinned vLLM runtime provides python3 but no unversioned python.
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python")
    .pip_install_from_requirements(str(HERE / "vllm-overlay.lock"))
    .pip_install("flash-linear-attention==0.5.2", extra_options="--no-deps")
    .pip_install("einops==0.8.1")
)
vllm_image = with_source(vllm_image)
grader_image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    HERE / "grader.py", "/grader/grader.py"
)


def campaign_dir(campaign):
    return Path("/runs") / campaign


def remaining(deadline, reserve=120):
    return int(deadline - time.time() - reserve)


@app.function(
    image=cpu_image,
    cpu=4,
    memory=24576,
    timeout=TIMEOUTS["prepare_group"],
    retries=0,
    max_containers=64,
    volumes={"/runs": volume},
    secrets=[hf_secret],
)
def prepare_group(campaign, name, files, revision, hashes, keep_records):
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    from prepare import write_shard

    local = Path("/tmp/download") / name

    def fetch(path):
        for attempt in range(8):
            try:
                result = hf_hub_download(
                    DATASET,
                    path,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=local,
                )
                break
            except HfHubHTTPError as error:
                if attempt == 7 or getattr(error.response, "status_code", None) not in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    raise
                time.sleep(30 * (attempt + 1))
        if path in hashes and file_sha256(result) != hashes[path]:
            raise ValueError(f"SHA256 mismatch for {path}")
        return result

    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(fetch, files))
    summary = write_shard(paths, campaign_dir(campaign) / "shards" / name, keep_records)
    volume.commit()
    return {"name": name, **summary}


@app.function(
    image=cpu_image,
    cpu=4,
    memory=32768,
    timeout=TIMEOUTS["finalize_data"],
    retries=0,
    volumes={"/runs": volume},
    secrets=[hf_secret],
)
def finalize_data(campaign, summaries, revision, vocab_size):
    import numpy as np
    from common import QWEN_VOCAB
    from huggingface_hub import hf_hub_download
    from prepare import build_vocab, byte_lengths

    volume.reload()
    root = campaign_dir(campaign)
    train = sorted(s["name"] for s in summaries if s["name"].startswith("train-"))
    heldout = sorted(s["name"] for s in summaries if s["name"].startswith("heldout-"))
    counts = np.zeros(QWEN_VOCAB, dtype=np.int64)
    for name in train:
        counts += np.load(root / "shards" / name / "counts.npy")
    table, kept = build_vocab(counts, vocab_size)
    np.save(root / "vocab-table.npy", table)
    np.save(root / "vocab-kept.npy", kept)
    tokenizer_path = hf_hub_download(
        DATASET,
        f"runs/{CAMPAIGN}/tokenizer/tokenizer.json",
        repo_type="dataset",
        revision=revision,
    )
    (root / "tokenizer.json").write_bytes(Path(tokenizer_path).read_bytes())
    np.save(
        root / "byte-lengths.npy",
        byte_lengths(json.loads(Path(tokenizer_path).read_text())),
    )
    heldout_counts = sum(np.load(root / "shards" / n / "counts.npy") for n in heldout)
    total = int(counts.sum())
    manifest = {
        "dataset": DATASET,
        "revision": revision,
        "train": [str(root / "shards" / n) for n in train],
        "heldout": [str(root / "shards" / n) for n in heldout],
        "train_tokens": total,
        "train_documents": sum(s["documents"] for s in summaries if s["name"] in train),
        "heldout_tokens": int(heldout_counts.sum()),
        "distinct_train_ids": int((counts > 0).sum()),
        "trimmed_vocab": vocab_size,
        "trimmed_train_coverage": float(counts[kept].sum() / total),
        "trimmed_heldout_coverage": float(
            heldout_counts[kept].sum() / heldout_counts.sum()
        ),
        "vocab_table": str(root / "vocab-table.npy"),
        "vocab_kept": str(root / "vocab-kept.npy"),
        "byte_lengths": str(root / "byte-lengths.npy"),
        "tokenizer": str(root / "tokenizer.json"),
        "tokenizer_sha256": file_sha256(root / "tokenizer.json"),
        "shards": summaries,
    }
    atomic_json(root / "manifest.json", manifest)
    volume.commit()
    return manifest


@app.function(image=cpu_image, timeout=TIMEOUTS["build_mbpp"], retries=0)
def build_mbpp():
    import pyarrow.parquet as pq
    from evalsets import mbpp_items
    from huggingface_hub import hf_hub_download

    rows = []
    for split in ("train", "test", "validation", "prompt"):
        path = hf_hub_download(
            MBPP,
            f"sanitized/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=MBPP_REVISION,
        )
        rows += pq.read_table(path).to_pylist()
    return mbpp_items(rows)


def run_vllm(mode, model, revision, items, out):
    import subprocess

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    items_path = out.with_suffix(".items.json")
    items_path.write_text(json.dumps(items))
    subprocess.run(
        [
            "python",
            "/validation/vllm_eval.py",
            mode,
            model,
            revision,
            str(items_path),
            str(out),
        ],
        cwd="/validation",
        check=True,
    )
    volume.commit()
    return json.loads(out.read_text())


@app.function(
    image=vllm_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=TIMEOUTS["teacher_job"],
    retries=0,
    volumes={"/runs": volume},
)
def teacher_job(campaign, mode, items):
    out = campaign_dir(campaign) / "teacher" / f"{mode}.json"
    return run_vllm(mode, TEACHER, TEACHER_REVISION, items, out)


@app.function(
    image=vllm_image,
    gpu="H100",
    cpu=8,
    memory=98304,
    timeout=TIMEOUTS["control_job"],
    retries=0,
    volumes={"/runs": volume},
)
def control_job(campaign, settings, items):
    import subprocess

    volume.reload()
    root = campaign_dir(campaign)
    run = root / "control" / settings["name"]
    run.mkdir(parents=True, exist_ok=True)
    base = run_vllm(
        "raw", CONTROL, CONTROL_REVISION, items, run / "base-completions.json"
    )
    (run / "settings.json").write_text(json.dumps(settings))
    subprocess.run(
        [
            "python",
            "/validation/control.py",
            str(run / "settings.json"),
            str(root / "manifest.json"),
            str(run),
        ],
        cwd="/validation",
        check=True,
    )
    volume.commit()
    tuned = run_vllm(
        "raw", str(run / "model"), "", items, run / "tuned-completions.json"
    )
    return {
        "base": base,
        "tuned": tuned,
        "summary": json.loads((run / "train-summary.json").read_text()),
    }


@app.function(
    image=torch_image,
    gpu="H100",
    cpu=8,
    memory=98304,
    timeout=TIMEOUTS["student_job"],
    retries=0,
    volumes={"/runs": volume},
)
def student_job(campaign, settings, items, eval_settings, deadline):
    from jobs import student_run

    volume.reload()
    root = campaign_dir(campaign)
    manifest = json.loads((root / "manifest.json").read_text())
    result = student_run(
        settings,
        manifest,
        items,
        root / "students" / settings["name"],
        deadline,
        eval_settings,
    )
    volume.commit()
    return result


def poll(call):
    """Non-blocking result check: ("pending" | "done" | "error", value).

    ``get(timeout=0)`` on an unfinished call raises the builtin TimeoutError; a child
    that exceeded its own timeout raises Modal's FunctionTimeoutError. Remote errors
    re-raise with the child's own exception type, so child code must not raise TimeoutError.
    """
    try:
        return "done", call.get(timeout=0)
    except (
        modal.exception.FunctionTimeoutError,
        modal.exception.OutputExpiredError,
    ) as error:
        return "error", repr(error)[:2000]
    except (TimeoutError, modal.exception.TimeoutError):
        return "pending", None
    except Exception as error:  # noqa: BLE001 - recorded, never retried
        return "error", repr(error)[:2000]


class Supervisor:
    """Spawns children within the deadline, records IDs and cancels at expiry."""

    def __init__(self, campaign, deadline):
        self.campaign, self.deadline = campaign, deadline
        self.state_path = campaign_dir(campaign) / "supervision.json"
        self.state = {
            "campaign": campaign,
            "deadline": deadline,
            "supervisor_call_id": modal.current_function_call_id(),
            "calls": {},
            "sandboxes": {},
            "status": "running",
        }
        self.save()

    def save(self):
        atomic_json(self.state_path, self.state)
        registry[self.campaign] = self.state
        volume.commit()

    def spawn(self, label, kind, *args):
        """Spawn ``kind`` (a TIMEOUTS key naming a function) bounded by the deadline."""
        function = FUNCTIONS[kind]
        budget = min(TIMEOUTS[kind], remaining(self.deadline))
        if budget < 300:
            raise TimeoutError(f"not enough time left to start {label}")
        call = function.with_options(timeout=budget).spawn(*args)
        self.state["calls"][label] = {
            "id": call.object_id,
            "timeout": budget,
            "spawned": time.time(),
        }
        self.save()
        return call

    def wait(self, calls, on_result=None):
        """Wait for ``{label: call}``; returns results and errors by label."""
        pending, results, errors = dict(calls), {}, {}
        while pending:
            if time.time() > self.deadline:
                self.cancel_all()
                raise TimeoutError("campaign deadline reached")
            for label, call in list(pending.items()):
                state, value = poll(call)
                if state == "pending":
                    continue
                if state == "done":
                    results[label] = value
                else:
                    errors[label] = value
                    self.state["calls"][label]["error"] = value
                    self.save()
                del pending[label]
                if label in results and on_result:
                    try:
                        on_result(label, results[label])
                    except Exception as error:  # noqa: BLE001 - outputs stay on the volume
                        errors[f"{label}:record"] = repr(error)[:2000]
            if pending:
                time.sleep(20)
        return results, errors

    def sandbox_grade(self, label, cases):
        if not cases:
            return {}
        budget = min(3600, remaining(self.deadline))
        if budget < 300:
            raise TimeoutError("not enough time left to grade")
        sandbox_app = modal.App.lookup(
            "lm-distillation-validation-grader", create_if_missing=True
        )
        sandbox = modal.Sandbox.create(
            "python",
            "/grader/grader.py",
            app=sandbox_app,
            image=grader_image,
            block_network=True,
            timeout=budget,
            cpu=8.0,
            memory=8192,
        )
        self.state["sandboxes"][label] = sandbox.object_id
        self.save()
        sandbox.stdin.write(json.dumps(cases).encode())
        sandbox.stdin.write_eof()
        sandbox.stdin.drain()
        output = sandbox.stdout.read()
        sandbox.wait()
        if sandbox.returncode != 0:
            raise RuntimeError(f"grader failed: {sandbox.stderr.read()[-2000:]}")
        return json.loads(output)

    def cancel_all(self):
        failures = {}
        for label, entry in self.state["calls"].items():
            try:
                modal.FunctionCall.from_id(entry["id"]).cancel()
            except modal.exception.Error as error:
                failures[label] = repr(error)[:500]
        for label, sandbox_id in self.state["sandboxes"].items():
            try:
                modal.Sandbox.from_id(sandbox_id).terminate()
            except modal.exception.Error as error:
                failures[label] = repr(error)[:500]
        self.state["cancel_failures"] = failures
        self.state["status"] = "cancelled_at_deadline"
        self.save()


def cached(path, compute):
    path = Path(path)
    volume.reload()
    if path.exists():
        return json.loads(path.read_text())
    value = compute()
    atomic_json(path, value)
    volume.commit()
    return value


def grade_completions(supervisor, label, items, completions):
    cases = [
        {
            "key": item["id"],
            "code": completions[item["id"]]["code"],
            "setup": item["setup"],
            "tests": item["tests"],
        }
        for item in items
        if (completions.get(item["id"]) or {}).get("code")
    ]
    return supervisor.sandbox_grade(label, cases)


@app.function(
    image=cpu_image,
    cpu=2,
    memory=8192,
    timeout=MAX_SECONDS,
    retries=0,
    volumes={"/runs": volume},
    secrets=[hf_secret],
)
def campaign(plan, deadline):
    from evalsets import curriculum_items, curriculum_problem
    from huggingface_hub import HfApi, hf_hub_download
    from jobs import score

    name = plan["campaign"]
    root = campaign_dir(name)
    root.mkdir(parents=True, exist_ok=True)
    supervisor = Supervisor(name, deadline)
    try:
        # 1. Data: fixed held-out split, fixed training order, nested token budgets.
        def data():
            api = HfApi()
            revision = api.dataset_info(DATASET).sha
            files = sorted(
                f
                for f in api.list_repo_files(
                    DATASET, repo_type="dataset", revision=revision
                )
                if f.startswith("data/") and f.endswith(".jsonl.gz")
            )
            index_files = [
                f
                for f in api.list_repo_files(
                    DATASET, repo_type="dataset", revision=revision
                )
                if f.startswith("runs/") and f.endswith("/index.json")
            ]
            hashes = {}
            for index in index_files:
                listing = json.loads(
                    Path(
                        hf_hub_download(
                            DATASET, index, repo_type="dataset", revision=revision
                        )
                    ).read_text()
                )
                hashes.update(
                    {entry["path"]: entry["sha256"] for entry in listing["files"]}
                )
            heldout = [f for f in files if is_heldout(f)]
            train = training_order([f for f in files if not is_heldout(f)])
            groups = {}
            workers = plan["prep_workers"]
            for index in range(workers):
                groups[f"train-{index:03d}"] = (
                    train[
                        index * len(train) // workers : (index + 1)
                        * len(train)
                        // workers
                    ],
                    False,
                )
            for index in range(plan["heldout_workers"]):
                part = heldout[index :: plan["heldout_workers"]]
                groups[f"heldout-{index:03d}"] = (part, True)
            calls = {
                f"prepare:{group}": supervisor.spawn(
                    f"prepare:{group}",
                    "prepare_group",
                    name,
                    group,
                    members,
                    revision,
                    {m: hashes[m] for m in members if m in hashes},
                    keep,
                )
                for group, (members, keep) in groups.items()
            }
            results, errors = supervisor.wait(calls)
            if errors:
                raise RuntimeError(f"data preparation failed: {errors}")
            summaries = list(results.values())
            call = supervisor.spawn(
                "finalize_data",
                "finalize_data",
                name,
                summaries,
                revision,
                plan["trimmed_vocab"],
            )
            manifest = supervisor.wait({"finalize_data": call})[0]["finalize_data"]
            return {
                "revision": revision,
                "files": len(files),
                "train_files": len(train),
                "heldout_files": len(heldout),
                "hashed_files": sum(1 for f in files if f in hashes),
                **{k: v for k, v in manifest.items() if k != "shards"},
            }

        data_summary = cached(root / "stages" / "data.json", data)

        # 2. Evaluation sets: MBPP plus teacher-tested held-out curriculum problems.
        mbpp = cached(
            root / "stages" / "mbpp.json",
            lambda: supervisor.wait(
                {"build_mbpp": supervisor.spawn("build_mbpp", "build_mbpp")}
            )[0]["build_mbpp"],
        )

        def curriculum():
            records = []
            for shard in json.loads((root / "manifest.json").read_text())["heldout"]:
                records += [
                    json.loads(line)
                    for line in (Path(shard) / "records.jsonl").read_text().splitlines()
                ]
            random.Random(plan["eval"]["seed"]).shuffle(records)
            candidates = [r for r in records if curriculum_problem(r)][
                : plan["eval"]["curriculum_candidates"]
            ]
            asks = [
                {"id": r["id"], "problem": curriculum_problem(r)} for r in candidates
            ]
            call = supervisor.spawn("teacher:tests", "teacher_job", name, "tests", asks)
            written = supervisor.wait({"teacher:tests": call})[0]["teacher:tests"][
                "results"
            ]
            tests = {key: value["tests"] for key, value in written.items()}
            cases = [
                {
                    "key": r["id"],
                    "code": r["code"],
                    "setup": "",
                    "tests": tests[r["id"]],
                }
                for r in candidates
                if len(tests.get(r["id"], [])) >= 3
            ]
            graded = supervisor.sandbox_grade("grade:curriculum-references", cases)
            items, counts = curriculum_items(
                candidates, tests, {k: v["passed"] for k, v in graded.items()}
            )
            return {
                "items": items[: plan["eval"]["curriculum_max"]],
                "agreement": counts,
                "reference_errors": {
                    k: v["error"] for k, v in graded.items() if not v["passed"]
                },
            }

        curriculum_stage = cached(root / "stages" / "curriculum.json", curriculum)
        items = mbpp + curriculum_stage["items"]
        atomic_json(root / "eval-items.json", items)
        volume.commit()

        # 3. Teacher ceiling, control and students run in parallel.
        results_path = root / "results.json"
        volume.reload()
        results = (
            json.loads(results_path.read_text())
            if results_path.exists()
            else {"models": {}}
        )
        results.update(
            {
                "campaign": name,
                "data": data_summary,
                "curriculum_agreement": curriculum_stage["agreement"],
                "eval_items": {
                    "mbpp": len(mbpp),
                    "curriculum": len(curriculum_stage["items"]),
                },
            }
        )

        def record(model, completions, extra):
            grades = grade_completions(supervisor, f"grade:{model}", items, completions)
            atomic_json(root / "grades" / f"{model}.json", grades)
            results["models"][model] = {
                "score": score(items, completions, grades),
                **extra,
            }
            atomic_json(results_path, results)
            volume.commit()

        calls = {}
        if "teacher" not in results["models"]:
            calls["teacher"] = supervisor.spawn(
                "teacher:solve", "teacher_job", name, "solve", items
            )
        control = plan["control"]
        if f"{control['name']}:tuned" not in results["models"]:
            calls["control"] = supervisor.spawn(
                "control", "control_job", name, control, items
            )
        for settings in plan["students"]:
            if settings["name"] not in results["models"]:
                merged = {**plan["student_defaults"], **settings}
                calls[settings["name"]] = supervisor.spawn(
                    f"student:{settings['name']}",
                    "student_job",
                    name,
                    merged,
                    items,
                    plan["eval"],
                    deadline,
                )

        def on_result(label, value):
            if label == "teacher":
                record("teacher", value["results"], {"model": TEACHER})
            elif label == "control":
                record(
                    f"{control['name']}:base",
                    value["base"]["results"],
                    {"model": CONTROL},
                )
                record(
                    f"{control['name']}:tuned",
                    value["tuned"]["results"],
                    {"model": CONTROL, "train": value["summary"]},
                )
            else:
                record(label, value["completions"], {"train": value["summary"]})

        _, errors = supervisor.wait(calls, on_result)
        results["errors"] = errors
        atomic_json(results_path, results)
        supervisor.state["status"] = "completed_with_errors" if errors else "completed"
        supervisor.save()
        return results
    except BaseException:
        if supervisor.state["status"] == "running":
            supervisor.state["status"] = "failed"
            supervisor.cancel_all()
            supervisor.state["status"] = "failed"
            supervisor.save()
        raise


FUNCTIONS = {
    "prepare_group": prepare_group,
    "finalize_data": finalize_data,
    "build_mbpp": build_mbpp,
    "teacher_job": teacher_job,
    "control_job": control_job,
    "student_job": student_job,
}


@app.local_entrypoint()
def main(launch: bool = False):
    budget = PLAN["budget_seconds"]
    if not 0 < budget <= MAX_SECONDS:
        raise ValueError("budget must be within 24 hours")
    limits = {name: min(value, budget) for name, value in TIMEOUTS.items()}
    print(
        json.dumps(
            {
                "campaign": PLAN["campaign"],
                "budget_seconds": budget,
                "timeouts": limits,
            },
            indent=1,
        )
    )
    print(
        f"{len(PLAN['students'])} students + control + teacher; dry run"
        if not launch
        else "launching"
    )
    if not launch:
        return
    state = registry.get(PLAN["campaign"])
    if state and state.get("status") == "running" and state["deadline"] > time.time():
        raise RuntimeError(f"campaign already running: {state['supervisor_call_id']}")
    deadline = time.time() + budget
    call = campaign.with_options(timeout=budget).spawn(PLAN, deadline)
    receipt = {
        "campaign": PLAN["campaign"],
        "supervisor_call_id": call.object_id,
        "deadline": deadline,
    }
    artifacts = HERE / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / f"launch-{PLAN['campaign']}-{int(time.time())}.json").write_text(
        json.dumps(receipt, indent=1)
    )
    print(json.dumps(receipt, indent=1))


@app.local_entrypoint()
def stop():
    """Cancel every recorded call and sandbox of the plan's campaign."""
    state = registry.get(PLAN["campaign"]) or {}
    ids = [state.get("supervisor_call_id")] + [
        c["id"] for c in state.get("calls", {}).values()
    ]
    for call_id in filter(None, ids):
        modal.FunctionCall.from_id(call_id).cancel()
    for sandbox_id in state.get("sandboxes", {}).values():
        modal.Sandbox.from_id(sandbox_id).terminate()
    if state:
        registry[PLAN["campaign"]] = {
            **state,
            "status": "stopped",
            "stopped_at": time.time(),
        }
    print(f"cancelled {len(ids)} calls and {len(state.get('sandboxes', {}))} sandboxes")
