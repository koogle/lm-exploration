"""Modal orchestration for the distillation validation ladder.

``campaign`` is a remote supervisor with an absolute deadline (launch + plan budget,
at most 24 hours). Every child call gets a timeout bounded by the remaining budget,
all call IDs are recorded in a Modal Dict and on the volume, and
outstanding work is cancelled at the deadline. Completed stages are cached on the
volume, so a relaunch resumes instead of repeating them. Retries are disabled.

Dry run (prints the plan and limits; launches nothing):
    modal run distillation/validation/modal_app.py::main
Launch (detached; the supervisor survives client disconnection):
    modal run --detach distillation/validation/modal_app.py::main --launch
"""

import hashlib
import json
import os
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
    DATASET,
    MBPP,
    MBPP_REVISION,
    TEACHER,
    atomic_json,
    file_sha256,
    is_heldout,
    training_order,
)

# Local launches may select another plan, e.g. VALIDATION_PLAN=plan-covering.json.
PLAN = json.loads((HERE / os.environ.get("VALIDATION_PLAN", "plan.json")).read_text())
MAX_SECONDS = 24 * 60 * 60
VLLM_IMAGE = "vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9"
TIMEOUTS = {
    "prepare_group": 3600,
    "finalize_data": 1800,
    "build_mbpp": 900,
    "grade_job": 3600,
    "teacher_job": 5400,
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
vllm_image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .entrypoint([])
    # The pinned vLLM runtime provides python3 but no unversioned python.
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python")
    .pip_install_from_requirements(str(HERE / "vllm-overlay.lock"))
)
vllm_image = with_source(vllm_image)


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
    max_containers=16,  # HF rate-limits per-file download tokens
    volumes={"/runs": volume},
    secrets=[hf_secret],
)
def prepare_group(campaign, name, files, revision, hashes, keep_records):
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    from prepare import write_shard

    out = campaign_dir(campaign) / "shards" / name
    done = out / "summary.json"
    members = hashlib.sha256("\n".join([revision, *files]).encode()).hexdigest()
    if done.exists() and json.loads(done.read_text()).get("members_sha256") == members:
        return {"name": name, **json.loads(done.read_text())}
    local = Path("/tmp/download") / name

    def fetch(path):
        # HF rate-limits per-file download tokens; wait as long as it asks.
        for attempt in range(20):
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
                status = getattr(error.response, "status_code", None)
                if attempt == 19 or status not in (429, 500, 502, 503, 504):
                    raise
                after = (error.response.headers or {}).get("Retry-After", "")
                time.sleep(
                    min(300, int(after) if after.isdigit() else 30 * (attempt + 1))
                )
        if path in hashes and file_sha256(result) != hashes[path]:
            raise ValueError(f"SHA256 mismatch for {path}")
        return result

    with ThreadPoolExecutor(2) as pool:
        paths = list(pool.map(fetch, files))
    summary = write_shard(paths, out, keep_records)
    summary["members_sha256"] = members  # reuse only for the same revision and files
    atomic_json(done, summary)
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


@app.function(
    image=cpu_image, cpu=8, memory=16384, timeout=TIMEOUTS["grade_job"], retries=0
)
def grade_job(cases):
    """Run hidden tests on CPU; no volume is mounted, so candidate code cannot touch runs."""
    from grader import grade

    return grade(cases)


def run_vllm(mode, items, out):
    import subprocess

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    items_path = out.with_suffix(".items.json")
    items_path.write_text(json.dumps(items))
    subprocess.run(
        ["python", "/validation/vllm_eval.py", mode, str(items_path), str(out)],
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
    return run_vllm(mode, items, out)


@app.function(
    image=vllm_image,  # students train with PyTorch here, then evaluate through vLLM
    gpu="H100",
    cpu=8,
    memory=98304,
    timeout=TIMEOUTS["student_job"],
    retries=0,
    volumes={"/runs": volume},
)
def student_job(campaign, data_campaign, settings, items, eval_settings, deadline):
    from jobs import student_run

    volume.reload()
    root = campaign_dir(campaign)
    manifest = json.loads((campaign_dir(data_campaign) / "manifest.json").read_text())
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
        modal.exception.InputCancellation,  # a BaseException: a child cancelled by hand
    ) as error:
        return "error", repr(error)[:2000]
    except (TimeoutError, modal.exception.TimeoutError):
        return "pending", None
    except Exception as error:  # noqa: BLE001 - recorded, never retried
        return "error", repr(error)[:2000]


def adoptable(previous, call_id):
    """Calls recorded by an earlier attempt of the same supervisor input."""
    if previous.get("supervisor_call_id") != call_id:
        return {}
    return {k: v for k, v in previous.get("calls", {}).items() if "error" not in v}


class Supervisor:
    """Spawns children within the deadline, records IDs and cancels at expiry."""

    def __init__(self, campaign, deadline):
        self.campaign, self.deadline = campaign, deadline
        self.state_path = campaign_dir(campaign) / "supervision.json"
        call_id = modal.current_function_call_id()
        previous = registry.get(campaign) or {}
        self.state = {
            "campaign": campaign,
            "deadline": deadline,
            "supervisor_call_id": call_id,
            # A restarted input (same call ID) adopts the calls it already spawned.
            "calls": adoptable(previous, call_id),
            "status": "running",
        }
        self.save()

    def save(self):
        atomic_json(self.state_path, self.state)
        registry[self.campaign] = self.state
        volume.commit()

    def spawn(self, label, kind, *args):
        """Spawn ``kind`` (a TIMEOUTS key naming a function) bounded by the deadline."""
        if label in self.state["calls"] and "error" not in self.state["calls"][label]:
            return modal.FunctionCall.from_id(self.state["calls"][label]["id"])
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

    def grade(self, label, cases):
        """Grade cases in a CPU child (``grade_job``) like any other recorded call."""
        if not cases:
            return {}
        results, errors = self.wait({label: self.spawn(label, "grade_job", cases)})
        if errors:
            raise RuntimeError(f"grading failed: {errors}")
        return results[label]

    def cancel_all(self):
        failures = {}
        for label, entry in self.state["calls"].items():
            try:
                modal.FunctionCall.from_id(entry["id"]).cancel()
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
    return supervisor.grade(label, cases)


@app.function(
    image=cpu_image,
    cpu=2,
    memory=8192,
    timeout=MAX_SECONDS,
    retries=0,
    nonpreemptible=True,  # a preempted supervisor used to cancel its children
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
    # A follow-up campaign reuses an earlier campaign's shards and evaluation sets.
    data_name = plan.get("data_campaign", name)
    data_root = campaign_dir(data_name)
    supervisor = Supervisor(name, deadline)

    def stage(path, compute):
        if data_name != name and not Path(path).exists():
            raise RuntimeError(f"{path} missing in data campaign {data_name}")
        return cached(path, compute)

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

        data_summary = stage(data_root / "stages" / "data.json", data)

        # 2. Evaluation sets: MBPP plus teacher-tested held-out curriculum problems.
        mbpp = stage(
            data_root / "stages" / "mbpp.json",
            lambda: supervisor.wait(
                {"build_mbpp": supervisor.spawn("build_mbpp", "build_mbpp")}
            )[0]["build_mbpp"],
        )

        def curriculum():
            records = []
            for shard in json.loads((data_root / "manifest.json").read_text())[
                "heldout"
            ]:
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
            graded = supervisor.grade("grade:curriculum-references", cases)
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

        curriculum_stage = stage(data_root / "stages" / "curriculum.json", curriculum)
        items = mbpp + curriculum_stage["items"]
        atomic_json(root / "eval-items.json", items)
        volume.commit()

        # 3. Teacher ceiling and students run in parallel.
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
        if plan.get("teacher", True) and "teacher" not in results["models"]:
            calls["teacher"] = supervisor.spawn(
                "teacher:solve", "teacher_job", name, "solve", items
            )
        for settings in plan["students"]:
            if settings["name"] not in results["models"]:
                merged = {**plan["student_defaults"], **settings}
                calls[settings["name"]] = supervisor.spawn(
                    f"student:{settings['name']}",
                    "student_job",
                    name,
                    data_name,
                    merged,
                    items,
                    plan["eval"],
                    deadline,
                )

        def on_result(label, value):
            if label == "teacher":
                record("teacher", value["results"], {"model": TEACHER})
            else:
                record(label, value["completions"], {"train": value["summary"]})

        _, errors = supervisor.wait(calls, on_result)
        results["errors"] = errors
        atomic_json(results_path, results)
        supervisor.state["status"] = "completed_with_errors" if errors else "completed"
        supervisor.save()
        return results
    except Exception:
        if supervisor.state["status"] == "running":
            supervisor.state["status"] = "failed"
            supervisor.cancel_all()
            supervisor.state["status"] = "failed"
            supervisor.save()
        raise
    except BaseException:
        # Preemption or cancellation of the supervisor itself: children keep running,
        # bounded by their own deadline-capped timeouts; a restart adopts them.
        supervisor.state["status"] = "interrupted"
        supervisor.save()
        raise


SHARD_PATTERNS = [
    "manifest.json",
    "vocab-*.npy",
    "byte-lengths.npy",
    "tokenizer.json",
    "shards/*/*.npy",
    "shards/*/summary.json",
    "shards/*/records.jsonl",
]


@app.function(
    image=cpu_image,
    cpu=8,
    memory=16384,
    timeout=6 * 3600,
    retries=0,
    volumes={"/runs": volume},
    secrets=[hf_secret],
)
def archive_shards(campaign, repo_id):
    """Upload the prepared shards to a private HF dataset and verify every file."""
    from huggingface_hub import HfApi

    volume.reload()
    root = campaign_dir(campaign)
    if not (root / "stages" / "data.json").exists():
        raise RuntimeError("data stage has not completed")
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(HERE / "shards-card.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=str(root),
        repo_type="dataset",
        allow_patterns=SHARD_PATTERNS,
        private=True,
    )
    local = {
        str(path.relative_to(root)): path.stat().st_size
        for pattern in SHARD_PATTERNS
        for path in root.glob(pattern)
    }
    info = api.dataset_info(repo_id, files_metadata=True)
    remote = {entry.rfilename: entry.size for entry in info.siblings}
    mismatched = sorted(
        name for name, size in local.items() if remote.get(name) != size
    )
    receipt = {
        "repo_id": repo_id,
        "revision": info.sha,
        "files": len(local),
        "bytes": sum(local.values()),
        "mismatched": mismatched,
    }
    atomic_json(root / "shards-archive.json", receipt)
    volume.commit()
    if mismatched:
        raise RuntimeError(f"{len(mismatched)} files differ on HF: {mismatched[:5]}")
    return receipt


RESULT_PATTERNS = ["*.json", "*.jsonl"]
RESULT_IGNORE = [
    "shards/*",
    "hf-cache/*",
    "*/model/*",
    "*.pt",
    "*.npy",
    "tokenizer.json",
]


@app.function(
    image=cpu_image,
    cpu=2,
    memory=8192,
    timeout=2 * 3600,
    retries=0,
    volumes={"/runs": volume},
    secrets=[hf_secret],
)
def archive_results(campaigns, repo_id, prefix):
    """Upload campaign outputs (no shards or weights) to HF and verify each file."""
    import fnmatch

    from huggingface_hub import HfApi

    volume.reload()
    api = HfApi()
    receipts = {}
    for campaign in campaigns:
        root = campaign_dir(campaign)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(root),
            path_in_repo=f"{prefix}/{campaign}",
            allow_patterns=RESULT_PATTERNS,
            ignore_patterns=RESULT_IGNORE,
            commit_message=f"Archive {campaign} results",
        )
        local = {
            str(p.relative_to(root)): p.stat().st_size
            for p in root.rglob("*")
            if p.is_file()
            and any(
                fnmatch.fnmatch(str(p.relative_to(root)), g) for g in RESULT_PATTERNS
            )
            and not any(
                fnmatch.fnmatch(str(p.relative_to(root)), g) for g in RESULT_IGNORE
            )
        }
        info = api.dataset_info(repo_id, revision=commit.oid, files_metadata=True)
        remote = {e.rfilename: e.size for e in info.siblings}
        missing = sorted(
            n
            for n, size in local.items()
            if remote.get(f"{prefix}/{campaign}/{n}") != size
        )
        receipts[campaign] = {
            "revision": commit.oid,
            "files": len(local),
            "mismatched": missing,
        }
        if missing:
            raise RuntimeError(
                f"{campaign}: {len(missing)} files differ on HF: {missing[:5]}"
            )
    return receipts


WEIGHT_PATTERNS = [
    "students/*/model.pt",
    "students/*/settings.json",
    "students/*/summary.json",
]


@app.function(
    image=cpu_image,
    cpu=8,
    memory=16384,
    timeout=6 * 3600,
    retries=0,
    volumes={"/runs": volume},
    secrets=[hf_secret],
)
def archive_weights(campaigns, repo_id):
    """Upload final student weights to a private HF dataset; verify sizes."""
    from huggingface_hub import HfApi

    volume.reload()
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(HERE / "weights-card.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    patterns = [f"{c}/{p}" for c in campaigns for p in WEIGHT_PATTERNS]
    root = Path("/runs")
    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=str(root),
        repo_type="dataset",
        allow_patterns=patterns,
        private=True,
    )
    local = {
        str(path.relative_to(root)): path.stat().st_size
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file()
    }
    info = api.dataset_info(repo_id, files_metadata=True)
    remote = {entry.rfilename: entry.size for entry in info.siblings}
    mismatched = sorted(n for n, size in local.items() if remote.get(n) != size)
    receipt = {
        "repo_id": repo_id,
        "revision": info.sha,
        "files": len(local),
        "bytes": sum(local.values()),
        "mismatched": mismatched,
    }
    if mismatched:
        raise RuntimeError(f"{len(mismatched)} files differ on HF: {mismatched[:5]}")
    return receipt


FUNCTIONS = {
    "prepare_group": prepare_group,
    "finalize_data": finalize_data,
    "build_mbpp": build_mbpp,
    "grade_job": grade_job,
    "teacher_job": teacher_job,
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
        f"{len(PLAN['students'])} students + teacher; dry run"
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
def archive(repo_id: str = "jfrick/qwen35-distillation-shards-20261023"):
    """Upload the plan's prepared shards to a private HF dataset (after the data stage)."""
    print(json.dumps(archive_shards.remote(PLAN["campaign"], repo_id), indent=1))


@app.local_entrypoint()
def archive_outputs(
    campaigns: str,
    repo_id: str = "jfrick/lm-exploration-results",
    prefix: str = "distillation-validation",
):
    """Archive comma-separated campaigns' result files to the HF results dataset."""
    print(
        json.dumps(
            archive_results.remote(campaigns.split(","), repo_id, prefix), indent=1
        )
    )


@app.local_entrypoint()
def archive_checkpoints(
    campaigns: str, repo_id: str = "jfrick/distillation-validation-checkpoints-20261023"
):
    """Archive comma-separated campaigns' final weights to a private HF dataset."""
    print(json.dumps(archive_weights.remote(campaigns.split(","), repo_id), indent=1))


@app.local_entrypoint()
def stop():
    """Cancel every recorded call of the plan's campaign."""
    state = registry.get(PLAN["campaign"]) or {}
    ids = [state.get("supervisor_call_id")] + [
        c["id"] for c in state.get("calls", {}).values()
    ]
    for call_id in filter(None, ids):
        modal.FunctionCall.from_id(call_id).cancel()
    if state:
        registry[PLAN["campaign"]] = {
            **state,
            "status": "stopped",
            "stopped_at": time.time(),
        }
    print(f"cancelled {len(ids)} calls")
