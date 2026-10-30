"""Prepare repair-SFT datasets on Modal CPU. No training is launched from here.

Run from training/:
    modal run -m modal_scripts.modal_sft_data::curriculum --output <repairs.jsonl>
    modal run -m modal_scripts.modal_sft_data::commitpack --data-id <fresh-id> \
        [--exclude-dataset <jsonl>] [--report <report.json>]
    modal run -m modal_scripts.modal_sft_data::private_patches --sources <sources.json> \
        --data-id <fresh-id> [--reserved <data-id>,...] [--replay <data-id>] [--full] \
        [--report <report.json>]
    modal run -m modal_scripts.modal_sft_data::math --data-id <fresh-id> [--report <report.json>]
    modal run -m modal_scripts.modal_sft_data::code --data-id <fresh-id> [--report <report.json>]
    modal run -m modal_scripts.modal_sft_data::mixture --data-id <fresh-id> \
        --math <math-data-id> --code <code-data-id> [--report <report.json>]
    modal run -m modal_scripts.modal_sft_data::rl_pool --data-id <fresh-id> [--report <report.json>]

`curriculum` execute-verifies every original arithmetic repair and writes the JSONL locally.
`commitpack`, `private_patches`, `math` (GSM8K + MetaMathQA word problems), `code` (OpenCodeInstruct +
MBPP functions) and `mixture` (prepared math and code rows in one file) write `examples.jsonl`
and `report.json` to `/runs/<data-id>` on the private run volume for SFT workflows
(`"data": {"volume": data-id}`). `rl_pool` writes the RL prompt pool `pool.jsonl` (`training_data.rl_tasks`).
No supplied code is executed during preparation. Every function has
no retries and a timeout under 24 hours; calls that start more than 24 hours after launch refuse
work.
"""

import json
import time
from pathlib import Path

import modal

from workflows.checks import run_id
from workflows.images import add_project

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60

app = modal.App("simple-llm-sft-data")
volume = modal.Volume.from_name("lm-exploration-training")
runtime_image = add_project(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "gcc", "g++", "rustc", "golang-go", "default-jdk-headless", "nodejs", "node-typescript"
    )
    .env({"PYTHONPATH": "/project"})
)
data_image = add_project(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tokenizers==0.22.2", "huggingface-hub==1.33.0", "pyarrow==26.0.0")
    .env({"PYTHONPATH": "/project"})
)


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing data preparation")


def fresh_output(data_id):
    output = Path("/runs") / run_id(data_id, "data_id")
    if output.exists():
        raise ValueError("Use a fresh data ID")
    return output


def write(path, value):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


@app.function(image=runtime_image, cpu=2, memory=4096, timeout=1800, retries=0, max_containers=9)
def verify(language, deadline):
    import tempfile

    from training_data.repair_curriculum import execute, rows

    check_deadline(deadline)
    records = rows(language)
    for row in records:
        results = {}
        for variant in ("broken", "fixed"):
            with tempfile.TemporaryDirectory() as folder:
                results[variant] = execute(language, row[variant], row["cases"], folder)
        if results["broken"]["passed"] or not results["fixed"]["passed"]:
            raise ValueError(f"Oracle qualification failed: {row['id']}: {results}")
        row["verification"] = results
    return records


@app.local_entrypoint()
def curriculum(output: str):
    import tempfile

    from training_data.repair_curriculum import LANGUAGES, VERSION, rows, write_dataset
    from training_data.repair_sft import REPAIR
    from training_data.subword import digest, load_frozen

    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    with tempfile.TemporaryDirectory() as folder:
        candidate = Path(folder) / "candidate.jsonl"
        write_dataset(candidate, [row for lang in LANGUAGES for row in rows(lang)])
        REPAIR.prepare(
            candidate, load_frozen(), 1024
        )  # Reject invalid data before any remote work.
    records = [
        row for result in verify.map(LANGUAGES, kwargs={"deadline": deadline}) for row in result
    ]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_dataset(destination, records)
    splits, data_hash = REPAIR.prepare(destination, load_frozen(), 1024)
    report = {
        "version": VERSION,
        "dataset_sha256": data_hash,
        "generator_sha256": digest("data/repair_curriculum.py"),
        "languages": list(LANGUAGES),
        "counts": {k: len(v) for k, v in splits.items()},
        "verified_pairs": len(records),
        "synthetic": True,
    }
    write(destination.with_suffix(".report.json"), report)


@app.function(
    image=data_image, cpu=8, memory=32768, timeout=3600, retries=0, volumes={"/runs": volume}
)
def prepare_commitpack(data_id, excluded_codes, deadline):
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download
    from training_data.commitpack_sft import LANGUAGES, REPO, REVISION, build

    check_deadline(deadline)
    volume.reload()
    output = fresh_output(data_id)

    def download(lang):
        path = hf_hub_download(
            REPO, f"data/{lang}/data.jsonl", repo_type="dataset", revision=REVISION, token=False
        )
        return lang, path

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(download, LANGUAGES))
    report = build(files, output, excluded_codes=excluded_codes)
    volume.commit()
    return report


@app.local_entrypoint()
def commitpack(data_id: str, exclude_dataset: str = "", report: str = ""):
    """`--exclude-dataset` blocks a local JSONL's validation code from training."""
    run_id(data_id, "data_id")
    excluded = []
    if exclude_dataset:
        rows = [json.loads(line) for line in Path(exclude_dataset).read_text().splitlines()]
        excluded = [
            row[field]
            for row in rows
            if row["split"] == "validation"
            for field in ("broken", "fixed")
        ]
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    write(report, prepare_commitpack.remote(data_id, excluded, deadline))


@app.function(
    image=data_image,
    cpu=2,
    memory=16384,
    timeout=6 * 60 * 60,
    retries=0,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": volume},
)
def prepare_private_patches(data_id, sources, reserved, replay, full, deadline):
    from training_data.download_private_patches import download
    from training_data.private_patch_sft import build

    check_deadline(deadline)
    volume.reload()
    output = fresh_output(data_id)
    reserved_paths = [
        Path("/runs") / run_id(name, "reserved") / "examples.jsonl" for name in reserved
    ]
    replay_path = Path("/runs") / run_id(replay, "replay") / "examples.jsonl" if replay else None
    for path in [*reserved_paths, *([replay_path] if replay_path else [])]:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_path = Path("/tmp/private-patch-sources.json")
    source_path.write_text(json.dumps(sources))
    sample = Path("/tmp/private-patch-downloads")
    download(source_path, sample, per_repository=None if full else 24)
    records = [json.loads(line) for line in (sample / "downloaded.jsonl").read_text().splitlines()]
    failures = [r for r in records if "error" in r]
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(records)} source shards failed to download")
    check_deadline(deadline)
    caps = (
        {"train_cap": None, "language_cap": None, "repository_cap": None}
        if full
        else {"train_cap": 180000, "language_cap": 30000, "repository_cap": 12000}
    )
    result = build(
        sample / "downloaded.jsonl",
        output,
        reserved_paths=reserved_paths,
        replay_path=replay_path,
        **caps,
    )
    (output / "selection.json").write_bytes((sample / "selection.json").read_bytes())
    volume.commit()
    return {k: v for k, v in result.items() if k != "sources"}


@app.local_entrypoint()
def private_patches(
    sources: str,
    data_id: str,
    reserved: str = "",
    replay: str = "",
    full: bool = False,
    report: str = "",
):
    """`--reserved`: volume data IDs whose validation repositories/code are blocked."""
    run_id(data_id, "data_id")
    reserved_ids = [name for name in reserved.split(",") if name]
    for name in [*reserved_ids, *([replay] if replay else [])]:
        run_id(name, "reserved/replay data ID")
    source_list = json.loads(Path(sources).read_text())
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    result = prepare_private_patches.remote(
        data_id, source_list, reserved_ids, replay, full, deadline
    )
    write(report, result)


@app.function(
    image=data_image,
    cpu=4,
    memory=16384,
    timeout=3600,
    retries=0,
    volumes={"/runs": volume},
)
def prepare_math(data_id, deadline):
    from training_data.hub_files import pinned_rows
    from training_data.math_sft import GSM8K, METAMATH, SVAMP, build

    check_deadline(deadline)
    volume.reload()
    output = fresh_output(data_id)
    gsm8k_train = pinned_rows(*GSM8K, "main/train-00000-of-00001.parquet")
    test_questions = [
        r["question"] for r in pinned_rows(*GSM8K, "main/test-00000-of-00001.parquet")
    ]
    test_questions += [
        f"{r['Body']} {r['Question']}"
        for r in pinned_rows(*SVAMP, "data/test-00000-of-00001.parquet")
    ]
    metamath = pinned_rows(*METAMATH, "MetaMathQA-395K.json")
    report = build(gsm8k_train, metamath, test_questions, output)
    volume.commit()
    return report


@app.local_entrypoint()
def math(data_id: str, report: str = ""):
    run_id(data_id, "data_id")
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    write(report, prepare_math.remote(data_id, deadline))


def benchmark_problems(mbpp_splits=("test", "validation", "prompt")):
    """(math test questions, code benchmark problems) used for decontamination; MBPP
    contributes `mbpp_splits`."""
    from training_data.code_sft import HUMANEVAL_PLUS, MBPP, MBPP_PLUS, benchmark_texts
    from training_data.hub_files import pinned_rows
    from training_data.math_sft import GSM8K, SVAMP

    math = [r["question"] for r in pinned_rows(*GSM8K, "main/test-00000-of-00001.parquet")]
    math += [
        f"{r['Body']} {r['Question']}"
        for r in pinned_rows(*SVAMP, "data/test-00000-of-00001.parquet")
    ]
    held_out = [
        row
        for split in mbpp_splits
        for row in pinned_rows(*MBPP, f"full/{split}-00000-of-00001.parquet")
    ]
    code = benchmark_texts(
        pinned_rows(*MBPP_PLUS, "data/test-00000-of-00001-d5781c9c51e02795.parquet"),
        pinned_rows(*HUMANEVAL_PLUS, "data/test-00000-of-00001-5973903632b82d40.parquet"),
        held_out,
    )
    return math, code


@app.function(
    image=data_image,
    cpu=4,
    memory=32768,
    timeout=3600,
    retries=0,
    volumes={"/runs": volume},
)
def prepare_code(data_id, deadline):
    from training_data.code_sft import MBPP, OPENCODEINSTRUCT, SFT_SHARDS, build, shard_name
    from training_data.hub_files import pinned_rows

    check_deadline(deadline)
    volume.reload()
    output = fresh_output(data_id)

    def shards():
        for index in SFT_SHARDS:
            check_deadline(deadline)
            yield pinned_rows(*OPENCODEINSTRUCT, shard_name(index))

    mbpp_train = pinned_rows(*MBPP, "full/train-00000-of-00001.parquet")
    report = build(shards(), mbpp_train, benchmark_problems()[1], output)
    volume.commit()
    return report


@app.function(
    image=data_image, cpu=2, memory=8192, timeout=1800, retries=0, volumes={"/runs": volume}
)
def prepare_mixture(data_id, datasets, deadline):
    from training_data.mixture_sft import combine
    from training_data.subword import digest

    check_deadline(deadline)
    volume.reload()
    output = fresh_output(data_id)
    paths = {}
    for task, source in datasets.items():
        folder = Path("/runs") / run_id(source, "data_id")
        report = json.loads((folder / "report.json").read_text())
        if digest(folder / "examples.jsonl") != report["dataset_sha256"]:
            raise ValueError(f"{source} differs from its preparation report")
        paths[task] = folder / "examples.jsonl"
    report = combine(paths, output)
    report["data_ids"] = datasets
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.function(
    image=data_image, cpu=2, memory=16384, timeout=1800, retries=0, volumes={"/runs": volume}
)
def prepare_rl_pool(data_id, deadline, math_lighteval=False, kodcode=False, code_train=10000):
    from training_data.code_sft import MBPP, OPENCODEINSTRUCT, RL_SHARD, shard_name
    from training_data.hub_files import pinned_rows
    from training_data.math_sft import GSM8K
    from training_data.rl_tasks import KODCODE, MATH_LIGHTEVAL, build
    from training_data.rl_trajectories import SANDBOX_BATCH
    from training_data.sandbox import Sandbox

    check_deadline(deadline)
    volume.reload()
    output = fresh_output(data_id)
    # MBPP validation problems are the pool's code validation set.
    math_tests, code_tests = benchmark_problems(("test", "prompt"))
    sandbox = Sandbox(app, min(1800, deadline - time.time())) if kodcode else None

    def check(pairs):
        return [
            result
            for start in range(0, len(pairs), SANDBOX_BATCH)
            for result in sandbox.run(pairs[start : start + SANDBOX_BATCH])
        ]

    try:
        report = build(
            pinned_rows(*GSM8K, "main/train-00000-of-00001.parquet"),
            pinned_rows(*OPENCODEINSTRUCT, shard_name(RL_SHARD)),
            pinned_rows(*MBPP, "full/validation-00000-of-00001.parquet"),
            math_tests,
            code_tests,
            output,
            math_lighteval=pinned_rows(*MATH_LIGHTEVAL, "data/train-00000-of-00001.parquet")
            if math_lighteval
            else (),
            kodcode=pinned_rows(*KODCODE, "data/train-00000-of-00001.parquet") if kodcode else (),
            check=check,
            code_train=code_train,
        )
    finally:
        if sandbox is not None:
            sandbox.close()
    volume.commit()
    return report


@app.local_entrypoint()
def code(data_id: str, report: str = ""):
    run_id(data_id, "data_id")
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    write(report, prepare_code.remote(data_id, deadline))


@app.local_entrypoint()
def mixture(data_id: str, math: str, code: str, report: str = ""):
    run_id(data_id, "data_id")
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    write(report, prepare_mixture.remote(data_id, {"math": math, "code": code}, deadline))


@app.local_entrypoint()
def rl_pool(
    data_id: str,
    report: str = "",
    math_lighteval: bool = False,
    kodcode: bool = False,
    code_train: int = 10000,
):
    """`--math-lighteval` and `--kodcode` add MATH levels 1-3 and KodCode-Light-RL easy and
    medium problems; `--code-train` caps the OpenCodeInstruct training problems (0 for
    none). The hybrid pool (`workflows/hybrid-720m-rl.json`) is GSM8K, MATH and KodCode:
    `--math-lighteval --kodcode --code-train 0`."""
    run_id(data_id, "data_id")
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    write(report, prepare_rl_pool.remote(data_id, deadline, math_lighteval, kodcode, code_train))
