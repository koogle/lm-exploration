"""Replace code SFT solutions with verified teacher solutions (`training_data.teacher`).

Run from training/:
    modal run -m modal_scripts.modal_teacher --source <code-data-id> --data-id <fresh-id> \
        [--shards 8] [--samples 2] [--limit 0] [--report <report.json>]

One CPU orchestrator call reads `/runs/<source>/examples.jsonl` (a `code-sft-v1` dataset),
spawns one H100 vLLM call per shard (after one call caches the teacher weights in the
`lm-exploration-hf-cache` volume) of problems, runs every answer against its problem's
tests in a network-blocked code sandbox, and writes `examples.jsonl` and `report.json` to
`/runs/<data-id>`. Each shard's teacher answers and kept rows are saved under
`/runs/teacher-work/<data-id>/` as soon as they exist, so rerunning the same command
after a failure reuses them instead of generating again. The launcher records a deadline 24 hours after launch: the
orchestrator refuses work after it, spawns a shard only if the shard's timeout ends
before it, cancels its unfinished shard calls when it stops, and its sandbox's own
timeout ends at the deadline. No call retries.
"""

import json
import time
from pathlib import Path

import modal

from workflows.checks import run_id

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
SHARD_SECONDS = 4 * 60 * 60
ORCHESTRATOR_SECONDS = 20 * 60 * 60
MAX_NEW_TOKENS = 1024
# Programs per sandbox request.
SANDBOX_BATCH = 2000

app = modal.App("simple-llm-teacher")
volume = modal.Volume.from_name("lm-exploration-training")
weights = modal.Volume.from_name("lm-exploration-hf-cache", create_if_missing=True)
# vLLM's FlashInfer kernels compile at load time, so the image carries nvcc (CUDA 13.0,
# matching vLLM 0.31's torch 2.13).
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.31.0", "huggingface-hub[hf_transfer]")
    # DeepGEMM's FP8 weight preprocessing fails for this model; CUTLASS FP8 runs it.
    .env(
        {
            "HF_HOME": "/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/project",
            "VLLM_USE_DEEP_GEMM": "0",
            "CUDA_HOME": "/usr/local/cuda",
        }
    )
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("trainer", "/project/trainer", ignore=["__pycache__", "kernels"])
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTHONPATH": "/project"})
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("trainer", "/project/trainer", ignore=["__pycache__", "kernels"])
)


@app.function(image=vllm_image, cpu=4, timeout=3600, retries=0, volumes={"/hf": weights})
def fetch():
    """Download the pinned teacher weights into the cache volume once."""
    from huggingface_hub import snapshot_download
    from training_data.teacher import TEACHER

    snapshot_download(TEACHER[0], revision=TEACHER[1])
    weights.commit()


@app.function(
    image=vllm_image,
    gpu="H100",
    timeout=SHARD_SECONDS,
    retries=0,
    max_containers=8,
    volumes={"/hf": weights},
)
def generate(rows, samples, deadline):
    """Each row's `samples` teacher answers (text)."""
    from training_data.teacher import TEACHER, messages
    from vllm import LLM, SamplingParams

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing generation")
    llm = LLM(model=TEACHER[0], revision=TEACHER[1], max_model_len=4096, enable_prefix_caching=True)
    params = SamplingParams(n=samples, temperature=0.7, top_p=0.95, max_tokens=MAX_NEW_TOKENS)
    outputs = llm.chat(
        [messages(row) for row in rows],
        params,
        chat_template_kwargs={"enable_thinking": False},
    )
    return [[choice.text for choice in output.outputs] for output in outputs]


@app.function(
    image=cpu_image,
    cpu=2,
    memory=16384,
    timeout=ORCHESTRATOR_SECONDS,
    retries=0,
    volumes={"/runs": volume},
)
def distill(source, data_id, shards, samples, limit, deadline):
    import hashlib

    from training_data.sandbox import Sandbox
    from training_data.subword import digest
    from training_data.teacher import TEACHER, choose, extract

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing distillation")
    volume.reload()
    folder = Path("/runs") / run_id(source, "source")
    if (
        digest(folder / "examples.jsonl")
        != json.loads((folder / "report.json").read_text())["dataset_sha256"]
    ):
        raise ValueError("Source differs from its preparation report")
    output = Path("/runs") / run_id(data_id, "data_id")
    if output.exists():
        raise ValueError("Use a fresh data ID")
    rows = [json.loads(line) for line in (folder / "examples.jsonl").read_text().splitlines()]
    rows = rows[:limit] if limit else rows
    size = -(-len(rows) // shards)
    chunks = [rows[i : i + size] for i in range(0, len(rows), size)]
    work = Path("/runs/teacher-work") / data_id
    work.mkdir(parents=True, exist_ok=True)
    settings = {"source": digest(folder / "examples.jsonl"), "shards": shards}
    settings.update(samples=samples, limit=limit, teacher=list(TEACHER))
    if (work / "settings.json").exists():
        if json.loads((work / "settings.json").read_text()) != settings:
            raise ValueError("Saved teacher work was made with other settings")
    else:
        (work / "settings.json").write_text(json.dumps(settings))
    pending = [i for i in range(len(chunks)) if not (work / f"answers-{i}.json").exists()]
    calls = {}
    if pending:
        if time.time() + SHARD_SECONDS >= deadline:
            raise TimeoutError("Not enough launch budget left for a generation shard")
        fetch.remote()
        calls = {i: generate.spawn(chunks[i], samples, deadline) for i in pending}
    sandbox = Sandbox(app, deadline - time.time(), cpu=32)
    kept, counts = [], {"problems": len(rows), "kept": 0, "no_code": 0, "no_pass": 0}
    try:
        for index, chunk in enumerate(chunks):
            done = work / f"kept-{index}.json"
            if done.exists():
                saved = json.loads(done.read_text())
                kept += saved["rows"]
                for key, value in saved["counts"].items():
                    counts[key] += value
                continue
            answers_path = work / f"answers-{index}.json"
            if index in calls:
                answers_path.write_text(json.dumps(calls[index].get()))
                volume.commit()
            answers = json.loads(answers_path.read_text())
            shard = {"kept": 0, "no_code": 0, "no_pass": 0}
            shard_rows = []
            codes = [[extract(text, row) for text in texts] for row, texts in zip(chunk, answers)]
            pairs = [
                (code, row["tests"])
                for row, options in zip(chunk, codes, strict=True)
                for code in options
                if code is not None
            ]
            results = iter(
                [
                    result
                    for start in range(0, len(pairs), SANDBOX_BATCH)
                    for result in sandbox.run(pairs[start : start + SANDBOX_BATCH])
                ]
            )
            for row, options in zip(chunk, codes, strict=True):
                found = [next(results) for code in options if code is not None]
                chosen = choose(row, options, found)
                if chosen:
                    shard_rows.append(chosen)
                    shard["kept"] += 1
                elif not found:
                    shard["no_code"] += 1
                else:
                    shard["no_pass"] += 1
            done.write_text(json.dumps({"rows": shard_rows, "counts": shard}))
            volume.commit()
            kept += shard_rows
            for key, value in shard.items():
                counts[key] += value
            print(json.dumps({"shard": index, **counts}), flush=True)
    finally:
        for call in calls.values():
            call.cancel()
        sandbox.close()
    output.mkdir(parents=True)
    path = output / "examples.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    report = {
        "format": "code-sft-v1",
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": {"data_id": source, "sha256": digest(folder / "examples.jsonl")},
        "teacher": TEACHER,
        "samples": samples,
        "selection": "shortest answer passing every test",
        "splits": {
            split: sum(r["split"] == split for r in kept) for split in ("train", "validation")
        },
        "counts": counts,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main(
    source: str, data_id: str, shards: int = 8, samples: int = 2, limit: int = 0, report: str = ""
):
    run_id(data_id, "data_id")
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    result = distill.remote(source, data_id, shards, samples, limit, deadline)
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
