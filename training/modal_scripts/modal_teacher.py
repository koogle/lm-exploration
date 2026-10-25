"""Replace code SFT solutions with verified teacher solutions (`training_data.teacher`).

Run from training/:
    modal run -m modal_scripts.modal_teacher --source <code-data-id> --data-id <fresh-id> \
        [--shards 8] [--samples 2] [--limit 0] [--report <report.json>]

One CPU orchestrator call reads `/runs/<source>/examples.jsonl` (a `code-sft-v1` dataset),
spawns one H100 vLLM call per shard (after one call caches the teacher weights in the
`lm-exploration-hf-cache` volume) of problems, runs every answer against its problem's
tests in a network-blocked code sandbox, and writes `examples.jsonl` and `report.json` to
`/runs/<data-id>`. The launcher records a deadline 24 hours after launch: the
orchestrator refuses work after it, spawns a shard only if the shard's timeout ends
before it, cancels its unfinished shard calls when it stops, and its sandbox's own
timeout ends at the deadline. No call retries.
"""

import json
import time
from pathlib import Path

import modal

from modal_scripts.modal_sft_data import data_image, fresh_output, write
from trainer.sft_recipe import run_id

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
SHARD_SECONDS = 4 * 60 * 60
ORCHESTRATOR_SECONDS = 20 * 60 * 60
MAX_NEW_TOKENS = 1024
# Programs per sandbox request.
SANDBOX_BATCH = 2000

app = modal.App("simple-llm-teacher")
volume = modal.Volume.from_name("lm-exploration-training")
weights = modal.Volume.from_name("lm-exploration-hf-cache", create_if_missing=True)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.31.0", "huggingface-hub[hf_transfer]")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/project"})
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
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
    image=data_image,
    cpu=2,
    memory=16384,
    timeout=ORCHESTRATOR_SECONDS,
    retries=0,
    volumes={"/runs": volume},
)
def distill(source, data_id, shards, samples, limit, deadline):
    import hashlib

    from training_data.code_execution import Sandbox
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
    output = fresh_output(data_id)
    rows = [json.loads(line) for line in (folder / "examples.jsonl").read_text().splitlines()]
    rows = rows[:limit] if limit else rows
    if time.time() + SHARD_SECONDS >= deadline:
        raise TimeoutError("Not enough launch budget left for a generation shard")
    fetch.remote()
    size = -(-len(rows) // shards)
    calls = [
        generate.spawn(rows[i : i + size], samples, deadline) for i in range(0, len(rows), size)
    ]
    sandbox = Sandbox(app, deadline - time.time(), cpu=32)
    kept, counts = [], {"problems": len(rows), "kept": 0, "no_code": 0, "no_pass": 0}
    try:
        for index, call in enumerate(calls):
            answers = call.get()
            chunk = rows[index * size : (index + 1) * size]
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
                    kept.append(chosen)
                    counts["kept"] += 1
                elif not found:
                    counts["no_code"] += 1
                else:
                    counts["no_pass"] += 1
            print(json.dumps({"shard": index, **counts}), flush=True)
    finally:
        for call in calls:
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
    write(report, distill.remote(source, data_id, shards, samples, limit, deadline))
