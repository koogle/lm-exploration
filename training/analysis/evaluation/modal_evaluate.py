"""Benchmark training checkpoints from the run volume on one Modal H100.

Run from training/:
    modal run -m evaluation.modal_evaluate --name NAME --snapshot ROOT/VERSION/snapshots/RUN \
        --checkpoints RUN-sNN/step-N.pt[,RUN-sMM/step-M.pt ...] [--output result.json]

Each checkpoint loads the model code its own run copied to `RUN-sNN/source`, verified
against the source hashes the checkpoint recorded, so any checkout can evaluate it. One
function call, a 3-hour hard timeout and no retries; the call refuses new work 24 hours
after launch, including time spent queued. Results go to `/runs/evaluation/NAME.json`
after each checkpoint and to `--output` (default `artifacts/evaluation/NAME.json`).
"""

import hashlib
import json
import time
from pathlib import Path

import modal

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
TIMEOUT_SECONDS = 3 * 60 * 60

# (repository, pinned revision, evaluated file, exemplar file for few-shot tasks)
DATASETS = {
    "hellaswag": (
        "Rowan/hellaswag",
        "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "data/validation-00000-of-00001.parquet",
        None,
    ),
    "arc_easy": (
        "allenai/ai2_arc",
        "210d026faf9955653af8916fad021475a3f00453",
        "ARC-Easy/test-00000-of-00001.parquet",
        None,
    ),
    "arc_challenge": (
        "allenai/ai2_arc",
        "210d026faf9955653af8916fad021475a3f00453",
        "ARC-Challenge/test-00000-of-00001.parquet",
        None,
    ),
    "openbookqa": (
        "allenai/openbookqa",
        "388097ea7776314e93a529163e0fea805b8a6454",
        "main/test-00000-of-00001.parquet",
        None,
    ),
    "winogrande": (
        "allenai/winogrande",
        "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "winogrande_xl/validation-00000-of-00001.parquet",
        None,
    ),
    "lambada": (
        "EleutherAI/lambada_openai",
        "900124bf3b8235c6daf21033af9948b3f07346c4",
        "data/lambada_test.jsonl",
        None,
    ),
    "gsm8k": (
        "openai/gsm8k",
        "740312add88f781978c0658806c59bc2815b9866",
        "main/test-00000-of-00001.parquet",
        "main/train-00000-of-00001.parquet",
    ),
    "svamp": (
        "ChilleD/SVAMP",
        "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef",
        "data/test-00000-of-00001.parquet",
        "data/train-00000-of-00001.parquet",
    ),
}
SETTINGS = {
    "precision": "bf16-autocast",
    "heldout_windows_per_source": 64,
    "shots": 5,
    "gsm8k_examples": 500,
    "gsm8k_max_new_tokens": 256,
    "svamp_max_new_tokens": 32,
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "numpy",
        "pyarrow==26.0.0",
        "pydantic>=2.11,<3",
        "tokenizers==0.22.2",
        "huggingface-hub==1.33.0",
    )
    .env({"PYTHONPATH": "/project"})
    .add_local_dir("analysis/evaluation", "/project/evaluation", ignore=["__pycache__"])
)
app = modal.App("checkpoint-evaluation")
run_volume = modal.Volume.from_name("lm-exploration-training")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def load_datasets():
    from huggingface_hub import hf_hub_download

    loaded = {}
    for task, (repo, revision, evaluated, exemplars) in DATASETS.items():

        def fetch(name, repo=repo, revision=revision):
            return rows(hf_hub_download(repo, name, repo_type="dataset", revision=revision))

        loaded[task] = (fetch(evaluated), fetch(exemplars) if exemplars else [])
    return loaded


def heldout_documents(snapshot, tokenizer, count):
    """Tokenized validation documents per source, in the raw loader's hash order."""
    import sqlite3
    import zlib

    manifest = json.loads((snapshot / "manifest.json").read_text())
    database = snapshot / manifest["database"]["path"]
    db = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    documents = {}
    try:
        for source in manifest["sources"]:
            cursor = db.execute(
                """SELECT text FROM documents WHERE source=? AND split='validation'
                AND conflict=0 ORDER BY hash""",
                (source,),
            )
            texts = []
            for (blob,) in cursor:
                texts.append(zlib.decompress(blob).decode())
                # Enough documents for `count` windows, with a generous margin.
                if sum(len(t) for t in texts) > count * 2048 * 8:
                    break
            encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
            documents[source] = [e.ids for e in encoded]
    finally:
        db.close()
    return manifest, documents


def open_checkpoint(path):
    """A verified model from a run's checkpoint, using the model code that run copied."""
    import importlib
    import sys
    from types import SimpleNamespace

    import torch
    from tokenizers import Tokenizer

    run = path.parent
    source = run / "source"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    architecture = saved["architecture"]
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module(architecture["module"])
    if not Path(module.__file__).is_relative_to(source):
        raise ValueError("Model code must come from the checkpoint's own run")
    definitions = {"model": module.__file__, **getattr(module, "MODEL_DEPENDENCIES", {})}
    for key, file in definitions.items():
        if sha256(file) != saved["source_sha256"][key]:
            raise ValueError(f"Model source differs from the checkpoint: {key}")
    tokenizer_path = run / "tokenizer" / "tokenizer.json"
    if sha256(tokenizer_path) != saved["data_identity"]["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the checkpoint")
    model = module.GPT(module.ModelConfig(**architecture["config"]))
    model.load_state_dict(saved["model"], strict=True)
    model = model.cuda().eval()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Non-finite parameters")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    special = saved["special_tokens"]

    def forward(tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(tokens)

    info = {
        "checkpoint": str(path.relative_to("/runs")),
        "checkpoint_sha256": sha256(path),
        "step": saved.get("step"),
        "trained_tokens": saved.get("tokens"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_source_sha256": saved["source_sha256"],
        "tokenizer_sha256": saved["data_identity"]["tokenizer_sha256"],
    }
    wrapper = SimpleNamespace(
        forward=forward,
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        bos_token_id=special["bos"],
        eos_token_id=special["eos"],
        # Padding sits after every scored position; PAD is ID 0 in our tokenizers.
        pad_token_id=0,
        block_size=architecture["config"]["block_size"],
        device="cuda",
    )
    return info, wrapper, tokenizer


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing further evaluation")


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
    volumes={"/runs": run_volume, "/data": data_volume},
)
def evaluate(name, snapshot, checkpoints, deadline):
    import torch

    from evaluation import tasks

    check_deadline(deadline)
    output = Path("/runs/evaluation") / f"{name}.json"
    result = {
        "name": name,
        "snapshot": snapshot,
        "settings": SETTINGS,
        "datasets": {
            task: {"repo": r, "revision": v, "file": f, "exemplars": e}
            for task, (r, v, f, e) in DATASETS.items()
        },
        "tasks_sha256": sha256("/project/evaluation/tasks.py"),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "models": [],
    }
    data = load_datasets()
    heldout = None
    for checkpoint in checkpoints:
        check_deadline(deadline)
        started = time.time()
        info, model, tokenizer = open_checkpoint(Path("/runs") / checkpoint)
        print(json.dumps(info), flush=True)
        if heldout is None:
            manifest, documents = heldout_documents(
                Path("/data") / snapshot, tokenizer, SETTINGS["heldout_windows_per_source"]
            )
            heldout = {
                source: tasks.packed_windows(
                    ids,
                    model.bos_token_id,
                    model.eos_token_id,
                    model.block_size,
                    SETTINGS["heldout_windows_per_source"],
                )
                for source, ids in documents.items()
            }
            result["snapshot_weights"] = {
                source: entry["weight_percent"] for source, entry in manifest["sources"].items()
            }
        scores = {"heldout": {s: tasks.bits_per_byte(model, w) for s, w in heldout.items()}}
        for task in ("hellaswag", "arc_easy", "arc_challenge", "openbookqa"):
            scores[task] = tasks.score_multiple_choice(model, task, data[task][0])
        scores["winogrande"] = tasks.score_winogrande(model, data["winogrande"][0])
        scores["lambada"] = tasks.score_lambada(model, data["lambada"][0])
        answers = {}
        for task, count, tokens in (
            ("gsm8k", SETTINGS["gsm8k_examples"], SETTINGS["gsm8k_max_new_tokens"]),
            ("svamp", None, SETTINGS["svamp_max_new_tokens"]),
        ):
            check_deadline(deadline)
            evaluated, exemplars = data[task]
            scores[task], answers[task] = tasks.score_math(
                model, task, evaluated[:count], exemplars[: SETTINGS["shots"]], tokens
            )
        result["models"].append(
            {**info, "scores": scores, "answers": answers, "seconds": time.time() - started}
        )
        print(json.dumps({"checkpoint": checkpoint, "scores": scores}), flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        run_volume.commit()
        del model
        torch.cuda.empty_cache()
    return result


@app.local_entrypoint()
def main(name: str, snapshot: str, checkpoints: str, output: str = ""):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    result = evaluate.remote(name, snapshot, checkpoints.split(","), deadline)
    path = Path(output or f"artifacts/evaluation/{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(path)}))
