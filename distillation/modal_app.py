"""Distillation training, evaluation and generations on Modal. Importing starts no job.

Run from the repository root (see distillation/README.md for recipes):
    modal run --detach -m distillation.modal_app --recipe <recipe.json> \
        [--run-id <fresh-id>] [--parent <run/step-N.pt>] [--resume <run/step-N.pt>] \
        [--output <result.json>]

Recipe kinds `byte`, `shared`, `students`, `evaluate` and `generations` run here; `curriculum`
runs in `modal_vllm.py`. Each launch makes one function call with no retries. The launcher
records a deadline 24 hours after launch. Short calls refuse to start unless their whole
function timeout fits before it; the 24-hour `shared` trainer instead caps its cooperative
training time to stop 30 minutes before the deadline. Outputs go to the
`lm-exploration-distillation` volume, and the returned report to the ignored
`distillation/artifacts/`.
"""

import json
import time
from pathlib import Path

import modal

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
FINAL_RESERVE_SECONDS = 30 * 60
TIMEOUTS = {
    "byte": 30 * 60,
    "shared": 24 * 60 * 60,
    "students": 2 * 60 * 60,
    "evaluate": 15 * 60,
    "generations": 15 * 60,
}

app = modal.App("lm-exploration-distillation", include_source=False)
artifacts = modal.Volume.from_name(
    "lm-exploration-distillation", create_if_missing=True
)
mixture = modal.Volume.from_name("lm-exploration-mixture-data")
training_runs = modal.Volume.from_name("lm-exploration-training")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0", "transformers==4.57.1", "tokenizers==0.22.2", "numpy==2.2.6"
    )
    .env({"PYTHONPATH": "/project", "HF_HOME": "/distill/hf-cache"})
    .add_local_dir(
        "distillation", "/project/distillation", ignore=["artifacts", "__pycache__"]
    )
    .add_local_dir("training/100m", "/project/model_100m")
    .add_local_dir("training/10m", "/project/model_10m")
    .add_local_dir(
        "training/data", "/project/training_data", ignore=["eu", "__pycache__"]
    )
    .add_local_dir(
        "training/tokenizer/structured-code-v1", "/project/tokenizer/structured-code-v1"
    )
    .add_local_dir(
        "training/analysis/sampling",
        "/project/sampling",
        ignore=["recipes", "__pycache__"],
    )
)
all_volumes = {"/distill": artifacts, "/data": mixture, "/runs": training_runs}


def start(kind, deadline):
    """Seconds left before the launch deadline; refuse calls whose timeout no longer fits."""
    remaining = deadline - time.time()
    if kind == "shared":
        remaining -= FINAL_RESERVE_SECONDS
        if remaining <= 0:
            raise TimeoutError("Launch deadline passed; refusing distillation work")
    elif remaining < TIMEOUTS[kind]:
        raise TimeoutError(f"{kind} timeout no longer fits before the launch deadline")
    return remaining


def kl_training(trainer, recipe, deadline):
    from distillation.recipe import validate_recipe

    spec = validate_recipe(recipe)
    remaining = start(spec["kind"], deadline)
    try:
        return trainer(
            "/distill/" + spec["run_id"],
            artifacts.commit,
            teacher_repo=spec["teacher"]["repo"],
            revision=spec["teacher"]["revision"],
            parent=spec["parent"],
            manifest="/data/" + spec["data"]["manifest"],
            database="/data/" + spec["data"]["database"],
            target_tokens=spec["target_tokens"],
            max_seconds=int(min(spec["max_seconds"], remaining)),
            resume=spec["resume"],
        )
    finally:
        artifacts.commit()


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    timeout=TIMEOUTS["byte"],
    retries=0,
    max_containers=1,
    volumes=all_volumes,
)
def byte(recipe, deadline):
    from distillation.train import run

    return kl_training(run, recipe, deadline)


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=TIMEOUTS["shared"],
    retries=0,
    max_containers=1,
    volumes=all_volumes,
)
def shared(recipe, deadline):
    from distillation.shared_train import run

    return kl_training(run, recipe, deadline)


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    timeout=TIMEOUTS["students"],
    retries=0,
    max_containers=1,
    volumes={"/distill": artifacts},
)
def students(recipe, deadline):
    from distillation.curriculum_train import run
    from distillation.recipe import validate_recipe

    spec = validate_recipe(recipe)
    start("students", deadline)
    corpus = Path("/distill") / spec["corpus"]
    if json.loads((corpus / "report.json").read_text())["status"] != "completed":
        raise ValueError("Corpus generation must be complete")
    try:
        return run(corpus, Path("/distill") / spec["run_id"], artifacts.commit)
    finally:
        artifacts.commit()


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=TIMEOUTS["evaluate"],
    retries=0,
    max_containers=1,
    volumes=all_volumes,
)
def evaluate(recipe, deadline):
    from distillation.evaluate import compare
    from distillation.recipe import validate_recipe

    spec = validate_recipe(recipe)
    start("evaluate", deadline)
    mounts = {"training": "/runs/", "distillation": "/distill/"}
    checkpoints = [
        mounts[v] + path
        for v, _, path in (c.partition(":") for c in spec["checkpoints"])
    ]
    output = Path("/distill") / spec["run_id"]
    output.mkdir(exist_ok=False)
    report = compare(
        checkpoints,
        "/data/" + spec["data"]["manifest"],
        "/data/" + spec["data"]["database"],
        spec["context"],
        spec["max_documents"],
        spec["windows_per_document"],
    )
    report["recipe"] = spec
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    artifacts.commit()
    return report


def checkpoint_tokenizer(relative, digest):
    """(prefix ids, EOS id, encode, decode) for a checkpoint's recorded tokenizer hash."""
    import hashlib

    from training_data.subword import TOKENIZER_SHA256, load_frozen
    from transformers import AutoTokenizer

    if digest == TOKENIZER_SHA256:
        frozen = load_frozen()

        def encode(text):
            return frozen.encode(text, add_special_tokens=False).ids

        def decode(ids):
            return frozen.decode(ids, skip_special_tokens=False)

        return [1], 2, encode, decode
    # Shared-tokenizer runs archive their tokenizer at the run root.
    folder = Path("/distill") / relative.split("/")[0] / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(folder)
    if (
        hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        != digest
    ):
        raise ValueError(f"{relative}: checkpoint/tokenizer mismatch")

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(ids):
        return tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    eos = tokenizer.eos_token_id
    return [eos], eos, encode, decode


def open_checkpoints(spec):
    """Yield (info, model) for the sampling toolkit from distillation-volume checkpoints."""
    from types import SimpleNamespace

    import torch
    from model_100m.model import GPT, ModelConfig

    for relative in spec["checkpoints"]:
        saved = torch.load(
            Path("/distill") / relative, map_location="cpu", weights_only=False
        )
        digest = saved["settings"]["tokenizer_sha256"]
        prefix, eos, encode, decode = checkpoint_tokenizer(relative, digest)
        model = GPT(ModelConfig(**saved["model_config"])).cuda().eval()
        model.load_state_dict(saved["model"], strict=True)
        info = {
            "name": relative.removesuffix(".pt"),
            "kind": "distillation-checkpoint",
            "checkpoint": relative,
            "format": saved.get("format"),
            "step": saved.get("step"),
            "trained_tokens": saved.get("tokens"),
            "parameters": sum(p.numel() for p in model.parameters()),
            "tokenizer_sha256": digest,
            "prefix_token_ids": prefix,
            "eos_token_id": eos,
        }
        del saved

        def encode_prompt(text, encode=encode, prefix=prefix, model=model):
            ids = prefix + encode(text)
            if len(ids) + spec["max_new_tokens"] > model.config.block_size:
                raise ValueError("Prompt plus max_new_tokens exceeds the model context")
            return ids

        yield (
            info,
            SimpleNamespace(
                forward=model,
                encode=encode_prompt,
                decode=decode,
                eos_token_id=eos,
                device="cuda",
            ),
        )
        del model
        torch.cuda.empty_cache()


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=TIMEOUTS["generations"],
    retries=0,
    max_containers=1,
    volumes={"/distill": artifacts},
)
def generations(recipe, deadline):
    import torch
    from sampling.sampler import run_recipe

    from distillation.recipe import validate_recipe

    spec = validate_recipe(recipe)
    start("generations", deadline)
    torch.set_num_threads(4)
    output = Path("/distill") / spec["run_id"]
    output.mkdir(exist_ok=False)
    # One "model" entry lists every checkpoint so the toolkit's resume logic is not needed.
    sampling = {**spec, "models": [{"name": spec["run_id"]}]}

    def save(report):
        report["device"] = torch.cuda.get_device_name()
        (output / "generations.json").write_text(json.dumps(report, indent=2) + "\n")
        artifacts.commit()

    report = run_recipe(
        sampling, lambda _spec, _done: open_checkpoints(spec), deadline, save=save
    )
    report["recipe"] = spec
    save(report)
    return report


WORKERS = {
    "byte": byte,
    "shared": shared,
    "students": students,
    "evaluate": evaluate,
    "generations": generations,
}


@app.local_entrypoint()
def main(
    recipe: str, run_id: str = "", parent: str = "", resume: str = "", output: str = ""
):
    from distillation.recipe import load_recipe

    spec = load_recipe(recipe, run_id, parent, resume)
    if spec["kind"] not in WORKERS:
        raise ValueError(f"{spec['kind']} recipes run with distillation.modal_vllm")
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    result = WORKERS[spec["kind"]].remote(spec, deadline)
    path = Path(output or f"distillation/artifacts/{spec['run_id']}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"recipe": spec, "result": result}, indent=2) + "\n")
    print(json.dumps({"output": str(path)}))
