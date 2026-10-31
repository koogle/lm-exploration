"""LiveCodeBench v6 pass@1 for checkpoints of any model family and for reference models.

Run from training/:
    modal run --detach -m evaluation.modal_livecodebench \
        [--checkpoints runs/RUN/step-N.pt,training/RUN/step-M.pt] [--references M,...|all] \
        [--limit N]

A checkpoint path starts with the volume it is on: `runs/` (lm-exploration-hybrid-runs)
or `training/` (lm-exploration-training). Checkpoints load through their family's
adapter (`trainer/families.py`); pretrained checkpoints get LiveCodeBench's plain prompt
(`base_prompt`) and post-trained ones its instruction as the family's user turn.
Reference models (`--references`, Hugging Face base models at pinned revisions) get the
plain prompt through transformers.

One H100 call per model, in parallel. Each call downloads the pinned release file,
generates one greedy completion per problem (up to `MAX_NEW_TOKENS`) and grades it inside
the call's container with `livecodebench.grade_all`. Each call has `retries=0`, a 6-hour
timeout and refuses to start 24 hours after launch; an independent supervisor cancels
any call still running at that deadline. Results (summary plus per-problem completions
and scores) go to `lm-exploration-hybrid-runs/benchmarks/<launch>/lcb-<model>.json`.
"""

import json
import time
from pathlib import Path

import modal
from model_hybrid import modal_benchmarks
from model_hybrid.image import LAUNCH_LIMIT_SECONDS
from model_hybrid.modal_benchmarks import REFERENCES, enforce_deadline, image, runs_volume, secret

app = modal.App("lm-exploration-livecodebench")
app.include(modal_benchmarks.app)  # the deadline supervisor
training_volume = modal.Volume.from_name("lm-exploration-training")
# Checkpoint paths start with one of these mounts.
MOUNTS = {"runs": runs_volume, "training": training_volume}
DATASET = "livecodebench/code_generation_lite"
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
RELEASE_FILE = "test6.jsonl"
MAX_NEW_TOKENS = 2048
BATCH_ROWS = 32
# Larger dense Qwen base models, beyond the shared benchmark references.
LARGER_QWEN = {
    "Qwen/Qwen3-1.7B-Base": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    "Qwen/Qwen3-4B-Base": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
    "Qwen/Qwen3-8B-Base": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
    "Qwen/Qwen3-14B-Base": "0b0bd3732e2c374d483664439ea334928b65f304",
    "Qwen/Qwen3.5-2B-Base": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
    "Qwen/Qwen3.5-4B-Base": "1001bb4d826a52d1f399e183466143f4da7b741b",
    "Qwen/Qwen3.5-9B-Base": "68c46c4b3498877f3ef123c856ecfde50c39f404",
}
ALL_REFERENCES = {**REFERENCES, **LARGER_QWEN}


def checkpoint_path(name):
    """The container path of `name` (`runs/...` or `training/...`)."""
    mount = name.split("/", 1)[0]
    if mount not in MOUNTS or ".." in Path(name).parts:
        raise ValueError(f"A checkpoint path starts with one of {sorted(MOUNTS)}: {name}")
    return Path("/") / name


def _checkpoint_completions(name, problems):
    from evaluation.benchmarks import livecodebench
    from trainer import checkpoints

    saved, description, family = checkpoints.parent(checkpoint_path(name))
    model = family.build(description.model_config).cuda()
    family.load(model, saved["model"])
    del saved
    model.eval()
    tokenizer = family.tokenizer()
    if description.stage == "pretrain":
        prompts = [family.text_ids(tokenizer, livecodebench.base_prompt(p)) for p in problems]
    else:
        prompts = [family.prompt_ids(tokenizer, livecodebench.instruction(p)) for p in problems]
    room = family.context(model) - MAX_NEW_TOKENS
    prompts = [ids[-room:] for ids in prompts]
    outputs = [None] * len(problems)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for start in range(0, len(order), BATCH_ROWS):
        group = order[start : start + BATCH_ROWS]
        samples, _ = family.sample(
            model, [prompts[i] for i in group], MAX_NEW_TOKENS, 0.0, None, 0, "graph"
        )
        for row, index in enumerate(group):
            outputs[index] = tokenizer.decode(samples.completions[row])
    return description.stage, family.name, outputs


def _reference_completions(model, revision, problems):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

    from evaluation.benchmarks import livecodebench

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.bfloat16).cuda()
    lm.eval()

    class ClosedBlock(StoppingCriteria):
        def __init__(self, width):
            self.width = width

        def __call__(self, input_ids, scores, **kwargs):
            texts = tokenizer.batch_decode(input_ids[:, self.width :], skip_special_tokens=True)
            return torch.tensor([livecodebench.finished(t) for t in texts], device=input_ids.device)

    texts = [livecodebench.base_prompt(p) for p in problems]
    outputs = [None] * len(problems)
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    for start in range(0, len(order), 16):
        group = order[start : start + 16]
        batch = tokenizer([texts[i] for i in group], return_tensors="pt", padding=True).to("cuda")
        width = batch["input_ids"].shape[1]
        with torch.no_grad():
            generated = lm.generate(
                **batch,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=[ClosedBlock(width)],
            )
        decoded = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
        for row, index in enumerate(group):
            outputs[index] = decoded[row]
    return outputs


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=6 * 60 * 60,
    retries=0,
    secrets=[secret],
    volumes={"/runs": runs_volume, "/training": training_volume},
)
def evaluate_lcb(model, revision, deadline, folder, limit=0):
    """`model` is `checkpoint:<path>` or a reference model's HF name at `revision`."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    from huggingface_hub import hf_hub_download

    from evaluation.benchmarks import livecodebench

    path = hf_hub_download(DATASET, RELEASE_FILE, repo_type="dataset", revision=DATASET_REVISION)
    problems = livecodebench.load(path)[: limit or None]
    started = time.time()
    if model.startswith("checkpoint:"):
        stage, family, completions = _checkpoint_completions(model.split(":", 1)[1], problems)
    else:
        stage, family = "reference", None
        completions = _reference_completions(model, revision, problems)
    generation_seconds = round(time.time() - started, 1)
    scores = livecodebench.grade_all(problems, completions)
    out = {
        "model": model,
        "revision": revision,
        "stage": stage,
        "family": family,
        "dataset": {"repo": DATASET, "revision": DATASET_REVISION, "file": RELEASE_FILE},
        "max_new_tokens": MAX_NEW_TOKENS,
        "limit": limit or None,
        "generation_seconds": generation_seconds,
        "summary": livecodebench.summary(problems, scores),
        "problems": [
            {"question_id": p["question_id"], "score": s, "completion": c}
            for p, s, c in zip(problems, scores, completions, strict=True)
        ],
    }
    print(json.dumps({"model": model, **out["summary"]}))
    name = model.removeprefix("checkpoint:").removesuffix(".pt").replace("/", "--")
    path = Path("/runs/benchmarks", folder, f"lcb-{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    runs_volume.commit()
    return out["summary"]


@app.local_entrypoint()
def main(checkpoints: str = "", references: str = "", limit: int = 0):
    """Spawn one call per checkpoint (comma-separated `--checkpoints`) and reference model
    (comma-separated `--references`, or `all`); calls outlive this client."""
    chosen = [(f"checkpoint:{name}", None) for name in filter(None, checkpoints.split(","))]
    for name in chosen:
        checkpoint_path(name[0].split(":", 1)[1])
    wanted = (
        list(ALL_REFERENCES) if references == "all" else list(filter(None, references.split(",")))
    )
    unknown = [m for m in wanted if m not in ALL_REFERENCES]
    if unknown:
        raise ValueError(f"Unknown reference models: {unknown}")
    chosen += [(m, ALL_REFERENCES[m]) for m in wanted]
    if not chosen:
        raise ValueError("Name --checkpoints and/or --references")
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    folder = f"lcb-{int(time.time())}{'-limit' if limit else ''}"
    calls = [evaluate_lcb.spawn(m, r, deadline, folder, limit) for m, r in chosen]
    supervisor = enforce_deadline.spawn([c.object_id for c in calls], deadline)
    print(
        json.dumps(
            {
                "results": f"lm-exploration-hybrid-runs/benchmarks/{folder}/",
                "calls": {m: c.object_id for (m, _), c in zip(chosen, calls, strict=True)},
                "supervisor": supervisor.object_id,
                "deadline_unix": deadline,
            },
            indent=1,
        )
    )
