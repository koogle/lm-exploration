"""LiveCodeBench v6 pass@1 for hybrid exports and the reference models.

Run from training/:
    modal run --detach -m model_hybrid.modal_livecodebench [--models M,...] [--limit N]

One H100 call per model, in parallel. Each call downloads the pinned release file,
generates one greedy completion per problem (up to `MAX_NEW_TOKENS`, stopping once
a code block closes) and grades it inside the call's container with
`livecodebench.grade_all`. Base models get LiveCodeBench's plain prompt; chat
exports get the same question as a system and user turn in the GLM chat format.
Each call has `retries=0`, a 6-hour timeout and refuses to start 24 hours after
launch; an independent supervisor cancels any call still running at that deadline.
Results (summary plus per-problem completions and scores) go to
`lm-exploration-hybrid-runs/benchmarks/<launch>/lcb-<model>.json`.
"""

import json
import time
from pathlib import Path

import modal
from model_hybrid import modal_benchmarks
from model_hybrid.image import LAUNCH_LIMIT_SECONDS
from model_hybrid.modal_benchmarks import REFERENCES, enforce_deadline, image, runs_volume, secret

app = modal.App("lm-exploration-hybrid-livecodebench")
app.include(modal_benchmarks.app)  # the deadline supervisor
DATASET = "livecodebench/code_generation_lite"
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
RELEASE_FILE = "test6.jsonl"
MAX_NEW_TOKENS = 2048
# Hybrid exports as "<run>/<export>", with whether they take the chat format.
EXPORTS = {
    "hybrid-720m-40b-01/tokens-039998980096": False,
    "hybrid-720m-chat-sft-01/tokens-001527250944": True,
    "hybrid-720m-rl-lr-low-01/tokens-000158595563": True,
    "hybrid-720m-mid32k-01/tokens-002999975936": False,
}
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


def _hybrid_completions(export, chat_format, problems):
    import torch
    from model_hybrid import chat, generate, livecodebench, sft_data
    from model_hybrid import tokenizer as glm

    run, name = export.split("/")
    lm = generate.load_export(
        Path("/runs", run, "export", name), dtype=torch.bfloat16, device="cuda"
    )
    tokenizer = glm.load()
    if chat_format:
        texts = [
            chat.render(livecodebench.chat_messages(p), add_generation_prompt=True)
            for p in problems
        ]
        stop_ids = sft_data.STOP_IDS
    else:
        texts = [livecodebench.base_prompt(p) for p in problems]
        stop_ids = (lm.config.eos_id,)
    room = lm.config.block_size - MAX_NEW_TOKENS
    prompts = [tokenizer.encode(t, add_special_tokens=False).ids[-room:] for t in texts]
    outputs = [None] * len(problems)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for start in range(0, len(order), 32):
        group = order[start : start + 32]

        def done(row, ids):
            return livecodebench.finished(tokenizer.decode(ids))

        generated = generate.generate_batch(
            lm, [prompts[i] for i in group], MAX_NEW_TOKENS, stop_ids, done=done
        )
        for row, index in enumerate(group):
            outputs[index] = tokenizer.decode([t for t in generated[row] if t not in stop_ids])
    return outputs


def _reference_completions(model, revision, problems):
    import torch
    from model_hybrid import livecodebench
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

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
    volumes={"/runs": runs_volume},
)
def evaluate_lcb(model, revision, deadline, folder, limit=0):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    from huggingface_hub import hf_hub_download
    from model_hybrid import livecodebench

    path = hf_hub_download(DATASET, RELEASE_FILE, repo_type="dataset", revision=DATASET_REVISION)
    problems = livecodebench.load(path)[: limit or None]
    started = time.time()
    if model.startswith("hybrid:"):
        export = model.split(":", 1)[1]
        completions = _hybrid_completions(export, EXPORTS[export], problems)
    else:
        completions = _reference_completions(model, revision, problems)
    generation_seconds = round(time.time() - started, 1)
    scores = livecodebench.grade_all(problems, completions)
    out = {
        "model": model,
        "revision": revision,
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
    name = model.removeprefix("hybrid:").replace("/", "--")
    path = Path("/runs/benchmarks", folder, f"lcb-{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    runs_volume.commit()
    return out["summary"]


@app.local_entrypoint()
def main(limit: int = 0, models: str = ""):
    """Spawn one call per model (all exports and references, or the comma-separated
    `--models`, "hybrid:<run>/<export>" for exports); calls outlive this client."""
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    chosen = (
        [(f"hybrid:{e}", None) for e in EXPORTS]
        + list(REFERENCES.items())
        + list(LARGER_QWEN.items())
    )
    if models:
        wanted = set(models.split(","))
        chosen = [(m, r) for m, r in chosen if m in wanted]
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
