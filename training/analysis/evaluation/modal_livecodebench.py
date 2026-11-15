"""LiveCodeBench v6 pass@1 for checkpoints of any model family and for reference models.

Run from training/:
    modal run --detach -m evaluation.modal_livecodebench \
        [--checkpoints runs/RUN/step-N.pt,training/RUN/step-M.pt] [--references M,...|all] \
        [--limit N] [--protocol plain|thinking] [--merge lcb-thinking-<launch>]

A checkpoint path starts with the volume it is on: `runs/` (lm-exploration-hybrid-runs)
or `training/` (lm-exploration-training). Checkpoints load through their family's
adapter (`trainer/families.py`); checkpoints trained to answer instructions
(`checkpoints.Description.instructed`) get its instruction as the family's user turn,
others LiveCodeBench's plain prompt (`base_prompt`).
Reference models (`--references`, Hugging Face base models at pinned revisions) get the
plain prompt through transformers.

One H100 call per model, in parallel. Each call downloads the pinned release file,
generates one greedy completion per problem (up to `MAX_NEW_TOKENS`, or half the
context of a shorter-context model) and grades it inside
the call's container with `livecodebench.grade_all`. Each call has `retries=0`, a 6-hour
timeout and refuses to start 24 hours after launch; an independent supervisor cancels
any call still running at that deadline. Results (summary plus per-problem completions
and scores) go to `lm-exploration-hybrid-runs/benchmarks/<launch>/lcb-<model>.json`.

`--protocol thinking` scores reasoning models (`THINKING_REFERENCES`) as their reports
do: the zero-shot instruction as a chat turn with thinking enabled, `THINKING["samples"]`
samples per problem at temperature 0.6 and top-p 0.95 with up to
`THINKING["max_new_tokens"]` new tokens (fewer when the model's positions run out),
generated with vLLM. A model whose report uses its own system prompt (MobileLLM-R1.5)
gets that prompt and the question. Each sample's answer is the last Python block after
its reasoning span (`livecodebench.final_answer`, `last_block`; a sample whose reasoning
never closes scores 0), and a problem's score is the mean over its samples. Each model
runs as `THINKING_SHARDS` calls, one per slice of the problems, each writing
`lcb-<model>.shard-NN-of-NN.json` and `.completions.jsonl`; once they finish,
`modal run -m evaluation.modal_livecodebench --merge lcb-thinking-<launch>`
writes the combined `lcb-<model>.json`.
"""

import json
import time
from pathlib import Path

import modal
from model_hybrid import modal_benchmarks
from model_hybrid.image import LAUNCH_LIMIT_SECONDS
from model_hybrid.modal_benchmarks import REFERENCES, enforce_deadline, image, runs_volume, secret

from workflows.images import add_project, vllm_base

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
# MobileLLM-R1.5's documented Python-coding system prompt (its model card).
MOBILELLM_CODE_SYSTEM = (
    "\nYou are a helpful and harmless assistant. You should think step-by-step before "
    "responding to the instruction below.\n\nPlease use python programming language only.\n"
    "You must use ```python for just the final solution code block with the following "
    "format:\n```python\n# Your code here\n```\n"
)
# Reasoning models for `--protocol thinking`: revision, positions, and the system prompt
# each model's own report uses (None: the question alone, with thinking enabled).
THINKING_REFERENCES = {
    "Qwen/Qwen3-0.6B": ("c1899de289a04d12100db370d81485cdf75e47ca", 40960, None),
    "Qwen/Qwen3-1.7B": ("70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", 40960, None),
    "facebook/MobileLLM-R1.5-950M": (
        "d4af70ac985faa1d504b3446d8a7448fedf91194",
        32768,
        MOBILELLM_CODE_SYSTEM,
    ),
    "facebook/MobileLLM-R1.5-360M": (
        "b8a298b98305a657651dc2d9f5899fd18b86abd3",
        32768,
        MOBILELLM_CODE_SYSTEM,
    ),
}
THINKING = {"samples": 16, "temperature": 0.6, "top_p": 0.95, "max_new_tokens": 32768, "seed": 1234}
# Calls per model: each H100 call takes every THINKING_SHARDS-th problem.
THINKING_SHARDS = 4
thinking_image = add_project(vllm_base())
hf_cache = modal.Volume.from_name("lm-exploration-hf-cache", create_if_missing=True)


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
    if description.instructed:
        prompts = [family.prompt_ids(tokenizer, livecodebench.instruction(p)) for p in problems]
    else:
        prompts = [family.text_ids(tokenizer, livecodebench.base_prompt(p)) for p in problems]
    # Short-context models get half their context for the answer.
    new_tokens = min(MAX_NEW_TOKENS, family.context(model) // 2)
    room = family.context(model) - new_tokens
    prompts = [ids[-room:] for ids in prompts]
    outputs = [None] * len(problems)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for start in range(0, len(order), BATCH_ROWS):
        group = order[start : start + BATCH_ROWS]
        samples, _ = family.sample(
            model, [prompts[i] for i in group], new_tokens, 0.0, None, 0, "graph"
        )
        for row, index in enumerate(group):
            outputs[index] = tokenizer.decode(samples.completions[row])
    prompt = "instruction" if description.instructed else "base_prompt"
    return (
        {"stage": description.stage, "family": family.name, "prompt": prompt},
        new_tokens,
        outputs,
    )


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
        loaded, new_tokens, completions = _checkpoint_completions(model.split(":", 1)[1], problems)
    else:
        loaded = {"stage": "reference", "family": None, "prompt": "base_prompt"}
        new_tokens = MAX_NEW_TOKENS
        completions = _reference_completions(model, revision, problems)
    generation_seconds = round(time.time() - started, 1)
    scores = livecodebench.grade_all(problems, completions)
    out = {
        "model": model,
        "revision": revision,
        **loaded,
        "dataset": {"repo": DATASET, "revision": DATASET_REVISION, "file": RELEASE_FILE},
        "max_new_tokens": new_tokens,
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


@app.function(
    image=thinking_image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=6 * 60 * 60,
    retries=0,
    secrets=[secret],
    volumes={"/runs": runs_volume, "/hf": hf_cache},
)
def evaluate_thinking(model, revision, deadline, folder, limit=0, shard=0, shards=1):
    """`THINKING["samples"]` reasoning samples per problem from a reference model, for
    every `shards`-th problem from `shard` on."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    from huggingface_hub import hf_hub_download
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.llama4 import Llama4ForCausalLM
    from vllm.model_executor.models.registry import ModelRegistry

    from evaluation.benchmarks import livecodebench

    # MobileLLM-R1.5 is a text-only Llama 4; its card registers the class this way.
    ModelRegistry.register_model("Llama4ForCausalLM", Llama4ForCausalLM)
    _, positions, system = THINKING_REFERENCES[model]
    path = hf_hub_download(DATASET, RELEASE_FILE, repo_type="dataset", revision=DATASET_REVISION)
    problems = livecodebench.load(path)[: limit or None][shard::shards]
    started = time.time()
    llm = LLM(
        model=model,
        revision=revision,
        max_model_len=min(positions, THINKING["max_new_tokens"] + 4096),
        seed=THINKING["seed"],
        enable_prefix_caching=True,
    )
    params = SamplingParams(
        n=THINKING["samples"],
        temperature=THINKING["temperature"],
        top_p=THINKING["top_p"],
        max_tokens=THINKING["max_new_tokens"],
        seed=THINKING["seed"],
    )
    outputs = llm.chat(
        [
            [{"role": "user", "content": livecodebench.instruction(p)}]
            if system is None
            else [
                {"role": "system", "content": system},
                {"role": "user", "content": livecodebench.question(p)},
            ]
            for p in problems
        ],
        params,
        chat_template_kwargs={"enable_thinking": True},
    )
    samples = [list(output.outputs) for output in outputs]
    generation_seconds = round(time.time() - started, 1)
    flat_problems = [p for p, group in zip(problems, samples, strict=True) for _ in group]
    answers = [
        livecodebench.last_block(livecodebench.final_answer(s.text) or "")
        for group in samples
        for s in group
    ]
    flat_scores = iter(livecodebench.grade_all(flat_problems, answers))
    sample_scores = [[next(flat_scores) for _ in group] for group in samples]
    scores = [sum(group) / len(group) for group in sample_scores]
    lengths = [len(s.token_ids) for group in samples for s in group]
    truncated = sum(s.finish_reason == "length" for group in samples for s in group)
    out = {
        "model": model,
        "revision": revision,
        "protocol": "thinking",
        "sampling": THINKING,
        "dataset": {"repo": DATASET, "revision": DATASET_REVISION, "file": RELEASE_FILE},
        "limit": limit or None,
        "generation_seconds": generation_seconds,
        "summary": livecodebench.summary(problems, scores),
        "shard": [shard, shards],
        "samples": _sample_stats(len(lengths), sum(lengths), truncated, answers.count("")),
        "problems": [
            {
                "question_id": p["question_id"],
                "difficulty": p["difficulty"],
                "platform": p["platform"],
                "score": score,
                "sample_scores": group,
            }
            for p, score, group in zip(problems, scores, sample_scores, strict=True)
        ],
    }
    print(json.dumps({"model": model, **out["summary"], **out["samples"]}))
    folder_path = Path("/runs/benchmarks", folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    name = model.replace("/", "--") + (f".shard-{shard:02d}-of-{shards:02d}" if shards > 1 else "")
    (folder_path / f"lcb-{name}.json").write_text(json.dumps(out, indent=1) + "\n")
    with (folder_path / f"lcb-{name}.completions.jsonl").open("w") as handle:
        for problem, group in zip(problems, samples, strict=True):
            row = {
                "question_id": problem["question_id"],
                "completions": [s.text for s in group],
                "finish_reasons": [s.finish_reason for s in group],
            }
            handle.write(json.dumps(row) + "\n")
    runs_volume.commit()
    return out["summary"]


def _sample_stats(count, tokens, truncated, unclosed):
    return {
        "count": count,
        "tokens": tokens,
        "mean_tokens": tokens / count,
        "truncated": truncated / count,
        "unclosed_reasoning": unclosed / count,
    }


@app.function(
    image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes={"/runs": runs_volume}
)
def merge_thinking(folder):
    """Combine each model's shard results in `folder` into `lcb-<model>.json`."""
    from evaluation.benchmarks import livecodebench

    folder_path = Path("/runs/benchmarks", folder)
    groups = {}
    for part in sorted(folder_path.glob("lcb-*.shard-*.json")):
        groups.setdefault(part.name.split(".shard-")[0], []).append(json.loads(part.read_text()))
    merged = {}
    for name, parts in groups.items():
        shards = parts[0]["shard"][1]
        if sorted(p["shard"][0] for p in parts) != list(range(shards)):
            raise ValueError(f"{name}: {len(parts)} of {shards} shards present")
        problems = [row for p in parts for row in p["problems"]]
        stats = [p["samples"] for p in parts]
        out = {
            **{
                k: parts[0][k]
                for k in ("model", "revision", "protocol", "sampling", "dataset", "limit")
            },
            "generation_seconds": [p["generation_seconds"] for p in parts],
            "summary": livecodebench.summary(problems, [row["score"] for row in problems]),
            "samples": _sample_stats(
                sum(s["count"] for s in stats),
                sum(s["tokens"] for s in stats),
                round(sum(s["truncated"] * s["count"] for s in stats)),
                round(sum(s["unclosed_reasoning"] * s["count"] for s in stats)),
            ),
            "problems": problems,
        }
        (folder_path / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")
        merged[name] = {**out["summary"], **out["samples"]}
    runs_volume.commit()
    return merged


def _thinking_launch(checkpoints, references):
    if checkpoints:
        raise ValueError(
            "--protocol thinking scores reference models; checkpoints need a family thinking mode"
        )
    wanted = (
        list(THINKING_REFERENCES)
        if references == "all"
        else list(filter(None, references.split(",")))
    )
    unknown = [m for m in wanted if m not in THINKING_REFERENCES]
    if unknown or not wanted:
        raise ValueError(f"Name --references from {sorted(THINKING_REFERENCES)}: {unknown}")
    return [(m, THINKING_REFERENCES[m][0]) for m in wanted]


@app.local_entrypoint()
def main(
    checkpoints: str = "",
    references: str = "",
    limit: int = 0,
    protocol: str = "plain",
    merge: str = "",
):
    """Spawn one call per checkpoint (comma-separated `--checkpoints`) and reference model
    (comma-separated `--references`, or `all`); calls outlive this client. `--merge
    lcb-thinking-<launch>` instead combines a finished thinking launch's shards."""
    if merge:
        print(json.dumps(merge_thinking.remote(merge), indent=1))
        return
    if protocol not in ("plain", "thinking"):
        raise ValueError("--protocol is plain or thinking")
    if protocol == "thinking":
        chosen = _thinking_launch(checkpoints, references)
    else:
        chosen = [(f"checkpoint:{name}", None) for name in filter(None, checkpoints.split(","))]
        for name in chosen:
            checkpoint_path(name[0].split(":", 1)[1])
        wanted = (
            list(ALL_REFERENCES)
            if references == "all"
            else list(filter(None, references.split(",")))
        )
        unknown = [m for m in wanted if m not in ALL_REFERENCES]
        if unknown:
            raise ValueError(f"Unknown reference models: {unknown}")
        chosen += [(m, ALL_REFERENCES[m]) for m in wanted]
        if not chosen:
            raise ValueError("Name --checkpoints and/or --references")
        function = evaluate_lcb
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    tag = "-thinking" if protocol == "thinking" else ""
    folder = f"lcb{tag}-{int(time.time())}{'-limit' if limit else ''}"
    if protocol == "thinking":
        calls = [
            evaluate_thinking.spawn(m, r, deadline, folder, limit, shard, THINKING_SHARDS)
            for m, r in chosen
            for shard in range(THINKING_SHARDS)
        ]
        chosen = [(f"{m}#{shard}", r) for m, r in chosen for shard in range(THINKING_SHARDS)]
    else:
        calls = [function.spawn(m, r, deadline, folder, limit) for m, r in chosen]
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
