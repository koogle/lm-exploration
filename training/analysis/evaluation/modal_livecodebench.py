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
writes the combined `lcb-<model>.json`. Checkpoints (`--checkpoints`) get the
instruction as their family's user turn with the reasoning span opened
(`prompt_ids(..., thinking=True)`) and sample with the family's own sampler, as
`CHECKPOINT_THINKING_SHARDS` calls each, with up to what the context leaves after the
prompt when that is under `THINKING["max_new_tokens"]`.
"""

import json
import time
from pathlib import Path

import modal
from model_hybrid import modal_benchmarks
from model_hybrid.image import LAUNCH_LIMIT_SECONDS
from model_hybrid.modal_benchmarks import REFERENCES, enforce_deadline, image, runs_volume, secret

from evaluation.benchmarks import livecodebench
from workflows.images import add_project, vllm_base

app = modal.App("lm-exploration-livecodebench")
app.include(modal_benchmarks.app)  # the deadline supervisor
training_volume = modal.Volume.from_name("lm-exploration-training")
# Checkpoint paths start with one of these mounts.
MOUNTS = {"runs": runs_volume, "training": training_volume}
DATASET, DATASET_REVISION = livecodebench.DATASET, livecodebench.DATASET_REVISION
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
    "Qwen/Qwen3-32B-FP8": ("aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df", 40960, None),
    "Qwen/Qwen3-30B-A3B-FP8": ("d206ba732169f29bb77fbf80fc2c4b81d4d30782", 40960, None),
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
# `--pool train`: teacher traces for SFT on `contest_sft.trace_problems`, shorter and
# fewer per problem; each passing problem keeps its shortest passing completion. The
# KV cache is FP8 there, so more sequences decode together.
TRACES = {**THINKING, "samples": 4, "max_new_tokens": 16384}
TRACE_SHARDS = 16
# Calls per model: each H100 call takes every THINKING_SHARDS-th problem.
THINKING_SHARDS = 4
# Checkpoints decode with the family's own sampler: more calls, and problems per batch
# (each batch decodes all of its problems' samples together until the longest ends).
CHECKPOINT_THINKING_SHARDS = 8
CHECKPOINT_THINKING_PROBLEMS = 4
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
def evaluate_thinking(model, revision, deadline, folder, limit=0, shard=0, shards=1, pool=""):
    """`THINKING["samples"]` reasoning samples per problem from a reference model, for
    every `shards`-th problem from `shard` on; with `pool` "train", `TRACES` samples of
    the trace problems instead, and the passing traces are written too."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.llama4 import Llama4ForCausalLM
    from vllm.model_executor.models.registry import ModelRegistry

    from evaluation.benchmarks import livecodebench

    # MobileLLM-R1.5 is a text-only Llama 4; its card registers the class this way.
    ModelRegistry.register_model("Llama4ForCausalLM", Llama4ForCausalLM)
    _, positions, system = THINKING_REFERENCES[model]
    problems = (_trace_problems if pool else _thinking_problems)(limit, shard, shards)
    sampling = TRACES if pool else THINKING
    started = time.time()
    llm = LLM(
        model=model,
        revision=revision,
        max_model_len=min(positions, sampling["max_new_tokens"] + 4096),
        seed=sampling["seed"],
        enable_prefix_caching=True,
        **({"kv_cache_dtype": "fp8"} if pool else {}),
    )
    params = SamplingParams(
        n=sampling["samples"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        max_tokens=sampling["max_new_tokens"],
        seed=sampling["seed"],
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
    samples = [
        [(s.text, len(s.token_ids), s.finish_reason) for s in output.outputs] for output in outputs
    ]
    return _write_thinking(
        model,
        revision,
        {},
        folder,
        limit,
        shard,
        shards,
        problems,
        samples,
        started,
        traces=bool(pool),
    )


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
def evaluate_thinking_checkpoint(name, deadline, folder, limit=0, shard=0, shards=1):
    """`THINKING["samples"]` reasoning samples per problem from a checkpoint whose family
    has a thinking prompt (`prompt_ids(..., thinking=True)`), for every `shards`-th
    problem from `shard` on; up to `THINKING["max_new_tokens"]` new tokens, or what the
    model's context leaves after the prompt."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import torch

    from evaluation.benchmarks import livecodebench
    from trainer import checkpoints

    problems = _thinking_problems(limit, shard, shards)
    started = time.time()
    saved, description, family = checkpoints.parent(checkpoint_path(name))
    model = family.build(description.model_config).cuda()
    family.load(model, saved["model"])
    del saved
    model.eval()
    tokenizer = family.tokenizer()
    prompts = [
        family.prompt_ids(tokenizer, livecodebench.instruction(p), thinking=True) for p in problems
    ]
    generator = torch.Generator(device="cuda").manual_seed(THINKING["seed"])
    count = THINKING["samples"]
    samples = []
    for start in range(0, len(prompts), CHECKPOINT_THINKING_PROBLEMS):
        rows = [
            ids
            for ids in prompts[start : start + CHECKPOINT_THINKING_PROBLEMS]
            for _ in range(count)
        ]
        limit_tokens = min(THINKING["max_new_tokens"], family.context(model) - max(map(len, rows)))
        result, stops = family.sample(
            model,
            rows,
            limit_tokens,
            THINKING["temperature"],
            generator,
            0,
            "graph",
            top_p=THINKING["top_p"],
        )
        group = [
            (
                tokenizer.decode(ids),
                len(ids) + (stop is not None),
                "stop" if stop is not None else "length",
            )
            for ids, stop in zip(result.completions, stops, strict=True)
        ]
        samples += [group[i : i + count] for i in range(0, len(group), count)]
    loaded = {"stage": description.stage, "family": family.name, "prompt": "instruction"}
    return _write_thinking(
        f"checkpoint:{name}",
        None,
        loaded,
        folder,
        limit,
        shard,
        shards,
        problems,
        samples,
        started,
        opened=True,
    )


def _thinking_problems(limit, shard, shards):
    from huggingface_hub import hf_hub_download

    from evaluation.benchmarks import livecodebench

    path = hf_hub_download(DATASET, RELEASE_FILE, repo_type="dataset", revision=DATASET_REVISION)
    return livecodebench.load(path)[: limit or None][shard::shards]


def _trace_problems(limit, shard, shards):
    from huggingface_hub import hf_hub_download
    from training_data import contest_sft

    releases = [
        hf_hub_download(DATASET, name, repo_type="dataset", revision=DATASET_REVISION)
        for name in contest_sft.TRACE_RELEASES
    ]
    leetcode = hf_hub_download(
        contest_sft.LEETCODE[0],
        "LeetCodeDataset-train.jsonl",
        repo_type="dataset",
        revision=contest_sft.LEETCODE[1],
    )
    # JSON lines end at "\n" only; statements can hold other line separators.
    rows = [json.loads(line) for line in Path(leetcode).read_text().split("\n") if line]
    return contest_sft.trace_problems(releases, rows)[: limit or None][shard::shards]


def _write_thinking(
    model,
    revision,
    loaded,
    folder,
    limit,
    shard,
    shards,
    problems,
    samples,
    started,
    opened=False,
    traces=False,
):
    """Grade (text, tokens, finish reason) samples per problem and write the shard's result
    and completions; with `opened`, the prompt opened the reasoning span, so a completion
    that never closes it scores 0. With `traces`, the samples are teacher traces of the
    trace problems, and each problem's shortest passing completion is written to
    `<name>.traces.jsonl` (the rows `contest_sft.teacher_traces` reads)."""
    from evaluation.benchmarks import livecodebench

    generation_seconds = round(time.time() - started, 1)
    flat_problems = [p for p, group in zip(problems, samples, strict=True) for _ in group]
    answers = [
        livecodebench.last_block(
            livecodebench.final_answer(("<think>" if opened else "") + text) or ""
        )
        for group in samples
        for text, _, _ in group
    ]
    flat_scores = iter(livecodebench.grade_all(flat_problems, answers))
    sample_scores = [[next(flat_scores) for _ in group] for group in samples]
    scores = [sum(group) / len(group) for group in sample_scores]
    lengths = [tokens for group in samples for _, tokens, _ in group]
    truncated = sum(finish == "length" for group in samples for _, _, finish in group)
    out = {
        "model": model,
        "revision": revision,
        **loaded,
        "protocol": "traces" if traces else "thinking",
        "sampling": TRACES if traces else THINKING,
        "dataset": (
            {"trace_problems": "contest_sft.trace_problems", "revision": DATASET_REVISION}
            if traces
            else {"repo": DATASET, "revision": DATASET_REVISION, "file": RELEASE_FILE}
        ),
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
    name = model.removeprefix("checkpoint:").removesuffix(".pt").replace("/", "--")
    name += f".shard-{shard:02d}-of-{shards:02d}" if shards > 1 else ""
    (folder_path / f"lcb-{name}.json").write_text(json.dumps(out, indent=1) + "\n")
    with (folder_path / f"lcb-{name}.completions.jsonl").open("w") as handle:
        for problem, group in zip(problems, samples, strict=True):
            row = {
                "question_id": problem["question_id"],
                "completions": [text for text, _, _ in group],
                "finish_reasons": [finish for _, _, finish in group],
            }
            handle.write(json.dumps(row) + "\n")
    if traces:
        with (folder_path / f"lcb-{name}.traces.jsonl").open("w") as handle:
            for problem, group, scores_ in zip(problems, samples, sample_scores, strict=True):
                passing = [(tokens, text) for (text, tokens, _), s in zip(group, scores_) if s]
                if passing:
                    row = {
                        key: problem[key]
                        for key in ("question_id", "platform", "difficulty", "contest_date")
                    }
                    row |= {
                        "question_content": problem["question_content"],
                        "starter_code": problem["starter_code"],
                        "completion": min(passing)[1],
                        "passing": len(passing),
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
                for k in (
                    "model",
                    "revision",
                    "stage",
                    "family",
                    "prompt",
                    "protocol",
                    "sampling",
                    "dataset",
                    "limit",
                )
                if k in parts[0]
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
    """(model or checkpoint:<path>, revision) to score with thinking."""
    chosen = [(f"checkpoint:{name}", None) for name in filter(None, checkpoints.split(","))]
    for name, _ in chosen:
        checkpoint_path(name.split(":", 1)[1])
    wanted = (
        list(THINKING_REFERENCES)
        if references == "all"
        else list(filter(None, references.split(",")))
    )
    unknown = [m for m in wanted if m not in THINKING_REFERENCES]
    if unknown or not (wanted or chosen):
        raise ValueError(
            f"Name --checkpoints or --references from {sorted(THINKING_REFERENCES)}: {unknown}"
        )
    return chosen + [(m, THINKING_REFERENCES[m][0]) for m in wanted]


@app.local_entrypoint()
def main(
    checkpoints: str = "",
    references: str = "",
    limit: int = 0,
    protocol: str = "plain",
    merge: str = "",
    pool: str = "",
):
    """Spawn one call per checkpoint (comma-separated `--checkpoints`) and reference model
    (comma-separated `--references`, or `all`); calls outlive this client. `--merge
    lcb-thinking-<launch>` instead combines a finished thinking launch's shards.
    `--protocol thinking --pool train --references <teacher>` samples teacher traces
    on the trace problems (`TRACES`, `TRACE_SHARDS` calls)."""
    if merge:
        print(json.dumps(merge_thinking.remote(merge), indent=1))
        return
    if protocol not in ("plain", "thinking"):
        raise ValueError("--protocol is plain or thinking")
    if pool not in ("", "train") or pool and (protocol != "thinking" or checkpoints):
        raise ValueError("--pool train samples reference models with --protocol thinking")
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
    tag = "-traces" if pool else "-thinking" if protocol == "thinking" else ""
    folder = f"lcb{tag}-{int(time.time())}{'-limit' if limit else ''}"
    if protocol == "thinking":
        calls, names = [], []
        for m, r in chosen:
            checkpoint = m.startswith("checkpoint:")
            shards = (
                CHECKPOINT_THINKING_SHARDS
                if checkpoint
                else TRACE_SHARDS
                if pool
                else THINKING_SHARDS
            )
            for shard in range(shards):
                if checkpoint:
                    call = evaluate_thinking_checkpoint.spawn(
                        m.split(":", 1)[1], deadline, folder, limit, shard, shards
                    )
                else:
                    call = evaluate_thinking.spawn(
                        m, r, deadline, folder, limit, shard, shards, pool
                    )
                calls.append(call)
                names.append((f"{m}#{shard}", r))
        chosen = names
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
