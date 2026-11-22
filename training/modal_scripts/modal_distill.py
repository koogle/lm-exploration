"""Student answers and a teacher's scores of them, for on-policy distillation
(`training_data.distillation`; trained by the RL loop's `off_policy/distill`).

Run from training/, in this order; `<data-id>` is a fresh folder on the training volume:
    modal run -m modal_scripts.modal_distill::prompts --data-id <data-id> \
        [--recipe data/recipes/chat-sft-reasoning-v4.json] [--limit 0] [--samples 3]
    modal run --detach -m modal_scripts.modal_distill::sample --data-id <data-id> \
        --checkpoint runs/<run>/step-N.pt [--shards 8] [--limit 0]
    modal run -m modal_scripts.modal_distill::merge --data-id <data-id>
    modal run --detach -m modal_scripts.modal_distill::score --data-id <data-id> [--shards 8]
    modal run -m modal_scripts.modal_distill::finish --data-id <data-id> \
        --checkpoint runs/<run>/step-N.pt

- `prompts` (one CPU call): the distinct thinking prompts of a prepared chat SFT recipe
  (`distillation.chat_prompts`; a seeded sample of `--limit` of them, if set), each
  `--samples` times in a seeded order, as `<data-id>/prompts.npz`.
- `sample` (`--shards` H100 calls): the checkpoint, through its family's sampler, writes
  one answer to every `--shards`-th prompt, `SAMPLING["rows"]` at a time, saved to
  `<data-id>/shards/answers-NN-of-NN.npz` every `SAVE_BATCHES` batches. Relaunching the
  same command continues each shard after its saved batches.
- `merge` (one CPU call): the shards as `tokens.npy` and `answers.npy`.
- `score` (`--shards` H100 calls): the teacher (`TEACHER`, same tokenizer and chat format
  as the student) reads each answer once and keeps its `TOP_K` likeliest next tokens and
  their log-probabilities at every answer token, as
  `<data-id>/shards/scores-NN-of-NN.npz` (saved as it goes; a relaunch continues).
- `finish` (one CPU call): `teacher_ids.npy`, `teacher_logprobs.npy` and `report.json`.

Every call has `retries=0` and refuses to start after its launch's deadline, 24 hours
after launch; `sample` and `score` spawn the shared supervisor
(`modal_benchmarks.enforce_deadline`), which cancels calls still running then.
"""

import json
import time
from pathlib import Path

import modal
from model_hybrid import modal_benchmarks
from model_hybrid.image import LAUNCH_LIMIT_SECONDS
from model_hybrid.modal_benchmarks import enforce_deadline, image, runs_volume, secret

app = modal.App("lm-exploration-distill")
app.include(modal_benchmarks.app)  # the deadline supervisor
training_volume = modal.Volume.from_name("lm-exploration-training")
data_volume = modal.Volume.from_name("lm-exploration-hybrid-data")
hf_cache = modal.Volume.from_name("lm-exploration-hf-cache", create_if_missing=True)
# Checkpoint paths start with one of these mounts (as in `evaluation.modal_livecodebench`).
MOUNTS = {"/runs": runs_volume, "/training": training_volume}
# The LiveCodeBench thinking protocol's sampling, so answers are the ones the benchmark
# sees; prompts and answers within the contest RL stage's limits.
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "answer_tokens": 8192,
    "prompt_tokens": 2048,
    "rows": 64,
    "seed": 1234,
}
SAVE_BATCHES = 4
TEACHER = ("zai-org/GLM-4.7-Flash", "7dd20894a642a0aa287e9827cb1a1f7f91386b67")
TOP_K = 20
# Padded tokens per teacher forward pass, and answers between saves.
SCORE_TOKENS = 12_000
SCORE_SAVE_ANSWERS = 500
SHARD_SECONDS = 12 * 60 * 60


def folder_of(data_id):
    from workflows.checks import run_id

    return Path("/training") / run_id(data_id, "data-id")


def refuse_after(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=3600,
    retries=0,
    volumes={"/training": training_volume, "/data": data_volume},
)
def collect_prompts(data_id, recipe, limit, samples, deadline):
    refuse_after(deadline)
    import random

    import numpy as np
    from training_data import distillation

    folder = folder_of(data_id)
    if folder.exists():
        raise FileExistsError(f"{folder} exists; name a fresh data ID")
    prompts = distillation.chat_prompts(
        recipe, Path("/data") / recipe["name"], SAMPLING["prompt_tokens"]
    )
    found = len(prompts)
    order = random.Random(SAMPLING["seed"])
    order.shuffle(prompts)
    if limit:
        prompts = prompts[:limit]
    # Each prompt `samples` times, apart from each other, so every shard answers a mix.
    prompts = [prompt for prompt in prompts for _ in range(samples)]
    order.shuffle(prompts)
    folder.mkdir(parents=True)
    np.savez(
        folder / "prompts.npz",
        tokens=np.concatenate([np.asarray(p, dtype=np.uint32) for p in prompts]),
        lengths=np.asarray([len(p) for p in prompts], dtype=np.int64),
    )
    summary = {
        "recipe": recipe["name"],
        "distinct_prompts": found,
        "samples_per_prompt": samples,
        "prompts": len(prompts),
    }
    (folder / "prompts.json").write_text(json.dumps(summary, indent=2))
    training_volume.commit()
    return summary


def load_prompts(folder):
    import numpy as np

    saved = np.load(folder / "prompts.npz")
    ends = np.cumsum(saved["lengths"])
    return [
        saved["tokens"][end - length : end].tolist()
        for end, length in zip(ends, saved["lengths"], strict=True)
    ]


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=SHARD_SECONDS,
    retries=0,
    volumes={**MOUNTS},
)
def sample_shard(data_id, checkpoint, shard, shards, limit, deadline):
    """One answer to every `shards`-th prompt from `shard` on (the first `limit`, if
    set), continuing after the batches this shard already saved."""
    refuse_after(deadline)
    import numpy as np
    import torch
    from training_data import distillation

    from trainer import checkpoints

    folder = folder_of(data_id)
    prompts = load_prompts(folder)[shard::shards]
    if limit:
        prompts = prompts[:limit]
    output = folder / "shards" / f"answers-{shard:02d}-of-{shards:02d}.npz"
    output.parent.mkdir(exist_ok=True)
    parts, done = [], 0
    if output.exists():
        saved = np.load(output)
        parts, done = [(saved["tokens"], saved["answers"])], int(saved["batches"])
    path = Path("/") / checkpoint
    if Path("/") / path.parts[1] not in map(Path, MOUNTS) or ".." in path.parts:
        raise ValueError(f"A checkpoint path starts with runs/ or training/: {checkpoint}")
    saved, description, family = checkpoints.parent(path)
    model = family.build(description.model_config).cuda()
    family.load(model, saved["model"])
    del saved
    model.eval()
    rows = SAMPLING["rows"]
    batches = range(done, -(-len(prompts) // rows))
    started = time.time()

    def save(batch):
        tokens, answers = distillation.merge_answers(parts)
        temporary = output.with_suffix(".partial.npz")
        np.savez(temporary, tokens=tokens, answers=answers, batches=batch)
        temporary.replace(output)
        training_volume.commit()
        return tokens, answers

    for batch in batches:
        # Stop with room to save before the launch deadline.
        if time.time() >= deadline - 1800:
            break
        chosen = prompts[batch * rows : (batch + 1) * rows]
        generator = torch.Generator(device="cuda").manual_seed(
            SAMPLING["seed"] + 100_003 * shard + batch
        )
        result, stops = family.sample(
            model,
            chosen,
            SAMPLING["answer_tokens"],
            SAMPLING["temperature"],
            generator,
            0,
            "graph",
            top_p=SAMPLING["top_p"],
        )
        parts.append(distillation.pack(chosen, result.completions, stops))
        done = batch + 1
        if done % SAVE_BATCHES == 0:
            save(done)
            print(json.dumps({"shard": shard, "batches": done, "of": len(batches)}), flush=True)
    _, answers = save(done)
    return {
        "shard": shard,
        "batches": done,
        "complete": done == -(-len(prompts) // rows),
        "answers": len(answers),
        "answer_tokens": int(answers[:, 2].sum()),
        "finished": float(answers[:, 3].mean()) if len(answers) else 0.0,
        "seconds": time.time() - started,
    }


@app.function(image=image, cpu=8, memory=131072, timeout=3600, retries=0, volumes={**MOUNTS})
def merge_shards(data_id, deadline):
    refuse_after(deadline)
    import numpy as np
    from training_data import distillation

    folder = folder_of(data_id)
    paths = sorted((folder / "shards").glob("answers-*-of-*.npz"))
    paths = [p for p in paths if ".partial" not in p.name]
    if not paths or len(paths) != int(paths[0].stem.rsplit("-", 1)[1]):
        raise ValueError(f"Missing answer shards: found {[p.name for p in paths]}")
    loaded = [np.load(p) for p in paths]
    tokens, answers = distillation.merge_answers([(s["tokens"], s["answers"]) for s in loaded])
    np.save(folder / "tokens.npy", tokens)
    np.save(folder / "answers.npy", answers)
    summary = {
        "answers": len(answers),
        "answer_tokens": int(answers[:, 2].sum()),
        "finished": float(answers[:, 3].mean()),
        "mean_answer_tokens": float(answers[:, 2].mean()),
        "shard_batches": [int(s["batches"]) for s in loaded],
    }
    (folder / "answers.json").write_text(json.dumps(summary, indent=2))
    training_volume.commit()
    return summary


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=4 * 3600,
    retries=0,
    secrets=[secret],
    volumes={"/hf": hf_cache},
)
def cache_teacher(deadline):
    """The teacher's weights in the shared Hugging Face cache, before the shards load it."""
    refuse_after(deadline)
    from huggingface_hub import snapshot_download

    snapshot_download(TEACHER[0], revision=TEACHER[1], cache_dir="/hf/hub")
    hf_cache.commit()


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=131072,
    timeout=SHARD_SECONDS,
    retries=0,
    secrets=[secret],
    volumes={"/training": training_volume, "/hf": hf_cache},
)
def score_shard(data_id, shard, shards, deadline):
    """The teacher's `TOP_K` next tokens at every answer token of the shard's answers (a
    consecutive range of them), continuing after the answers this shard already saved."""
    refuse_after(deadline)
    import numpy as np
    import torch
    from training_data import distillation
    from transformers import AutoModelForCausalLM

    from tokenizer import glm

    folder = folder_of(data_id)
    tokens = np.load(folder / "tokens.npy", mmap_mode="r")
    answers = np.load(folder / "answers.npy")
    first, last = (len(answers) * s // shards for s in (shard, shard + 1))
    offsets, _ = distillation.score_offsets(answers[first:last])
    rows = int(answers[first:last, 2].sum())
    output = folder / "shards" / f"scores-{shard:02d}-of-{shards:02d}.npz"
    ids = np.zeros((rows, TOP_K), dtype=np.int32)
    logprobs = np.zeros((rows, TOP_K), dtype=np.float16)
    scored = np.zeros(last - first, dtype=bool)
    sums = {"nll": 0.0, "mass": 0.0, "top1": 0.0}
    if output.exists():
        saved = np.load(output)
        ids, logprobs, scored = saved["ids"], saved["logprobs"], saved["scored"]
        sums = json.loads(str(saved["sums"]))
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER[0],
        revision=TEACHER[1],
        dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir="/hf/hub",
    )
    model.eval()
    if model.config.vocab_size != glm.VOCAB_SIZE:
        raise ValueError("The teacher's vocabulary differs from the student's")

    def save():
        temporary = output.with_suffix(".partial.npz")
        np.savez(
            temporary,
            ids=ids,
            logprobs=logprobs,
            scored=scored,
            sums=json.dumps(sums),
            complete=bool(scored.all()),
        )
        temporary.replace(output)
        training_volume.commit()

    # Longest first, so the largest batch runs before hours of work depend on it.
    pending = sorted(np.flatnonzero(~scored), key=lambda i: -int(answers[first + i, 1:3].sum()))
    started, since_save = time.time(), 0
    while pending and time.time() < deadline - 1800:
        width = int(answers[first + pending[0], 1:3].sum()) - 1
        group = pending[: max(1, SCORE_TOKENS // width)]
        del pending[: len(group)]
        x = torch.full((len(group), width), glm.EOS_ID, dtype=torch.long)
        attention = torch.zeros((len(group), width), dtype=torch.long)
        for b, i in enumerate(group):
            start, prompt, answer, _ = (int(v) for v in answers[first + i])
            length = prompt + answer - 1
            x[b, :length] = torch.from_numpy(tokens[start : start + length].astype(np.int64))
            attention[b, :length] = 1
        with torch.no_grad():
            logits = model(input_ids=x.cuda(), attention_mask=attention.cuda()).logits
        for b, i in enumerate(group):
            start, prompt, answer, _ = (int(v) for v in answers[first + i])
            # Position p predicts token p + 1: the answer's tokens are predicted at
            # prompt - 1 .. prompt + answer - 2.
            scores = logits[b, prompt - 1 : prompt + answer - 1].float().log_softmax(-1)
            target = torch.from_numpy(
                tokens[start + prompt : start + prompt + answer].astype(np.int64)
            ).cuda()
            top = scores.topk(TOP_K, dim=-1)
            row = int(offsets[i])
            ids[row : row + answer] = top.indices.cpu().numpy()
            logprobs[row : row + answer] = top.values.cpu().numpy()
            sums["nll"] += -scores.gather(-1, target[:, None]).sum().item()
            sums["mass"] += top.values.exp().sum().item()
            sums["top1"] += (top.indices[:, 0] == target).sum().item()
            scored[i] = True
        del logits
        since_save += len(group)
        if since_save >= SCORE_SAVE_ANSWERS:
            save()
            since_save = 0
            print(
                json.dumps({"shard": shard, "scored": int(scored.sum()), "of": len(scored)}),
                flush=True,
            )
    save()
    done = max(1, int(answers[first:last, 2][scored].sum()))
    return {
        "shard": shard,
        "answers": int(scored.sum()),
        "complete": bool(scored.all()),
        "teacher_nll_of_student_tokens": sums["nll"] / done,
        "listed_mass": sums["mass"] / done,
        "teacher_top1_is_student_token": sums["top1"] / done,
        "seconds": time.time() - started,
    }


@app.function(
    image=image,
    cpu=8,
    memory=131072,
    timeout=2 * 3600,
    retries=0,
    volumes={"/training": training_volume},
)
def finish_scores(data_id, details, deadline):
    refuse_after(deadline)
    import numpy as np
    from training_data import distillation

    folder = folder_of(data_id)
    paths = sorted((folder / "shards").glob("scores-*-of-*.npz"))
    paths = [p for p in paths if ".partial" not in p.name]
    if not paths or len(paths) != int(paths[0].stem.rsplit("-", 1)[1]):
        raise ValueError(f"Missing score shards: found {[p.name for p in paths]}")
    loaded = [np.load(p) for p in paths]
    if not all(bool(s["complete"]) for s in loaded):
        raise ValueError("A score shard is unfinished; relaunch `score`")
    ids, logprobs = distillation.merge_scores([(s["ids"], s["logprobs"]) for s in loaded])
    np.save(folder / "teacher_ids.npy", ids)
    np.save(folder / "teacher_logprobs.npy", logprobs)
    sums = [json.loads(str(s["sums"])) for s in loaded]
    rows = len(ids)
    details = {
        **details,
        **json.loads((folder / "prompts.json").read_text()),
        "teacher": {"repo": TEACHER[0], "revision": TEACHER[1]},
        "teacher_nll_of_student_tokens": sum(s["nll"] for s in sums) / rows,
        "listed_mass": sum(s["mass"] for s in sums) / rows,
        "teacher_top1_is_student_token": sum(s["top1"] for s in sums) / rows,
    }
    report = distillation.write_report(folder, details)
    training_volume.commit()
    return report


def launch(calls):
    """The supervisor for spawned `calls`, and what to print."""
    deadline = calls["deadline"]
    supervisor = enforce_deadline.spawn([c.object_id for c in calls["calls"]], deadline)
    print(
        json.dumps(
            {
                "calls": [c.object_id for c in calls["calls"]],
                "supervisor": supervisor.object_id,
                "deadline_unix": deadline,
            },
            indent=1,
        )
    )


@app.local_entrypoint()
def prompts(
    data_id: str,
    recipe: str = "data/recipes/chat-sft-reasoning-v4.json",
    limit: int = 0,
    samples: int = 3,
):
    spec = json.loads(Path(recipe).read_text())
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    print(json.dumps(collect_prompts.remote(data_id, spec, limit, samples, deadline), indent=1))


@app.local_entrypoint()
def sample(data_id: str, checkpoint: str, shards: int = 8, limit: int = 0):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    calls = [
        sample_shard.spawn(data_id, checkpoint, shard, shards, limit, deadline)
        for shard in range(shards)
    ]
    launch({"calls": calls, "deadline": deadline})


@app.local_entrypoint()
def merge(data_id: str):
    print(json.dumps(merge_shards.remote(data_id, time.time() + LAUNCH_LIMIT_SECONDS), indent=1))


@app.local_entrypoint()
def score(data_id: str, shards: int = 8):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    cache_teacher.remote(deadline)
    calls = [score_shard.spawn(data_id, shard, shards, deadline) for shard in range(shards)]
    launch({"calls": calls, "deadline": deadline})


@app.local_entrypoint()
def finish(data_id: str, checkpoint: str):
    details = {"student": checkpoint, "sampling": SAMPLING}
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    print(json.dumps(finish_scores.remote(data_id, details, deadline), indent=1))
