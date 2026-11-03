"""Read-only model investigation: deterministic sampling and instrumented live evaluation."""

import hashlib
import json
import threading
import time
from pathlib import Path


def request_seed(seed, task_id, turn):
    digest = hashlib.sha256(f"{seed}:{task_id}:{turn}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % (2**31)


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_live(config, variant, run, deadline, app, registry):
    from evaluation.agentic_behavior import behavior
    from training_data import open_swe_format as swe
    from training_data import pr_task_agent as agent
    from training_data import qwen_agent_chat as chat
    from vllm import LLM, SamplingParams

    from modal_scripts import modal_agentic as agentic

    source = Path("/runs/agentic-reference", variant["model_run"], "model")
    before = file_sha(source / "model.safetensors")
    if before != variant["sha256"]:
        raise ValueError("Unexpected checkpoint bytes")
    out = Path("/runs/weight-space", config["experiment"], run)
    out.mkdir(parents=True, exist_ok=False)
    (out / "protocol.json").write_text(json.dumps(config, indent=2))
    engine = LLM(
        model=str(source),
        max_model_len=40960,
        hf_overrides={"max_position_embeddings": 40960},
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    tokenizer = engine.get_tokenizer()
    all_tasks = agentic.load_tasks(Path("/data/pr-tasks-pilot"))
    tasks = {f"{t['repository']}@{t['sha']}": t for t in all_tasks}
    task_lookup = {
        agent.first_messages(tasks[key], tasks[key]["issue"])[1]["content"]: key
        for key in config["task_ids"]
    }
    if len(task_lookup) != len(config["task_ids"]):
        raise ValueError("Ambiguous task identity")
    old_app, old_select, old_recorder = agentic.app, agentic._select, agentic._recorder
    agentic.app = app
    lock, sandbox_ids = threading.Lock(), []
    current = {}

    def select(unused_tasks, limit, heldout_only=False):
        selected = [tasks[key] for key in current["task_ids"]]
        if not all(task["heldout"] for task in selected):
            raise ValueError("Investigation must use held-out tasks")
        return selected

    def recorder(folder):
        record = old_recorder(folder)

        def tracked(sandbox_id):
            with lock:
                sandbox_ids.append(sandbox_id)
                registry[run] = list(sandbox_ids)
            record(sandbox_id)

        return tracked

    agentic._select, agentic._recorder = select, recorder

    def policy(conversations):
        if time.time() >= deadline - 300:
            raise TimeoutError("Live investigation deadline")
        replies, requests, params, meta = [None] * len(conversations), [], [], []
        for index, messages in enumerate(conversations):
            key = task_lookup[messages[1]["content"]]
            turn = sum(m["role"] == "assistant" for m in messages)
            prompt = chat.render(swe.conversation(messages), swe.TOOLS, True)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) + config["max_tokens"] > 40960:
                continue
            seed = request_seed(current["seed"], key, turn)
            requests.append({"prompt_token_ids": ids})
            params.append(
                SamplingParams(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    max_tokens=config["max_tokens"],
                    stop_token_ids=[151645, 151643],
                    skip_special_tokens=False,
                    seed=seed,
                )
            )
            meta.append(
                {
                    "index": index,
                    "id": key,
                    "turn": turn + 1,
                    "seed": seed,
                    "prompt_tokens": len(ids),
                    "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                }
            )
        if requests:
            outputs = engine.generate(requests, params, use_tqdm=False)
            with (out / f"raw-{current['label']}.jsonl").open("a") as handle:
                for item, output in zip(meta, outputs, strict=True):
                    completion = output.outputs[0]
                    replies[item["index"]] = swe.parse_assistant(completion.text)
                    handle.write(
                        json.dumps(
                            {
                                **item,
                                "text": completion.text,
                                "tokens": len(completion.token_ids),
                                "finish_reason": completion.finish_reason,
                                "stop_reason": completion.stop_reason,
                            }
                        )
                        + "\n"
                    )
        return replies

    summaries = []
    schedule = config.get("batches")
    if schedule is None:
        schedule = [
            {"seed": seed, "batch": begin // config["batch_tasks"]}
            for seed in config["seeds"]
            for begin in range(0, len(config["task_ids"]), config["batch_tasks"])
        ]
    try:
        for scheduled in schedule:
            seed = scheduled["seed"]
            begin = scheduled["batch"] * config["batch_tasks"]
            if time.time() >= deadline - 900:
                raise TimeoutError("Insufficient time for another task batch")
            label = f"seed{seed}-batch{begin // config['batch_tasks']}"
            current.update(
                seed=seed,
                label=label,
                task_ids=config["task_ids"][begin : begin + config["batch_tasks"]],
            )
            batch_run = f"{run}-{label}"
            print(
                json.dumps({"start_batch": batch_run, "tasks": len(current["task_ids"])}),
                flush=True,
            )
            summary = agentic._evaluate_heldout(
                policy,
                {
                    "model": variant["name"],
                    "seed": seed,
                    "request_seed_scheme": "sha256(seed:task:turn)",
                    "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
                    "max_tokens": config["max_tokens"],
                    "checkpoint_sha256": before,
                },
                "pr-tasks-pilot",
                batch_run,
                len(current["task_ids"]),
                8,
                deadline,
                "cd / && mv /work/repo /testbed && ln -s /testbed /work/repo",
                config["max_turns"],
            )
            rows = [
                json.loads(line)
                for line in Path("/data/pr-tasks-pilot/agentic", batch_run, "trajectories.jsonl")
                .read_text()
                .splitlines()
            ]
            summary["behavior"] = behavior(rows)
            summaries.append(summary)
            (out / "completed.json").write_text(json.dumps(summaries, indent=2))
            agentic.runs_volume.commit()
            print(
                json.dumps({"completed_batch": batch_run, "behavior": summary["behavior"]}),
                flush=True,
            )
        after = file_sha(source / "model.safetensors")
        if after != before:
            raise ValueError("Checkpoint changed during investigation")
        result = {"variant": variant, "summaries": summaries, "weights_unchanged": True}
        (out / "result.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        agentic.app, agentic._select, agentic._recorder = old_app, old_select, old_recorder
        agentic.runs_volume.commit()
