"""Finite Qwen3.5 trajectory generation; native tokenizer and sparse targets."""

import ast
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

from core import atomic_json
from curriculum import specification
from megakernel.model import MODEL, REVISION


def accepted(text, finish):
    if finish != "stop":
        return False
    try:
        value = json.loads(text)
        if not all(
            isinstance(value.get(k), str) and value[k].strip()
            for k in ["problem", "code", "explanation"]
        ):
            return False
        tree = ast.parse(value["code"])
        return any(
            isinstance(n, ast.FunctionDef) and n.name == "solve" for n in tree.body
        )
    except (ValueError, SyntaxError, TypeError, AttributeError):
        return False


def sparse(logprobs, ids):
    result = {
        k: [] for k in ["topk_ids", "topk_logprobs", "selected_logprobs", "tail_mass"]
    }
    for row, token in zip(logprobs, ids, strict=True):
        items = sorted(row.items(), key=lambda pair: pair[1].logprob, reverse=True)[:64]
        values = [float(v.logprob) for _, v in items]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite target")
        result["topk_ids"].append([int(i) for i, _ in items])
        result["topk_logprobs"].append(values)
        result["selected_logprobs"].append(float(row[token].logprob))
        result["tail_mass"].append(max(0.0, 1 - sum(math.exp(x) for x in values)))
    return result


def run(root, worker, deadline, backend, batch_size, max_batches=0, sweep=False):
    import modal

    volume = modal.Volume.from_name("lm-exploration-teacher-rollouts")
    root = Path(root)
    folder = root / f"producer-{worker}"
    folder.mkdir(exist_ok=True, parents=True)
    progress = folder / "progress.json"
    state = (
        json.loads(progress.read_text())
        if progress.exists()
        else {
            "worker": worker,
            "next_index": 0,
            "generated_tokens": 0,
            "accepted_tokens": 0,
            "records": 0,
            "accepted_records": 0,
            "chunks": [],
            "backend": backend,
        }
    )
    if state["backend"] != backend:
        raise ValueError("Cannot change teacher backend in existing producer")
    if time.time() >= deadline:
        return
    if backend == "vllm":
        from vllm import LLM, SamplingParams

        engine = LLM(
            model=MODEL,
            revision=REVISION,
            tokenizer_revision=REVISION,
            dtype="bfloat16",
            download_dir="/runs/teacher-hf-cache",
            language_model_only=True,
            max_model_len=4096,
            max_num_seqs=batch_size,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.85,
            max_logprobs=64,
            logprobs_mode="raw_logprobs",
        )
        tokenizer = engine.get_tokenizer()
    else:
        from megakernel.model import Decoder, load_teacher
        from megakernel.targets import trajectory

        model, head, tokenizer = load_teacher("/runs/teacher-hf-cache")
        if sweep:
            from megakernel.batch_probe import qualify

            qualify(model, head, tokenizer, folder / "batch-head-probe.json")
            volume.commit()
        engine = Decoder(
            model, head, "/experiment/megakernel/kernel.so", max_seq_len=4096
        )
    tokenizer_path = folder / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    tokenizer_sha = hashlib.sha256(
        (tokenizer_path / "tokenizer.json").read_bytes()
    ).hexdigest()
    batches = 0
    schedule = []
    if sweep:
        sizes = [1, 4, 8, 16, 32, 32, 16, 8, 4, 1] if backend == "vllm" else [32, 32]
        for trial, size in enumerate(sizes):
            for offset in range(0, 32, size):
                schedule.append((trial, size, list(range(offset, offset + size))))
        state["trials"] = {}
    while time.time() < deadline - 30:
        if sweep:
            if batches >= len(schedule):
                break
            trial, size, indices = schedule[batches]
        else:
            indices = list(range(state["next_index"], state["next_index"] + batch_size))
        prompts = [specification(worker, i) for i in indices]
        prompt_ids = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in prompts
        ]
        before = time.monotonic()
        rows = []
        if backend == "vllm":
            params = [
                SamplingParams(
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    max_tokens=2048,
                    logprobs=64,
                    seed=7341 + worker * 10000000 + i,
                )
                for i in indices
            ]
            results = engine.generate(
                [{"prompt_token_ids": p} for p in prompt_ids], params, use_tqdm=False
            )
            for result in results:
                choice = result.outputs[0]
                rows.append(
                    {
                        "token_ids": list(choice.token_ids),
                        "text": choice.text,
                        "finish_reason": choice.finish_reason,
                        "targets": sparse(choice.logprobs, choice.token_ids),
                    }
                )
        else:
            rows = []
            for i, pids in zip(indices, prompt_ids, strict=True):
                result = trajectory(
                    engine,
                    pids,
                    max_tokens=2048,
                    eos_ids=[tokenizer.eos_token_id],
                    seed=7341 + worker * 10000000 + i,
                )
                result["text"] = tokenizer.decode(
                    result["token_ids"], skip_special_tokens=True
                )
                rows.append(result)
        for i, prompt, pids, row in zip(
            indices, prompts, prompt_ids, rows, strict=True
        ):
            row.update(
                id=f"{root.name}:{worker}:{i}",
                worker=worker,
                index=i,
                prompt=prompt,
                prompt_ids=pids,
                teacher=MODEL,
                revision=REVISION,
                tokenizer_sha256=tokenizer_sha,
                backend=backend,
                accepted=accepted(row["text"], row["finish_reason"]),
                verification="JSON shape and Python syntax only; semantics unverified",
                sampling={
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "seed": 7341 + worker * 10000000 + i,
                },
                target_semantics="Raw teacher distribution top-64 plus residual mass; not renormalized top-64. Native Qwen3.5 IDs, incompatible with old student IDs.",
            )
        filename = f"chunk-{len(state['chunks']):06d}.jsonl.gz"
        temp = folder / (filename + ".tmp")
        with gzip.open(temp, "wt") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        temp.replace(folder / filename)
        generated = sum(len(r["token_ids"]) for r in rows)
        kept = sum(len(r["token_ids"]) for r in rows if r["accepted"])
        state["generated_tokens"] += generated
        state["accepted_tokens"] += kept
        state["records"] += len(rows)
        state["accepted_records"] += sum(r["accepted"] for r in rows)
        state["next_index"] += len(rows)
        state["chunks"].append(
            {
                "file": filename,
                "sha256": hashlib.sha256((folder / filename).read_bytes()).hexdigest(),
                "generated_tokens": generated,
                "accepted_tokens": kept,
                "records": len(rows),
            }
        )
        state.update(
            status="generating",
            last_batch_seconds=time.monotonic() - before,
            deadline=deadline,
            updated_at=time.time(),
            tokenizer_sha256=tokenizer_sha,
        )
        atomic_json(progress, state)
        volume.commit()
        if sweep:
            measured = state["trials"].setdefault(
                str(trial),
                {
                    "batch_size": size,
                    "records": 0,
                    "accepted_records": 0,
                    "generated_tokens": 0,
                    "accepted_tokens": 0,
                    "durable_seconds": 0.0,
                },
            )
            measured["records"] += len(rows)
            measured["accepted_records"] += sum(r["accepted"] for r in rows)
            measured["generated_tokens"] += generated
            measured["accepted_tokens"] += kept
            measured["durable_seconds"] += time.monotonic() - before
            measured["accepted_tokens_per_second"] = (
                measured["accepted_tokens"] / measured["durable_seconds"]
            )
            measured["complete"] = measured["records"] == 32
            atomic_json(progress, state)
            volume.commit()
        print(json.dumps({k: v for k, v in state.items() if k != "chunks"}), flush=True)
        batches += 1
        if max_batches and batches >= max_batches:
            break
    state["status"] = "generation_window_completed"
    atomic_json(progress, state)
    volume.commit()


if __name__ == "__main__":
    run(
        sys.argv[1],
        int(sys.argv[2]),
        float(sys.argv[3]),
        sys.argv[4],
        int(sys.argv[5]),
        int(sys.argv[6]) if len(sys.argv) > 6 else 0,
        len(sys.argv) > 7 and sys.argv[7] == "sweep",
    )
