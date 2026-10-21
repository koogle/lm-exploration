"""Streaming synthetic corpus shards for the authorized 500M-token run."""

import ast
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

TOPICS = (
    "factorials and products",
    "binomial coefficients and Pascal's triangle",
    "greatest common divisors and least common multiples",
    "prime numbers and sieves",
    "modular arithmetic",
    "integer digit algorithms",
    "fractions and rational arithmetic",
    "integer square roots",
    "powers and exponentiation",
    "arithmetic progressions",
    "prefix sums and interval queries",
    "difference arrays",
    "sliding windows",
    "two-pointer algorithms",
    "binary search and boundary positions",
    "sorting and stable ties",
    "deduplication with order preservation",
    "frequency dictionaries",
    "set operations",
    "stack algorithms",
    "queue algorithms",
    "balanced parentheses",
    "string parsing",
    "substring matching",
    "run length encoding",
    "palindromes",
    "anagrams",
    "matrix traversal",
    "matrix multiplication",
    "grid neighbors and flood fill",
    "graph breadth-first search",
    "graph depth-first search",
    "topological sorting",
    "shortest paths with nonnegative weights",
    "connected components",
    "tree traversals",
    "memoization",
    "dynamic programming for sequences",
    "knapsack variants",
    "coin change",
    "interval merging",
    "scheduling with greedy algorithms",
    "recursive decomposition",
    "backtracking over small inputs",
    "permutations and combinations",
    "boolean logic and truth tables",
    "bit manipulation",
    "histograms and counting",
    "means medians and exact integer statistics",
    "polynomial evaluation",
    "base conversions",
    "finite state machines",
    "nested list processing",
    "dictionary inversion and grouping",
    "sparse vector operations",
    "distance metrics",
    "calendar arithmetic without date libraries",
    "rational probability on small finite spaces",
    "invariants in loops",
    "monotonic stacks",
    "union find",
    "bounded simulation",
)
CONTEXTS = (
    "a direct mathematical exercise",
    "a library utility",
    "a small scheduling problem",
    "an inventory problem",
    "a board game",
    "text processing",
    "a numeric puzzle",
    "a data cleaning task",
    "an event stream",
    "a resource allocation problem",
    "a classroom worked example",
    "an algorithm interview exercise",
)


def specification(worker, index):
    rng = random.Random(20261022 + worker * 10_000_000 + index)
    topic = TOPICS[rng.randrange(len(TOPICS))]
    context = CONTEXTS[rng.randrange(len(CONTEXTS))]
    level = rng.choice(("introductory", "intermediate", "intermediate", "advanced"))
    boundary = rng.choice(
        (
            "empty or zero inputs",
            "repeated values and ties",
            "inclusive versus exclusive boundaries",
            "negative values where meaningful",
            "single-element inputs",
            "off-by-one cases",
            "impossible or absent results",
        )
    )
    focus = rng.choice(
        (
            "Explain the loop invariant.",
            "Show a small trace of the changing state.",
            "Explain the mathematical definition before the code.",
            "Contrast a plausible wrong solution with the correct one.",
            "Explain why the base case matters.",
            "Show how the data structure represents the problem.",
        )
    )
    numbers = [rng.randint(2, 1000) for _ in range(4)]
    return (
        f"Create a self-contained {level} Python exercise about {topic}, framed as {context}. "
        f"Make {boundary} explicit in the contract. {focus} "
        f"Optional scenario constants: {numbers}. Variation seed: {worker}-{index}. "
        "Invent a concrete problem with a precise input/output contract; do not merely paraphrase the topic. "
        "Provide a complete function named solve, a concise explanation of why it works, and two worked examples. "
        "Use Python's arbitrary-precision integer semantics accurately. Only standard-library imports, no I/O, network or packages. "
        'Return only JSON with string fields "problem", "code", "explanation". Put worked examples in the explanation. '
        "Keep the complete answer under 900 tokens. Do not include the variation seed in the answer."
    )


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def generate(root, worker, commit, target=125_000_000):
    import os
    import shutil
    import sqlite3

    os.environ["HF_HOME"] = "/distill/hf-cache"
    from training_data.subword import load_frozen
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from distillation.modal_curriculum import REVISION, TEACHER

    output = Path(root) / f"producer-{worker}"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "source").exists():
        shutil.copytree(
            "/project/distillation",
            output / "source",
            ignore=shutil.ignore_patterns("__pycache__", "results"),
        )
    progress_file = output / "progress.json"
    state = (
        json.loads(progress_file.read_text())
        if progress_file.exists()
        else {
            "next_index": 0,
            "chunks": [],
            "tokens": 0,
            "accepted_records": 0,
            "attempts": 0,
            "rejected": 0,
            "teacher_tokens": 0,
            "worker": worker,
            "target": target,
            "status": "running",
            "generation_seconds": 0,
            "verification": "syntax and format only; semantics and explanations unverified",
        }
    )
    if state["target"] != target:
        raise ValueError("Cannot change target on resume")
    if state["tokens"] >= target:
        return state
    frozen = load_frozen()
    qwen = AutoTokenizer.from_pretrained(TEACHER, revision=REVISION)
    student_tokenizer = AutoTokenizer.from_pretrained(
        "/distill/qwen4b-shared-50m-20261021-01/tokenizer"
    )
    eos = student_tokenizer.eos_token_id
    seen = sqlite3.connect(output / "seen.sqlite")
    seen.execute(
        "CREATE TABLE IF NOT EXISTS records(hash TEXT PRIMARY KEY, chunk INTEGER)"
    )
    # Roll back dedup state for any batch whose volume transaction was incomplete.
    seen.execute("DELETE FROM records WHERE chunk>=?", (len(state["chunks"]),))
    seen.commit()
    llm = LLM(
        model=TEACHER,
        revision=REVISION,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        max_num_seqs=128,
        enforce_eager=True,
        enable_prefix_caching=True,
        seed=42 + worker,
    )
    params = SamplingParams(temperature=0.85, top_p=0.95, max_tokens=1024)
    started = time.monotonic()
    elapsed_before = state["generation_seconds"]
    while state["tokens"] < target and time.monotonic() - started < 80000:
        prompts = [
            qwen.apply_chat_template(
                [{"role": "user", "content": specification(worker, i)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for i in range(state["next_index"], state["next_index"] + 128)
        ]
        responses = llm.generate(prompts, params, use_tqdm=False)
        chunk = len(state["chunks"])
        accepted, raw = [], []
        chunk_tokens = 0
        for i, response in enumerate(responses):
            completion = response.outputs[0]
            index = state["next_index"] + i
            state["attempts"] += 1
            state["teacher_tokens"] += len(completion.token_ids)
            audit = {
                "index": index,
                "raw": completion.text,
                "finish_reason": completion.finish_reason,
            }
            try:
                answer = json.loads(completion.text)
                if (
                    completion.finish_reason != "stop"
                    or not isinstance(answer, dict)
                    or not all(
                        isinstance(answer.get(k), str) and answer[k].strip()
                        for k in ("problem", "code", "explanation")
                    )
                ):
                    raise ValueError("Invalid response")
                tree = ast.parse(answer["code"])
                if not any(
                    isinstance(n, ast.FunctionDef) and n.name == "solve"
                    for n in tree.body
                ):
                    raise ValueError("Missing solve")
                text = f"### Problem\n{answer['problem']}\n### Python\n{answer['code']}\n### Explanation\n{answer['explanation']}\n"
                fingerprint = hashlib.sha256(text.encode()).hexdigest()
                if seen.execute(
                    "SELECT 1 FROM records WHERE hash=?", (fingerprint,)
                ).fetchone():
                    raise ValueError("Exact duplicate within producer")
                ids = frozen.encode(text).ids + [2]
                if state["tokens"] + chunk_tokens >= target:
                    continue
                # Final record is explicitly truncated only at the global token budget.
                take = min(len(ids), target - state["tokens"] - chunk_tokens)
                ids = ids[:take]
                qids = student_tokenizer.encode(text, add_special_tokens=False) + [eos]
                accepted.append(
                    {
                        "id": fingerprint,
                        "worker": worker,
                        "index": index,
                        "text": text,
                        "ids": ids,
                        "qwen_ids": qids,
                        "truncated_at_budget": take < len(frozen.encode(text).ids) + 1,
                    }
                )
                seen.execute("INSERT INTO records VALUES (?,?)", (fingerprint, chunk))
                chunk_tokens += len(ids)
                audit["accepted"] = True
            except (ValueError, TypeError, SyntaxError, KeyError):
                state["rejected"] += 1
                audit["accepted"] = False
            raw.append(audit)
        filename = f"chunk-{chunk:06d}.jsonl.gz"
        for name, rows in ((filename, accepted), (f"raw-{chunk:06d}.jsonl.gz", raw)):
            temporary = output / (name + ".tmp")
            with gzip.open(temporary, "wt") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            temporary.replace(output / name)
        state["chunks"].append(
            {
                "file": filename,
                "tokens": chunk_tokens,
                "records": len(accepted),
                "sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest(),
            }
        )
        state["tokens"] += chunk_tokens
        state["accepted_records"] += len(accepted)
        state["next_index"] += 128
        state["generation_seconds"] = elapsed_before + time.monotonic() - started
        state["status"] = "completed" if state["tokens"] == target else "running"
        seen.commit()
        atomic_json(progress_file, state)
        commit()
        print(json.dumps({k: v for k, v in state.items() if k != "chunks"}), flush=True)
        if state["attempts"] >= 4096 and not state["tokens"]:
            raise ValueError("No usable generation output")
    if state["tokens"] < target:
        state["status"] = "time_limit_resume_required"
        atomic_json(progress_file, state)
        commit()
    seen.close()
    return {k: v for k, v in state.items() if k != "chunks"}


def available_chunks(root):
    chunks = []
    # Round-robin order is deterministic regardless of producer speed.
    states = [
        json.loads(p.read_text()) if p.exists() else {"chunks": []}
        for p in [Path(root) / f"producer-{i}" / "progress.json" for i in range(4)]
    ]
    for index in range(max((len(s["chunks"]) for s in states), default=0)):
        for worker, state in enumerate(states):
            if index >= len(state["chunks"]):
                if state.get("status") == "completed":
                    continue
                return chunks
            chunks.append((worker, index, state["chunks"][index]))
    return chunks


def read_chunk(root, item):
    worker, _, info = item
    path = Path(root) / f"producer-{worker}" / info["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("Corpus shard hash mismatch")
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]
