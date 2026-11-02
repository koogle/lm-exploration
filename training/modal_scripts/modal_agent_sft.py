"""Agent SFT on mixes of public trajectories, on eight H100s.

Run from training/:
    modal run --detach -m modal_scripts.modal_agent_sft::prepare --mix <name>
    modal run --detach -m modal_scripts.modal_agent_sft::train --mix <name> --base B [--tag 01]
    modal run --detach -m modal_scripts.modal_agent_sft::serve --run <run> [--hours 3]

`MIXES` names each mix: sources, counts and the seed. Sources (`SOURCES`), all pinned:
- `open-swe`: NVIDIA Open-SWE-Traces, mini-swe-agent v2 subset (real SWE-rebench V2 and
  Scale-SWE tasks; Qwen3.6-27B, Qwen3.8-27B and Qwen3.8-Flash-Next teachers), resolved
  trajectories only, at most two per task, in its own bash-tool format
  (`training_data.open_swe_format`).
- `terminal-lego`: Terminal-Lego DeepSeek-V3.2 trajectories on real Stack Overflow
  tasks, reward 1 only (the dataset carries no licence on HF; see
  `plans/AGENTIC_TRAJECTORIES.md`), Terminus 2 format.
- `pr-glm`: our GLM-5.3 trajectories on the PR tasks (resolved, training repositories),
  converted to the Open-SWE bash-only layout; mixes can repeat it.
- `nemotron-terminal`: NVIDIA Nemotron-Terminal-Corpus skill-based synthetic tasks
  (DeepSeek-V3.2), Terminus 2 format.

Every source drops trajectories from the SWE-bench Verified repositories and from the
repositories held out from our PR tasks (`pr_task_agent.heldout`), and drops reasoning.

`prepare` (CPU) samples and tokenizes a mix with the Qwen3 tokenizer into
`lm-exploration-hybrid-runs/agent-mix/<mix>/` (tokens, loss mask, offsets, manifest).
`train` fully fine-tunes a pinned Qwen3 base on it with `torchrun` over eight GPUs
(`modal_scripts.agent_sft_worker`) and writes the model, metrics and summary to
`agentic-reference/<run>/`, where `modal_agentic.py::reference_evaluate` finds it.
`serve` runs a trained model behind vLLM's OpenAI API through a Modal tunnel for Harbor's
Terminal-Bench runs (Terminus 2). All have `retries=0` and a timeout under 24 hours; training stops and saves 30 minutes
before its timeout; one supervisor per launch cancels the call at 24 hours.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import modal
from model_hybrid import modal_benchmarks

from modal_scripts.modal_agentic import (
    LAUNCH_LIMIT_SECONDS,
    REFERENCE_FOLDER,
    REFERENCE_MODELS,
    hf_cache,
    reference_base,
    reference_image,
    runs_volume,
)
from modal_scripts.modal_agentic import (
    volume as data_volume,
)
from workflows.images import add_project

app = modal.App("lm-exploration-agent-sft")
app.include(modal_benchmarks.app)  # the deadline supervisor
MIX_FOLDER = "agent-mix"
CONTEXT = 40960  # Qwen3's position limit; Open-SWE trajectories are long
TRAIN_SECONDS = 10 * 60 * 60
PREPARE_SECONDS = 4 * 60 * 60
TOKENIZER = "Qwen/Qwen3-1.7B-Base"  # the Qwen3 0.6B-4B models share it
SWE_BENCH_VERIFIED_REPOS = {
    "django/django",
    "sympy/sympy",
    "sphinx-doc/sphinx",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "astropy/astropy",
    "pydata/xarray",
    "pytest-dev/pytest",
    "pylint-dev/pylint",
    "psf/requests",
    "mwaskom/seaborn",
    "pallets/flask",
}
SOURCES = {
    "open-swe": {
        "repo": "nvidia/Open-SWE-Traces",
        "revision": "4fa4c2ae9afc6b397bc50391d43450c79993c138",
        "prefixes": [
            f"data/minisweagent/{teacher}/{tasks}/"
            for teacher in ("qwen38_27b", "qwen36_27b", "qwen38_flash_next")
            for tasks in ("swe-rebench-v2", "scale-swe")
        ],
    },
    "terminal-lego": {
        "repo": "Lego-X/Terminal-Lego-Traj-Deepseek-V3-2-15k",
        "revision": "922c2fcb019854911fc05eb1a208d2685ac6e006",
        "files": ["terminal-lego-deepseek-v3-2-15k.json"],
    },
    # Our GLM-5.3 trajectories on the PR tasks (resolved, training repositories),
    # converted to the bash-only Open-SWE layout (`open_swe_format.from_harness`).
    "pr-glm": {"path": "pr-tasks-pilot/agentic/glm-5.3-1793298716/trajectories.jsonl"},
    "nemotron-terminal": {
        "repo": "nvidia/Nemotron-Terminal-Corpus",
        "revision": "a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9",
        "prefixes": ["synthetic_tasks/skill_based/"],
    },
}
MIXES = {
    "smoke-200": {
        "seed": 0,
        "parts": {"open-swe": 100, "terminal-lego": 50, "nemotron-terminal": 50},
    },
    "openswe-20k": {"seed": 0, "parts": {"open-swe": 20000}},
    "openswe-20k-terminal-30k": {
        "seed": 0,
        "parts": {"open-swe": 20000, "terminal-lego": 15000, "nemotron-terminal": 15000},
    },
    # Our own trajectories, all of them, each read 5 times.
    "openswe-20k-pr5": {
        "seed": 0,
        "parts": {"open-swe": 20000, "pr-glm": {"count": None, "repeat": 5}},
    },
    "openswe-20k-terminal-30k-pr5": {
        "seed": 0,
        "parts": {
            "open-swe": 20000,
            "terminal-lego": 15000,
            "nemotron-terminal": 15000,
            "pr-glm": {"count": None, "repeat": 5},
        },
    },
}
SETTINGS = {
    "context": CONTEXT,
    "gpus": 8,
    "accumulation": 2,  # trajectories per GPU per step: 16 per step in all
    "learning_rate": 2e-5,
    "warmup_steps": 30,
    "min_learning_rate_fraction": 0.1,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "loss_chunk": 4096,
    "epochs": 1,
}
cpu_image = reference_image


def excluded(repo):
    from training_data import pr_task_agent

    return repo in SWE_BENCH_VERIFIED_REPOS or pr_task_agent.heldout(repo)


def _files(source):
    from huggingface_hub import HfApi

    spec = SOURCES[source]
    if "files" in spec:
        return spec["files"]
    names = HfApi().list_repo_files(spec["repo"], repo_type="dataset", revision=spec["revision"])
    return sorted(
        n for n in names if n.endswith(".parquet") and n.startswith(tuple(spec["prefixes"]))
    )


def _download(source, name):
    from huggingface_hub import hf_hub_download

    spec = SOURCES[source]
    return hf_hub_download(spec["repo"], name, repo_type="dataset", revision=spec["revision"])


def _sample(source, count, rng):
    """(format, messages) items for one source, oversampled 1.5x for context drops
    (every item when `count` is None)."""
    from pyarrow import parquet
    from training_data import open_swe_format, terminus_format

    want, items = (None if count is None else int(count * 1.5)), []
    if source == "pr-glm":
        from training_data import pr_task_agent

        for line in Path("/data", SOURCES[source]["path"]).read_text().splitlines():
            row = json.loads(line)
            if row["resolved"] and not pr_task_agent.heldout(row["repository"]):
                items.append(("open-swe", open_swe_format.from_harness(row["messages"])))
        rng.shuffle(items)
    elif source == "open-swe":
        files = _files(source)
        rng.shuffle(files)
        per_task = {}
        for name in files:
            columns = ["instance_id", "repo", "resolved", "messages"]
            rows = parquet.read_table(_download(source, name), columns=columns).to_pylist()
            rng.shuffle(rows)
            for row in rows:
                if not row["resolved"] or excluded(row["repo"]):
                    continue
                if per_task.get(row["instance_id"], 0) >= 2:
                    continue
                per_task[row["instance_id"]] = per_task.get(row["instance_id"], 0) + 1
                items.append(("open-swe", open_swe_format.training_messages(row)))
            if len(items) >= want:
                break
    elif source == "terminal-lego":
        rows = json.loads(Path(_download(source, SOURCES[source]["files"][0])).read_text())
        rng.shuffle(rows)
        for row in rows:
            if (row.get("metadata") or {}).get("reward") != 1:
                continue
            messages = terminus_format.training_messages(row["conversations"])
            if messages:
                items.append(("terminus", messages))
    else:
        rows = []
        for name in _files(source):
            table = parquet.read_table(_download(source, name), columns=["conversations"])
            rows.extend(table.to_pylist())
        rng.shuffle(rows)
        for row in rows[:want]:
            messages = terminus_format.training_messages(row["conversations"])
            if messages:
                items.append(("terminus", messages))
    return items[:want]


_tokenizer = None


def _encode(item):
    from training_data import mini_swe_format, open_swe_format, qwen_agent_chat
    from transformers import AutoTokenizer

    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=REFERENCE_MODELS[TOKENIZER])
    tokenize = lambda text: _tokenizer.encode(text, add_special_tokens=False)
    kind, messages = item
    if kind == "open-swe":
        return qwen_agent_chat.encode(tokenize, messages, open_swe_format.TOOLS)
    return mini_swe_format.encode(tokenize, messages)


@app.function(
    image=cpu_image,
    cpu=32,
    memory=131072,
    timeout=PREPARE_SECONDS,
    retries=0,
    volumes={"/runs": runs_volume, "/hf": hf_cache, "/data": data_volume},
)
def prepare_mix(mix, deadline):
    """Sample and tokenize a mix; sources keep their order, examples are shuffled."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    import random
    from multiprocessing import Pool

    import numpy as np

    spec, started = MIXES[mix], time.time()
    rng = random.Random(spec["seed"])
    kept, counts = [], {}
    for source, part in spec["parts"].items():
        count, repeat = (part["count"], part["repeat"]) if isinstance(part, dict) else (part, 1)
        items = _sample(source, count, rng)
        with Pool(30) as pool:
            encoded = pool.map(_encode, items, chunksize=16)
        fits = [(ids, mask) for ids, mask in encoded if len(ids) <= CONTEXT and sum(mask)]
        chosen = fits if count is None else fits[:count]
        counts[source] = {
            "sampled": len(items),
            "fit": len(fits),
            "kept": len(chosen),
            "repeat": repeat,
        }
        kept.extend((source, ids, mask) for ids, mask in chosen * repeat)
    rng.shuffle(kept)
    out = Path("/runs", MIX_FOLDER, mix)
    out.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(ids) for _, ids, _ in kept], dtype=np.int64)
    np.save(out / "offsets.npy", np.concatenate([[0], np.cumsum(lengths)]))
    np.concatenate([np.asarray(ids, dtype=np.uint32) for _, ids, _ in kept]).tofile(
        out / "tokens.u32"
    )
    np.concatenate([np.asarray(mask, dtype=np.uint8) for _, _, mask in kept]).tofile(
        out / "mask.u8"
    )
    manifest = {
        "mix": mix,
        "spec": spec,
        "sources": {s: SOURCES[s] for s in spec["parts"]},
        "tokenizer": [TOKENIZER, REFERENCE_MODELS[TOKENIZER]],
        "context": CONTEXT,
        "counts": counts,
        "examples": len(kept),
        "source_of_example": [source for source, _, _ in kept],
        "tokens": int(lengths.sum()),
        "target_tokens": int(sum(sum(mask) for _, _, mask in kept)),
        "seconds": round(time.time() - started),
    }
    (out / "manifest.json").write_text(json.dumps(manifest) + "\n")
    runs_volume.commit()
    return {k: v for k, v in manifest.items() if k != "source_of_example"}


@app.function(
    image=reference_image,
    gpu="H100:8",
    cpu=32,
    memory=262144,
    timeout=TRAIN_SECONDS,
    retries=0,
    volumes={"/runs": runs_volume, "/hf": hf_cache},
)
def train_mix(base, mix, run, deadline):
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    runs_volume.reload()
    config = {
        "base": base,
        "revision": REFERENCE_MODELS[base],
        "mix": str(Path("/runs", MIX_FOLDER, mix)),
        "out": str(Path("/runs", REFERENCE_FOLDER, run)),
        "settings": SETTINGS,
        "stop_unix": min(deadline - 1800, time.time() + TRAIN_SECONDS - 1800),
    }
    Path(config["out"]).mkdir(parents=True, exist_ok=True)
    config_path = Path(config["out"], "train-config.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    subprocess.run(
        [
            "python",
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={SETTINGS['gpus']}",
            "-m",
            "modal_scripts.agent_sft_worker",
            str(config_path),
        ],
        check=True,
        env=env,
    )
    runs_volume.commit()
    return json.loads(Path(config["out"], "summary.json").read_text())


SERVE_SECONDS = 6 * 60 * 60
IM_END_ID = 151645  # <|im_end|> in the Qwen3 tokenizer
endpoints = modal.Dict.from_name("lm-exploration-agent-endpoints", create_if_missing=True)


def _start_server(run):
    """vLLM's OpenAI server for `agentic-reference/<run>/model` on localhost:8000."""
    runs_volume.reload()
    model = Path("/runs", REFERENCE_FOLDER, run, "model")
    # The models end every turn with <|im_end|>; the Qwen3 base generation config only
    # stops at <|endoftext|>, so vLLM's server would generate past each turn.
    generation = json.loads((model / "generation_config.json").read_text())
    stops = generation.get("eos_token_id")
    stops = stops if isinstance(stops, list) else [stops]
    if IM_END_ID not in stops:
        generation["eos_token_id"] = [IM_END_ID, *[s for s in stops if s is not None]]
        (model / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n")
        runs_volume.commit()
    # Trained at CONTEXT, beyond the Qwen3 base configs' 32,768 positions (RoPE).
    env = {**os.environ, "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
    return subprocess.Popen(
        ["vllm", "serve", str(model), "--served-model-name", run, "--port", "8000",
         "--max-model-len", str(CONTEXT), "--enable-prefix-caching", "--seed", "0",
         # vLLM sizes the RoPE cache from max_position_embeddings; positions past the
         # base config's 32,768 would index outside it.
         "--hf-overrides", json.dumps({"max_position_embeddings": CONTEXT})],
        env=env,
    )  # fmt: skip


def _wait_healthy(server, end):
    import urllib.request

    while time.time() < end:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited with {server.returncode}")
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=5)
            return
        except OSError:
            time.sleep(10)
    raise TimeoutError("vLLM did not become healthy")


@app.function(
    image=reference_image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=SERVE_SECONDS,
    retries=0,
    volumes={"/runs": runs_volume, "/hf": hf_cache},
)
def serve_model(run, seconds, deadline):
    """Serve `agentic-reference/<run>/model` with vLLM's OpenAI API through a Modal tunnel
    for `seconds` (bounded by the timeout and the launch deadline). The tunnel URL is
    stored in the `lm-exploration-agent-endpoints` Dict under the run name."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    server = _start_server(run)
    end = min(time.time() + seconds, deadline - 600, time.time() + SERVE_SECONDS - 300)
    try:
        with modal.forward(8000) as tunnel:
            _wait_healthy(server, end)
            endpoints[run] = {"url": tunnel.url, "until_unix": end}
            print(json.dumps({"run": run, "url": tunnel.url, "until_unix": end}), flush=True)
            while time.time() < end and server.poll() is None:
                time.sleep(30)
    finally:
        endpoints.pop(run, None)
        server.terminate()
    return {"run": run, "served_until": time.time()}


TERMINAL_BENCH = "terminal-bench/terminal-bench-2-1"
TERMINAL_BENCH_SECONDS = 9 * 60 * 60
# Harbor (the Terminal-Bench runner) in its own tool environment, next to vLLM.
terminal_bench_image = add_project(
    reference_base.run_commands(
        "pip install uv==0.9.5",
        "uv tool install 'harbor[modal]==0.25.0' --with 'modal[api-proxy-support]'",
    ).env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin"})
)


@app.function(
    image=terminal_bench_image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=TERMINAL_BENCH_SECONDS,
    retries=0,
    volumes={"/runs": runs_volume, "/hf": hf_cache},
)
def terminal_bench(run, attempts, concurrency, deadline):
    """Terminal-Bench 2.1 for a trained run: vLLM on localhost and Harbor's Terminus 2 in
    the same container, task sandboxes on Modal (each capped below this call's end),
    temperature 0.7. Harbor's job folder goes to `terminal-bench/<run>/` on the runs
    volume; returns the reward summary."""
    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing work")
    started = time.time()
    end = min(deadline - 900, started + TERMINAL_BENCH_SECONDS - 900)
    server = _start_server(run)
    out = Path("/runs", "terminal-bench", run)
    out.mkdir(parents=True, exist_ok=True)
    try:
        _wait_healthy(server, end)
        model_info = {"max_input_tokens": CONTEXT, "max_output_tokens": 4096,
                      "input_cost_per_token": 0, "output_cost_per_token": 0}  # fmt: skip
        command = [
            "harbor", "run", "-d", TERMINAL_BENCH, "-a", "terminus-2",
            "-m", f"hosted_vllm/{run}", "--ak", "api_base=http://localhost:8000/v1",
            "--ak", "temperature=0.7", "--ak", f"model_info={json.dumps(model_info)}",
            "-e", "modal", "--ek", f"sandbox_timeout_secs={int(end - time.time())}",
            "-k", str(attempts), "-n", str(concurrency), "-o", str(out),
        ]  # fmt: skip
        # Harbor's preflight looks for a Modal token file or variables; inside a Modal
        # container the SDK authenticates with the container's own credentials.
        Path.home().joinpath(".modal.toml").touch()
        env = {**os.environ, "HOSTED_VLLM_API_KEY": "none"}
        subprocess.run(command, env=env, timeout=max(60, end - time.time()), check=False)
    finally:
        server.terminate()
        runs_volume.commit()
    rewards, exceptions = [], {}
    for result in out.glob("*/*/result.json"):
        data = json.loads(result.read_text())
        reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        rewards.append(reward == 1)
        kind = (data.get("exception_info") or {}).get("exception_type") or "none"
        exceptions[kind] = exceptions.get(kind, 0) + 1
    summary = {"run": run, "dataset": TERMINAL_BENCH, "trials": len(rewards),
               "solved": sum(rewards), "exceptions": exceptions,
               "seconds": round(time.time() - started)}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    runs_volume.commit()
    return summary


def _launch(function, *arguments):
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    call = function.spawn(*arguments, deadline)
    supervisor = modal_benchmarks.enforce_deadline.spawn([call.object_id], deadline)
    print(
        json.dumps(
            {"call": call.object_id, "supervisor": supervisor.object_id, "deadline_unix": deadline}
        )
    )


@app.local_entrypoint()
def bench(run: str, attempts: int = 1, concurrency: int = 24):
    """Terminal-Bench 2.1 for a trained run, entirely on Modal."""
    _launch(terminal_bench, run, attempts, concurrency)


@app.local_entrypoint()
def serve(run: str, hours: float = 3.0):
    """Serve a trained run for Harbor; read the URL from the endpoints Dict."""
    _launch(serve_model, run, int(hours * 3600))


@app.local_entrypoint()
def prepare(mix: str):
    print(json.dumps({"results": f"lm-exploration-hybrid-runs/{MIX_FOLDER}/{mix}/"}))
    _launch(prepare_mix, mix)


@app.local_entrypoint()
def train(mix: str, base: str = "Qwen/Qwen3-1.7B-Base", tag: str = "01"):
    run = f"{base.split('/')[1].lower()}-{mix}-{tag}"
    print(json.dumps({"results": f"lm-exploration-hybrid-runs/{REFERENCE_FOLDER}/{run}/"}))
    _launch(train_mix, base, mix, run)
