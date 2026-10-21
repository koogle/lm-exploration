"""Run inside the bounded Modal worker; stdlib HTTP client to vLLM's scheduler."""

import asyncio
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core import Store, atomic_json, digest, select_best, summarize, validate

URL = "http://127.0.0.1:8000"


def request(path, body=None, timeout=90):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        URL + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode()
        return json.loads(raw) if raw else None


def sample_memory(stop, samples):
    while not stop.is_set():
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            samples.append({"time": time.time(), "used_mib": int(result.strip())})
        except (subprocess.SubprocessError, ValueError):
            pass
        stop.wait(0.2)


def body(c, prompt, seed, max_tokens=None):
    return {
        "model": c["model"],
        "messages": [{"role": "user", "content": prompt["prompt"]}],
        "temperature": c["temperature"],
        "top_p": c["top_p"],
        "top_k": c["top_k"],
        "max_tokens": max_tokens or c["max_tokens"],
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def trial(c, prompts, store, concurrency, round_id, attempt, commit):
    trial_id = f"c{concurrency}-r{round_id}"
    summary_path = store.root / f"{trial_id}.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    # Python's default thread pool can be smaller than requested concurrency.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=concurrency)
    )
    sem = asyncio.Semaphore(concurrency)
    records = []
    resumed = False
    jobs = []
    persistence_seconds = 0.0
    for repeat in range(c["repeats"]):
        for prompt in prompts:
            seed = c["seed"] + repeat  # identical across concurrency and rounds
            identity = {
                "trial": trial_id,
                "prompt_id": prompt["id"],
                "repeat": repeat,
                "manifest": digest(
                    json.loads((store.root / "manifest.json").read_text())
                ),
            }
            key = digest(identity)
            old = store.read(key)
            if old is not None:
                resumed = True
                records.append(old)
            else:
                jobs.append((prompt, seed, key))

    async def one(prompt, seed, key):
        nonlocal persistence_seconds
        async with sem:
            started = time.perf_counter()
            payload = body(c, prompt, seed)
            record = {
                "schema_version": 1,
                "example_id": key,
                "manifest_sha256": digest(
                    json.loads((store.root / "manifest.json").read_text())
                ),
                "teacher": c["model"],
                "teacher_revision": c["revision"],
                "tokenizer_revision": c["revision"],
                "runtime": c["runtime"],
                "prompt_id": prompt["id"],
                "synthetic": True,
                "trial_id": trial_id,
                "attempt": attempt,
                "request": payload,
                "prompt": prompt,
                "output": "",
                "finish_reason": None,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "error": None,
                "verification_seconds": 0,
                "validation": None,
            }
            try:
                response = await asyncio.to_thread(
                    request, "/v1/chat/completions", payload
                )
                choice = response["choices"][0]
                record.update(
                    output=choice["message"]["content"] or "",
                    finish_reason=choice["finish_reason"],
                    prompt_tokens=response["usage"]["prompt_tokens"],
                    output_tokens=response["usage"]["completion_tokens"],
                    response=response,
                )
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["request_seconds"] = time.perf_counter() - started
            # Raw output is durable before verification. Separate files remain immutable.
            t = time.perf_counter()
            store.put(record)
            commit()
            persistence_seconds += time.perf_counter() - t
            records.append(record)

    before = time.perf_counter()
    await asyncio.gather(*(one(*job) for job in jobs))
    generation_seconds = max(time.perf_counter() - before, 1e-9)
    # Keep verification outside the generation timing and GPU scheduler queue.
    for r in records:
        t = time.perf_counter()
        r["validation"] = validate(r["prompt"], r["output"], r["finish_reason"])
        if r["error"]:
            r["validation"] = {"accepted": False, "reasons": ["request_error"]}
        r["verification_seconds"] = time.perf_counter() - t
        atomic_json(store.root / "validated" / f"{r['example_id']}.json", r)
    result = summarize(records, generation_seconds, c["gpu_usd_per_hour"])
    result.update(
        trial_id=trial_id,
        concurrency=concurrency,
        round=round_id,
        comparable=not resumed,
        resumed=resumed,
        attempt=attempt,
        persistence_seconds=persistence_seconds,
        timing_scope="HTTP dispatch through durable raw output; excludes post-hoc validation",
    )
    if resumed:
        # Mixed-attempt records have no valid common elapsed-time denominator.
        for k in [
            "generated_tokens_per_second",
            "completed_examples_per_hour",
            "accepted_examples_per_hour",
            "estimated_steady_gpu_usd_per_accepted",
        ]:
            result[k] = None
    atomic_json(summary_path, result)
    commit()
    return result


def run(root, attempt, manifest, commit=lambda: None):
    start = time.perf_counter()
    c, prompts = manifest["config"], manifest["prompts"]
    store = Store(root, manifest)
    (store.root / "validated").mkdir(exist_ok=True)
    attempt_dir = store.root / "attempts" / attempt
    attempt_dir.mkdir(parents=True, exist_ok=False)
    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(stop, samples), daemon=True)
    sampler.start()
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        c["model"],
        "--revision",
        c["revision"],
        "--tokenizer-revision",
        c["revision"],
        "--dtype",
        "bfloat16",
        "--language-model-only",
        "--host",
        "127.0.0.1",
        "--max-model-len",
        str(c["max_model_len"]),
        "--max-num-seqs",
        str(max(c["concurrency"])),
        "--gpu-memory-utilization",
        "0.85",
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--seed",
        str(c["seed"]),
        "--profiler-config",
        json.dumps(
            {
                "profiler": "torch",
                "torch_profiler_dir": str(attempt_dir / "traces"),
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": True,
            }
        ),
    ]
    atomic_json(
        attempt_dir / "runtime.json",
        {
            "command": cmd,
            "python": sys.version,
            "packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
            },
            "gpu": subprocess.check_output(["nvidia-smi"], text=True),
            "actual_billed_usd": None,
        },
    )
    log = (attempt_dir / "server.log").open("w")
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ, "VLLM_SERVER_DEV_MODE": "1"},
    )
    report = {"status": "starting", "startup_to_ready_seconds": None, "trials": []}
    try:
        deadline = start + min(600, c["runtime_seconds"] - 120)
        while True:
            if proc.poll() is not None:
                raise RuntimeError("vLLM exited; inspect server.log")
            try:
                request("/health", timeout=2)
                break
            except (OSError, ValueError):
                if time.perf_counter() > deadline:
                    raise TimeoutError("model startup deadline")
                time.sleep(1)
        report["startup_to_ready_seconds"] = time.perf_counter() - start
        # Separate warm-up; no benchmark record, no acceptance or throughput credit.
        t = time.perf_counter()
        request("/v1/chat/completions", body(c, prompts[0], c["seed"], 16))
        report["warmup_seconds"] = time.perf_counter() - t
        for round_id in range(c["rounds"]):
            order = (
                c["concurrency"]
                if round_id % 2 == 0
                else list(reversed(c["concurrency"]))
            )
            for concurrency in order:
                metrics = urllib.request.urlopen(URL + "/metrics", timeout=10).read()
                (attempt_dir / f"c{concurrency}-r{round_id}-before.prom").write_bytes(
                    metrics
                )
                result = asyncio.run(
                    trial(c, prompts, store, concurrency, round_id, attempt, commit)
                )
                report["trials"].append(result)
                (attempt_dir / f"c{concurrency}-r{round_id}-after.prom").write_bytes(
                    urllib.request.urlopen(URL + "/metrics", timeout=10).read()
                )
        best = select_best(report["trials"], c["rounds"])
        profile_done = (
            best is not None and (store.root / f"c{best}-rprofile.json").exists()
        )
        if best is not None and not profile_done:
            report["best_concurrency"] = best
            request("/start_profile", {})
            t = time.perf_counter()
            try:
                asyncio.run(trial(c, prompts, store, best, "profile", attempt, commit))
            finally:
                request("/stop_profile", {})
            report["profile_pipeline_seconds"] = time.perf_counter() - t
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
        stop.set()
        sampler.join(timeout=6)
        load_times = re.findall(
            r"Loading model weights took ([0-9.]+) seconds",
            (attempt_dir / "server.log").read_text(),
        )
        report["model_weight_loading_seconds_from_log"] = (
            float(load_times[-1]) if load_times else None
        )
        report["total_pipeline_seconds"] = time.perf_counter() - start
        report["estimated_attempt_gpu_usd"] = (
            report["total_pipeline_seconds"] * c["gpu_usd_per_hour"] / 3600
        )
        accepted = sum(
            r["accepted"]
            for r in report["trials"]
            if not r["resumed"] and r["attempt"] == attempt
        )
        report["estimated_pipeline_gpu_usd_per_accepted"] = (
            report["estimated_attempt_gpu_usd"] / accepted
            if accepted
            and all(
                not r["resumed"] and r["attempt"] == attempt for r in report["trials"]
            )
            else None
        )
        report["actual_billed_usd"] = None
        report["peak_sampled_gpu_memory_mib"] = max(
            (s["used_mib"] for s in samples), default=None
        )
        atomic_json(attempt_dir / "memory.json", samples)
        atomic_json(attempt_dir / "report.json", report)
        commit()
    return report


if __name__ == "__main__":
    import modal

    volume = modal.Volume.from_name("lm-exploration-teacher-rollouts")
    manifest = json.loads(Path(sys.argv[3]).read_text())
    run(sys.argv[1], sys.argv[2], manifest, volume.commit)
