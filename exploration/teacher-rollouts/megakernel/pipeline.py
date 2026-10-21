"""Qualify the selected kernel, then measure the shared durable-output pipeline."""

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import Store, atomic_json
from runner import sample_memory, trial


def run(root, attempt, manifest):
    import modal
    import torch

    from megakernel.model import Decoder, load_teacher
    from megakernel.probe import run as probe

    c = manifest["config"]
    if c["temperature"] != 0 or c["max_tokens"] != 128:
        raise ValueError(
            "Qualified pipeline requires greedy generation and max_tokens=128"
        )
    store = Store(root, manifest)
    (store.root / "validated").mkdir(exist_ok=True)
    out = store.root / "attempts" / attempt
    out.mkdir(parents=True, exist_ok=False)
    volume = modal.Volume.from_name("lm-exploration-teacher-rollouts")
    started = time.perf_counter()
    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(stop, samples), daemon=True)
    sampler.start()
    report = {
        "status": "starting",
        "backend": "manual-t512-g132",
        "trials": [],
        "actual_billed_usd": None,
        "output_token_count_excludes_eos": True,
        "concurrency": 1,
        "max_cache_tokens": 1024,
    }

    def save():
        report["total_pipeline_seconds"] = time.perf_counter() - started
        atomic_json(out / "report.json", report)
        volume.commit()

    try:
        teacher = load_teacher("/runs/teacher-hf-cache")
        report["model_load_and_download_seconds"] = time.perf_counter() - started
        library = Path("/experiment/megakernel/pipeline.so")
        qualification = probe(
            out / "qualification", "probe", manifest, teacher, library
        )
        report["qualification"] = qualification
        if not (
            qualification.get("correctness_passed")
            and qualification.get("reset_reproducible")
            and qualification.get("performance_eligible")
        ):
            raise RuntimeError(
                "Selected kernel failed qualification; no pipeline benchmark"
            )
        model, head, tokenizer = teacher
        decoder = Decoder(model, head, library)

        @torch.inference_mode()
        def generate(path, payload):
            text = tokenizer.apply_chat_template(
                payload["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            generated = decoder.generate(
                ids, payload["max_tokens"], tokenizer.eos_token_id
            )
            return {
                "choices": [
                    {
                        "message": {
                            "content": tokenizer.decode(
                                generated, skip_special_tokens=True
                            )
                        },
                        "finish_reason": "length"
                        if len(generated) == payload["max_tokens"]
                        else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(ids),
                    "completion_tokens": len(generated),
                },
                "backend": "manual-t512-g132",
                "completion_tokens_exclude_eos": True,
            }

        for round_id in range(c["rounds"]):
            # One request worker: Decoder holds mutable sequence state, so never batch it.
            result = asyncio.run(
                trial(
                    c,
                    manifest["prompts"],
                    store,
                    1,
                    round_id,
                    attempt,
                    volume.commit,
                    request_fn=generate,
                    runtime_label="Luce-derived Qwen3.5-9B CUDA12.8.1 C++20; torch==2.10.0",
                )
            )
            result["timing_scope"] = (
                "request dispatch through durable raw output; no HTTP; excludes post-hoc validation"
            )
            report["trials"].append(result)
            save()
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop.set()
        sampler.join(timeout=6)
        report["peak_sampled_gpu_memory_mib"] = max(
            (s["used_mib"] for s in samples), default=None
        )
        report["estimated_attempt_gpu_usd"] = (
            (time.perf_counter() - started) * c["gpu_usd_per_hour"] / 3600
        )
        atomic_json(out / "memory.json", samples)
        save()
    return report


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], json.loads(Path(sys.argv[3]).read_text()))
