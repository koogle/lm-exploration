"""Paired live episodes using the existing sandboxed PR-task evaluation harness.

Run from training/. Task IDs and checkpoint variants are supplied explicitly.
One-hour launch deadline; the remote guard tracks both calls and every sandbox.
"""

import argparse
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import modal

if modal.is_local():
    import truststore

    truststore.inject_into_ssl()

from modal_scripts import modal_agentic as agentic

app = modal.App("qwen-weight-space-live-check")
registry = modal.Dict.from_name("qwen-weight-space-launches", create_if_missing=True)
image = agentic.reference_image


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=3300,
    retries=0,
    volumes={"/runs": agentic.runs_volume, "/data": agentic.volume, "/hf": agentic.hf_cache},
)
def evaluate(variant, run, deadline, config):
    if time.time() >= deadline - 300:
        raise TimeoutError("Live evaluation launch deadline reached")
    allowed = config["task_ids"]
    select = agentic._select
    previous_app = agentic.app

    def heldout_select(tasks, limit, heldout_only=False):
        by_id = {f"{t['repository']}@{t['sha']}": t for t in tasks}
        selected = [by_id[key] for key in allowed]
        if heldout_only and not all(t["heldout"] for t in selected):
            raise ValueError("Live evaluation includes a training task")
        return selected

    agentic._select = heldout_select
    # The existing harness creates its execution and grading sandboxes in this app.
    agentic.app = app
    original_recorder = agentic._recorder

    def recorder(folder):
        record = original_recorder(folder)
        ids, lock = [], threading.Lock()

        def tracked(sandbox_id):
            with lock:
                ids.append(sandbox_id)
                registry[run] = list(ids)
            record(sandbox_id)

        return tracked

    agentic._recorder = recorder
    # Preserve unparsed replies so an invalid tool call can be distinguished from
    # an intentional answer or a generation cut off at the token limit.
    import vllm

    original_llm = vllm.LLM
    raw_path = Path("/runs/weight-space", config["experiment"], f"raw-{run}.jsonl")

    class RecordedLLM(original_llm):
        def generate(self, *args, **kwargs):
            outputs = super().generate(*args, **kwargs)
            with raw_path.open("a") as handle:
                for output in outputs:
                    completion = output.outputs[0]
                    handle.write(
                        json.dumps(
                            {
                                "prompt_sha256": hashlib.sha256(
                                    json.dumps(output.prompt_token_ids).encode()
                                ).hexdigest(),
                                "text": completion.text,
                                "finish_reason": completion.finish_reason,
                                "stop_reason": completion.stop_reason,
                                "tokens": len(completion.token_ids),
                            }
                        )
                        + "\n"
                    )
            return outputs

    vllm.LLM = RecordedLLM
    source = variant["model_run"]
    if variant.get("export_path"):
        root = Path("/runs/agentic-reference", source)
        root.mkdir()
        shutil.copytree(variant["export_path"], root / "model")
        metadata = json.loads(
            Path("/runs/agentic-reference", config["source_run"], "summary.json").read_text()
        )
        metadata["weight_edit"] = variant
        (root / "summary.json").write_text(json.dumps(metadata, indent=2))
        agentic.runs_volume.commit()
    try:
        result = agentic.evaluate_reference.local(
            f"sft:{source}",
            "open-swe",
            0.7,
            "pr-tasks-pilot",
            run,
            len(allowed),
            8,
            config["max_turns"],
            deadline,
        )
        from evaluation.agentic_behavior import behavior

        rows = [
            json.loads(line)
            for line in Path("/data/pr-tasks-pilot/agentic", run, "trajectories.jsonl")
            .read_text()
            .splitlines()
        ]
        result["behavior"] = behavior(rows)
        output = Path("/runs/weight-space", config["experiment"], f"live-{run}.json")
        output.write_text(json.dumps(result, indent=2))
        agentic.runs_volume.commit()
        return result
    finally:
        for sandbox_id in registry.get(run, []):
            try:
                modal.Sandbox.from_id(sandbox_id).terminate()
            except Exception as error:  # noqa: BLE001 - already terminated sandboxes are harmless
                print(f"Sandbox cleanup {sandbox_id}: {type(error).__name__}", flush=True)
        agentic.volume.commit()
        agentic._select = select
        agentic._recorder = original_recorder
        agentic.app = previous_app
        vllm.LLM = original_llm


@app.function(image=image, cpu=0.25, memory=1024, timeout=3900, retries=0)
def supervise(calls, runs, deadline):
    print(f"Live supervisor active until {deadline}", flush=True)
    while True:
        complete = True
        for call_id in calls:
            try:
                modal.FunctionCall.from_id(call_id).get(timeout=0)
            except TimeoutError:
                complete = False
            except Exception as error:  # noqa: BLE001 - retain cancellation on uncertain API failures
                print(type(error).__name__, flush=True)
                complete = False
        if complete or time.time() >= deadline:
            for call_id in calls:
                modal.FunctionCall.from_id(call_id).cancel()
            for run in runs:
                for sandbox_id in registry.get(run, []):
                    try:
                        modal.Sandbox.from_id(sandbox_id).terminate()
                    except Exception as error:  # noqa: BLE001 - cleanup is idempotent
                        print(type(error).__name__, flush=True)
            return
        time.sleep(min(20, max(0, deadline - time.time())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    deadline = time.time() + 3600
    stamp = int(time.time())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    variants = config["variants"]
    runs = [
        f"weight-space-{config['experiment']}-{variant['name']}-{stamp}" for variant in variants
    ]
    for run in runs:
        registry[run] = []
    with modal.enable_output(), app.run(detach=True):
        calls = [
            evaluate.spawn(variant, run, deadline, config)
            for variant, run in zip(variants, runs, strict=True)
        ]
        ids = [call.object_id for call in calls]
        ledger = {
            "calls": ids,
            "runs": runs,
            "deadline_unix": deadline,
            "worker_timeout_seconds": 3300,
            "retries": 0,
            "config": config,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (out / f"launch-live-{stamp}.json").write_text(json.dumps(ledger, indent=2))
        try:
            guard = supervise.spawn(ids, runs, deadline)
        except BaseException:
            for call in calls:
                call.cancel()
            raise
        ledger["supervisor_id"] = guard.object_id
        (out / f"launch-live-{stamp}.json").write_text(json.dumps(ledger, indent=2))
        print(json.dumps(ledger), flush=True)
        results = [call.get() for call in calls]
        (out / f"live-result-{stamp}.json").write_text(json.dumps(results, indent=2))
        (out / "live-result.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
