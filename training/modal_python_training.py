"""Durable Python 124M / exact 5B workflow; explicit start is required."""

import json
import time
from pathlib import Path

import modal

app = modal.App("simple-llm-python-training")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
run_volume = modal.Volume.from_name("lm-exploration-training")
RUN = "python-124m-muon-5b-20261021-v1"
REPO = "jfrick/simple-llm-python-124m-5b"
SNAPSHOT = "/data/python-124m-5b-20261021-snapshot-v1"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "numpy",
        "pydantic>=2.11,<3",
        "tokenizers==0.22.2",
        "huggingface-hub==1.33.0",
        "safetensors>=0.4",
        "matplotlib",
    )
    .env({"PYTHONPATH": "/project", "RAYON_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("100m", "/project/model_100m")
    .add_local_dir("10m", "/project/model_10m")
    .add_local_dir("trainer", "/project/trainer")
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer/structured-code-v1", "/project/tokenizer/structured-code-v1")
    .add_local_dir("tests", "/project/tests")
    .add_local_dir("publication", "/project/publication")
    .add_local_dir("evaluation", "/project/evaluation")
    .add_local_file("recipes/distributed/python-124m-muon-5b.json", "/project/settings.json")
)
sandbox_image = modal.Image.debian_slim(python_version="3.12")


def record(stage, **details):
    from trainer.distributed_train import atomic_json

    folder = Path("/runs") / RUN
    folder.mkdir(parents=True, exist_ok=True)
    value = {"run_id": RUN, "stage": stage, "updated_unix": time.time(), **details}
    atomic_json(folder / "workflow.json", value)
    run_volume.commit()
    print(json.dumps(value), flush=True)
    return value


@app.function(
    image=image, gpu="H100", cpu=8, memory=32768, timeout=3600, volumes={"/runs": run_volume}
)
def qualify():
    import torch
    from model_100m.model import GPT, ModelConfig
    from torch.nn import functional as F
    from training_data.subword import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, load_frozen

    from trainer.distributed_config import DistributedConfig
    from trainer.optimizers import make_optimizer
    from trainer.python_evaluation import generate

    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = GPT(ModelConfig(vocab_size=32768, block_size=8192)).cuda()
    tokenizer = load_frozen()
    text = "def add(a, b):\n    return a + b\n"
    ids = [BOS_TOKEN_ID, *tokenizer.encode(text, add_special_tokens=False).ids, EOS_TOKEN_ID]
    x = torch.tensor([ids[:-1]], device="cuda")
    y = torch.tensor([ids[1:]], device="cuda")
    with torch.no_grad():
        before = model(x).clone()
        model.eval()
        torch.testing.assert_close(model(x), before, rtol=0, atol=0)
        changed = x.clone()
        changed[:, -1] = PAD_TOKEN_ID
        torch.testing.assert_close(model(changed)[:, :-1], before[:, :-1], rtol=0, atol=0)
        prefix = x[:, :8]
        torch.testing.assert_close(model(prefix), before[:, :8], rtol=1e-4, atol=1e-5)
        for block in model.blocks:
            block.attention.cache_rotary(8192)
        torch.testing.assert_close(model(x), before, rtol=1e-4, atol=1e-5)
    model.train()
    config = DistributedConfig(**json.loads(Path("/project/settings.json").read_text()))
    optimizer = make_optimizer(model, config)
    params = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if len(params) != len(set(params)) or len(params) != len(list(model.parameters())):
        raise ValueError("Optimizer parameter partition is not exhaustive and unique")
    if id(model.token_embedding.weight) not in {
        id(p) for g in optimizer.adamw.param_groups for p in g["params"]
    }:
        raise ValueError("Tied embedding is not assigned to auxiliary optimizer")
    initial = float(F.cross_entropy(before.flatten(0, 1), y.flatten()))
    losses = []
    for step in range(160):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise ValueError("Non-finite tiny-fit loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss))
        if step >= 20 and losses[-1] < 0.02:
            break
    model.eval()
    sample = generate(
        model, tokenizer, "def add(a, b):\n    ", temperature=0.0, seed=42, max_tokens=32
    )
    if losses[-1] >= 0.1 or "return a + b" not in sample["completion"]:
        raise ValueError("Tiny-set fitting or simple Python completion failed")
    result = {
        "passed": True,
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_loss": initial,
        "final_loss": losses[-1],
        "updates": len(losses),
        "losses": losses,
        "completion": sample,
        "logit_alignment": "passed",
        "causality": "passed",
        "cached_rope": "passed",
        "optimizer_partition": "passed",
        "note": "Separate diagnostic model; its weights are discarded",
    }
    Path("/runs", RUN).mkdir(exist_ok=True)
    Path("/runs", RUN, "qualification.json").write_text(json.dumps(result, indent=2))
    run_volume.commit()
    return result


@app.function(
    image=image,
    gpu="H100:8",
    cpu=32,
    memory=131072,
    timeout=25200,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def train_segment(
    index: int, resume: str = "", max_steps: int | None = None, stop_tokens: int | None = None
):
    import os
    import shutil
    import subprocess

    import torch

    started = time.monotonic()
    if torch.cuda.device_count() != 8:
        raise ValueError("Expected eight H100s")
    source = Path(SNAPSHOT)
    local = Path("/tmp/python-snapshot")
    if not (local / "manifest.json").exists():
        shutil.copytree(source, local)
    output = Path("/runs") / f"{RUN}-s{index:03d}"
    if output.exists():
        raise ValueError("Refuse to overwrite training segment")
    output.mkdir()
    for package in ("model_100m", "model_10m", "trainer", "training_data"):
        shutil.copytree(
            Path("/project") / package,
            output / "source" / package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    shutil.copytree("/project/tokenizer/structured-code-v1", output / "tokenizer")
    shutil.copyfile(local / "manifest.json", output / "data-manifest.json")
    shutil.copyfile("/project/settings.json", output / "request.json")
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "trainer.distributed_train",
        "--config",
        str(output / "request.json"),
        "--manifest",
        str(local / "manifest.json"),
        "--output",
        str(output),
        "--max-seconds",
        "21600",
        "--volume-name",
        "lm-exploration-training",
    ]
    if resume:
        if not resume.startswith("/runs/" + RUN + "-s") or not Path(resume).is_file():
            raise ValueError("Untrusted resume checkpoint")
        command.extend(["--resume", resume])
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if stop_tokens is not None:
        command.extend(["--stop-tokens", str(stop_tokens)])
    try:
        with (output / "torchrun.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=22200)
            except subprocess.TimeoutExpired:
                import signal

                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=120)
                raise
        if code:
            raise RuntimeError((output / "torchrun.log").read_text()[-6000:])
        report = json.loads((output / "report.json").read_text())
        report["worker_elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        run_volume.commit()


@app.function(
    image=image, gpu="H100", cpu=8, memory=32768, timeout=3600, volumes={"/runs": run_volume}
)
def generations(checkpoint: str):
    from trainer.python_evaluation import evaluate_checkpoint

    result = evaluate_checkpoint(checkpoint, "/project/evaluation/python-tasks.json")
    folder = Path("/runs", RUN, "evaluations")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{result['tokens']:012d}-generation.json"
    path.write_text(json.dumps(result, indent=2))
    run_volume.commit()
    return str(path)


@app.function(image=sandbox_image, cpu=1, memory=512, timeout=90, max_containers=8, retries=0)
def execute_case(case: dict):
    import uuid

    marker = uuid.uuid4().hex
    candidate = case.get("code", case.get("prompt", "") + case.get("completion", ""))
    payload = {"code": candidate, "tests": case["tests"], "imports": case.get("test_imports", [])}
    # Generated code sees no credentials, network, mounted volumes or host files.
    # Its stdout/stderr are discarded; only the harness's final marker is read.
    script = f"""
import os, resource
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024*1024, 1024*1024))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
os.environ.clear()
result_fd = os.dup(1)
null = os.open('/dev/null', os.O_WRONLY)
os.dup2(null, 1)
os.dup2(null, 2)
payload = {payload!r}
namespace = {{'__name__': '__main__'}}
try:
    for statement in payload['imports']:
        exec(compile(statement, '<setup>', 'exec'), namespace)
    exec(compile(payload['code'], '<candidate>', 'exec'), namespace)
    for statement in payload['tests']:
        exec(compile(statement, '<hidden-test>', 'exec'), namespace)
except BaseException:
    os.write(result_fd, b'FAILED')
else:
    os.write(result_fd, {marker.encode()!r})
"""
    sb = modal.Sandbox.create(
        "python",
        "-I",
        "-S",
        "-c",
        script,
        app=modal.App.lookup("simple-llm-python-training"),
        image=sandbox_image,
        cpu=1,
        memory=512,
        timeout=10,
        block_network=True,
        secrets=[],
        volumes={},
        include_oidc_identity_token=False,
    )
    try:
        sb.wait()
        output = sb.stdout.read()
        return {
            "task_id": case.get("task_id"),
            "passed": output == marker and sb.returncode == 0,
            "sandbox_id": sb.object_id,
            "returncode": sb.returncode,
            "network": "blocked",
            "credentials": "none",
            "cpu_seconds_limit": 2,
            "wall_seconds_limit": 10,
            "address_space_bytes_limit": 268435456,
        }
    finally:
        sb.terminate()


@app.function(image=image, cpu=2, memory=4096, timeout=1800, volumes={"/runs": run_volume})
def execution_checks(generation_path: str = ""):
    if generation_path:
        run_volume.reload()
        result = json.loads(Path(generation_path).read_text())
        cases = result["functions"]
    else:
        tasks = json.loads(Path("/project/evaluation/python-tasks.json").read_text())["tasks"]
        cases = [
            {
                "task_id": t["task_id"],
                "code": t["reference"],
                "tests": t["tests"],
                "test_imports": t["test_imports"],
            }
            for t in tasks
        ]
        cases.append(
            {
                "task_id": "isolation_probe",
                "code": "import os, socket\nassert not os.environ\ntry:\n    s = socket.create_connection(('example.com', 443), timeout=1)\nexcept OSError:\n    blocked = True\nelse:\n    s.close()\n    blocked = False\n",
                "tests": ["assert blocked"],
                "test_imports": [],
            }
        )
        result = {"reference_harness": True}
    scores = list(execute_case.map(cases))
    result["execution"] = scores
    result["execution_pass_rate"] = sum(s["passed"] for s in scores) / len(scores)
    if not generation_path and not all(s["passed"] for s in scores):
        raise ValueError("Reference solutions failed isolated evaluator: " + json.dumps(scores))
    if generation_path:
        samples = result["samples"]
        result["summary"] = {
            "syntax_valid_fraction": sum(s["syntax_valid"] for s in samples) / len(samples),
            "mean_repeated_4gram_fraction": sum(s["repeated_token_4gram_fraction"] for s in samples)
            / len(samples),
            "long_loop_fraction": sum(s["long_loop"] for s in samples) / len(samples),
            "mean_new_tokens": sum(s["new_tokens"] for s in samples) / len(samples),
            "eos_fraction": sum(s["stop_reason"] == "eos" for s in samples) / len(samples),
        }
        for case in result["functions"]:
            case.pop("tests", None)
            case.pop("test_imports", None)
        path = Path(generation_path.replace("-generation.json", "-evaluated.json"))
    else:
        path = Path("/runs", RUN, "execution-harness.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    run_volume.commit()
    return str(path)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": run_volume, "/data": data_volume},
)
def publish(checkpoint: str, evaluation_path: str):
    import shutil

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError
    from training_data.subword import digest

    from trainer.hub_checkpoint import export_checkpoint

    run_volume.reload()
    output = Path("/tmp/python-publication")
    shutil.rmtree(output, ignore_errors=True)
    progress = export_checkpoint(checkpoint, output, Path(checkpoint).parent, RUN)
    api = HfApi()
    try:
        api.repo_info(REPO)
        card = Path(hf_hub_download(REPO, "README.md")).read_text()
        if f"Run: `{RUN}`" not in card:
            raise ValueError("Refuse to overwrite an unrelated model repository")
    except RepositoryNotFoundError:
        api.create_repo(REPO, private=False, exist_ok=False)
    metrics, validation = [], []
    reports = []
    for folder in sorted(Path("/runs").glob(RUN + "-s*")):
        for name, rows in [("metrics.jsonl", metrics), ("validation.jsonl", validation)]:
            file = folder / name
            if file.exists():
                rows.extend(json.loads(line) for line in file.read_text().splitlines())
        if (folder / "report.json").exists():
            reports.append(json.loads((folder / "report.json").read_text()))
    for name, rows in [("metrics.jsonl", metrics), ("validation.jsonl", validation)]:
        (output / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    evaluations = output / "evaluations"
    evaluations.mkdir()
    for path in Path("/runs", RUN, "evaluations").glob("*-evaluated.json"):
        shutil.copyfile(path, evaluations / path.name)
    current = json.loads(Path(evaluation_path).read_text())
    (output / "evaluation.json").write_text(json.dumps(current, indent=2))
    shutil.copyfile(Path(SNAPSHOT) / "capacity.json", output / "capacity.json")
    for file in ["qualification.json", "execution-harness.json"]:
        shutil.copyfile(Path("/runs", RUN, file), output / file)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    tokens = [row["tokens"] for row in metrics]
    for axis, key, title in [
        (axes[0, 0], "loss", "Training loss"),
        (axes[0, 1], "gradient_norm", "Gradient norm"),
        (axes[1, 0], "lr", "Auxiliary learning rate"),
        (axes[1, 1], "nonpadding_tokens_per_second", "Nonpadding targets / second"),
    ]:
        axis.plot(tokens, [row[key] for row in metrics], linewidth=0.7)
        axis.set_title(title)
        axis.set_xlabel("Training targets")
    figure.tight_layout()
    figure.savefig(output / "training-curves.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 5))
    for category in (
        current.get("per_source", {})
        or json.loads((output / "data-manifest.json").read_text())["sources"]
    ):
        rows = [row for row in validation if category in row.get("per_source", {})]
        axis.plot(
            [r["tokens"] for r in rows], [r["per_source"][category] for r in rows], label=category
        )
    axis.set_xlabel("Training targets")
    axis.set_ylabel("Held-out loss (nats / target)")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "validation-curves.png", dpi=160)
    plt.close(figure)
    estimate = {
        "kind": "GPU compute estimate, not measured invoice",
        "training_worker_reported_seconds": sum(
            r.get("worker_elapsed_seconds", r["elapsed_seconds"]) for r in reports
        ),
        "h100_usd_per_second": 0.001097,
        "excludes": [
            "GPU work outside recorded training segments",
            "CPU preparation and evaluation",
            "storage and transfer",
        ],
    }
    estimate["training_gpu_usd_estimate"] = (
        estimate["training_worker_reported_seconds"] * 8 * estimate["h100_usd_per_second"]
    )
    (output / "runtime-cost-estimate.json").write_text(json.dumps(estimate, indent=2))
    status = "complete" if progress["tokens"] == 5_000_000_000 else "intermediate"
    card = f"""---
language: en
library_name: pytorch
tags:
- causal-lm
- python
- from-scratch
- muon
- experimental
---
# Simple LLM Python 124M / 5B

Run: `{RUN}`. This is an **{status}** base-model checkpoint, freshly initialized.
Processed **{progress["tokens"]:,} nonpadding training targets** in **{progress["step"]:,} optimizer updates**.
The exact target is 5,000,000,000. This model is not instruction-tuned.

124,389,632 parameters; 14 layers, width 768, 12 attention heads, RMSNorm,
RoPE, SwiGLU, tied embeddings and the existing frozen 32K tokenizer. Training
uses eight H100s, 8K context, cropped padding, cached RoPE, torch.compile,
microbatch 2 and global batch 32 document-local windows. Hidden matrices use
the previously tested five-step Muon (LR .02, momentum .95); tied embeddings,
norms and biases use auxiliary AdamW (LR .0003, betas .9/.95). Weight decay .1,
gradient clipping 1, 20M-target warmup and cosine decay to 10%.

## Data
Target proportions: 65% Python implementation, 15% Python tests/examples,
10% Python programming documentation (including explanatory technical Q&A),
5% CS/math, 3% general educational prose, 2% licensed/public-domain books.
Complete recent collected repository snapshots are preferred. Supplemental
sources include pinned CodeParrot pools, Common Pile StackExchange, FineWeb-Edu,
FineMath and public-domain Gutenberg material. All code allocations are Python.
The capacity table and exact per-category targets are in `capacity.json` and
`data-manifest.json`; one pass is preferred and two is the maximum.

Repository/thread splits precede windows. Exact held-out hashes and repositories
are excluded across sources. Exact deduplication and repository caps are applied;
near clones can remain. Path/content filtering and license metadata are imperfect.
Some supplements are older and may contain outdated practices or errors.
Private documents and sensitive source-level provenance are not published.
StackExchange sources carry per-document CC-BY-SA attribution in the private
provenance ledger; Common Pile source revision is
`5ec3aa2e65b7c61b7c18f1ff18ccbc4d3279a23c`.

## Evaluation
Held-out category losses, all fixed greedy/sampled Python generations, syntax,
repetition, length/EOS metrics and hidden-test execution results are included.
Function evaluation uses 32 reproducibly selected MBPP sanitized test tasks
(CC-BY-4.0, Google Research), revision
`4bb6404fdc6cacfda99d4ac4205087b89d32030c`. Tests are omitted from model prompts.
Exact reference hashes are excluded from training; no exhaustive benchmark
contamination audit or benchmark-clean claim is made. Generated code executes
in isolated network-blocked, credential-free sandboxes with resource limits.
Low early accuracy alone does not stop training.

![Training curves](training-curves.png)
![Validation curves](validation-curves.png)

All fixed-suite generations are retained without selection in `evaluations/`.
Each Hub commit preserves an inference checkpoint. Optimizer, RNG and data cursors
remain in private Modal training checkpoints for exact continuation.
Weight hashes are in `provenance.json`; runtime/cost figures are explicitly estimates.
Use the bundled custom PyTorch model, not Transformers AutoModel:

```bash
pip install -r requirements.txt
python generate.py --prompt 'def add(a, b):' --tokens 128
```

Code and execution evidence: https://github.com/koogle/lm-exploration/pull/30 .
"""
    (output / "README.md").write_text(card)
    commit = api.upload_folder(
        repo_id=REPO,
        folder_path=output,
        commit_message=f"{RUN}: {progress['tokens']} training targets",
    )
    remote = api.model_info(REPO, revision=commit.oid, files_metadata=True)
    weight = next(s for s in remote.siblings if s.rfilename == "model.safetensors")
    local_sha = digest(output / "model.safetensors")
    if weight.lfs is None or weight.lfs.sha256 != local_sha:
        raise ValueError("Uploaded weight hash does not match local export")
    if progress["tokens"] == 5_000_000_000:
        downloaded = hf_hub_download(
            REPO, "model.safetensors", revision=commit.oid, local_dir="/tmp/python-final-download"
        )
        if digest(downloaded) != local_sha:
            raise ValueError("Final downloaded weights failed hash verification")
    result = {
        **progress,
        "repo": REPO,
        "commit": commit.oid,
        "url": commit.commit_url,
        "weights_sha256": local_sha,
        "remote_lfs_hash_verified": True,
        "download_hash_verified": progress["tokens"] == 5_000_000_000,
    }
    Path("/runs", RUN, "publication.json").write_text(json.dumps(result, indent=2))
    run_volume.commit()
    return result


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def start(finalization_call_id: str, qualification_call_id: str):
    import math

    from training_data.python_experiment import TARGETS

    run_volume.reload()
    status_path = Path("/runs", RUN, "workflow.json")
    if status_path.exists():
        saved_status = json.loads(status_path.read_text())
        if saved_status.get("stage") == "complete":
            return saved_status
    record("waiting_for_preparation")
    try:
        prepared = modal.FunctionCall.from_id(finalization_call_id).get()
    except Exception as exc:
        record("preparation_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    if prepared.get("stage") != "frozen":
        return record("more_data_required", preparation=prepared)
    try:
        checked = modal.FunctionCall.from_id(qualification_call_id).get()
    except Exception as exc:
        record("qualification_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    if not checked.get("passed"):
        raise ValueError("Qualification did not pass")
    record("checking_isolated_evaluator")
    harness = Path("/runs", RUN, "execution-harness.json")
    if not harness.exists() or json.loads(harness.read_text()).get("execution_pass_rate") != 1.0:
        execution_checks.remote()
    run_volume.reload()
    reports = []
    for folder in sorted(Path("/runs").glob(RUN + "-s*")):
        path = folder / "report.json"
        if path.exists():
            reports.append(json.loads(path.read_text()))
    index = len(list(Path("/runs").glob(RUN + "-s*")))
    latest = reports[-1] if reports else None
    resume = latest["final_checkpoint"] if latest else ""
    tokens = latest["total_loss_tokens"] if latest else 0
    try:
        while len(reports) < 2:
            record("eight_gpu_qualification", segment=index, tokens=tokens)
            result = train_segment.remote(index, resume, max_steps=2)
            run_volume.reload()
            if result["session_steps"] != 2 or result["total_loss_tokens"] <= tokens:
                raise ValueError("Eight-GPU qualification did not produce optimizer updates")
            if latest:
                a, b = latest["final_validation"], result["initial_validation"]
                if a["tokens"] != b["tokens"] or any(
                    not math.isclose(
                        a["per_source"][k], b["per_source"][k], rel_tol=0, abs_tol=1e-6
                    )
                    for k in a["per_source"]
                ):
                    raise ValueError("Checkpoint restore changed held-out logits/loss")
            reports.append(result)
            latest = result
            resume, tokens = result["final_checkpoint"], result["total_loss_tokens"]
            index += 1
        record(
            "running",
            tokens=tokens,
            steps=latest["steps"],
            checkpoint=resume,
            eight_gpu_updates_and_restore="verified",
        )
        for milestone in [50_000_000, *range(250_000_000, 5_000_000_001, 250_000_000)]:
            if tokens >= milestone:
                continue
            while tokens < milestone:
                record(
                    "running",
                    tokens=tokens,
                    steps=latest["steps"],
                    checkpoint=resume,
                    next_evaluation_target=milestone,
                    segment=index,
                )
                result = train_segment.remote(index, resume, stop_tokens=milestone)
                run_volume.reload()
                if result["total_loss_tokens"] <= tokens or result["stop_reason"] in (
                    "data_error",
                    "source_two_pass_limit",
                ):
                    raise ValueError(
                        "Training did not advance with valid data: " + json.dumps(result)
                    )
                reports.append(result)
                latest = result
                resume, tokens = result["final_checkpoint"], result["total_loss_tokens"]
                index += 1
            record("evaluating", tokens=tokens, steps=latest["steps"], checkpoint=resume)
            generated = generations.remote(resume)
            evaluated = execution_checks.remote(generated)
            try:
                published = publish.remote(resume, evaluated)
            except ValueError:
                raise
            except Exception as exc:
                # Publication transport errors do not invalidate optimizer steps.
                record(
                    "publication_pending",
                    tokens=tokens,
                    checkpoint=resume,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if tokens == 5_000_000_000:
                    raise
            else:
                record(
                    "checkpoint_published", tokens=tokens, checkpoint=resume, publication=published
                )
        if tokens != 5_000_000_000 or latest["loss_token_exposures"] != TARGETS:
            raise ValueError("Final exact token/category budgets were not met")
        if "published" not in locals():
            generated = generations.remote(resume)
            evaluated = execution_checks.remote(generated)
            published = publish.remote(resume, evaluated)
        return record(
            "complete",
            tokens=tokens,
            steps=latest["steps"],
            checkpoint=resume,
            publication=published,
        )
    except Exception as exc:
        record("failed", tokens=tokens, checkpoint=resume, error=f"{type(exc).__name__}: {exc}")
        raise
