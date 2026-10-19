"""Archive every saved code-heavy phase checkpoint on private Hugging Face."""

import json
import os
import shutil
import time
from pathlib import Path

import modal

from modal_simple_llm_8k import image, run_volume

image = image.add_local_file("modal_simple_llm_8k.py", "/project/modal_simple_llm_8k.py")
app = modal.App("simple-llm-code-heavy-publication")
RUN = "simple-llm-code60-fresh-5b-20261019-r02"
REPO = "jfrick/124M-10.879B"
FINAL = f"{RUN}-s02/step-000066065.pt"


def model_card(count):
    return f"""---
language: en
library_name: pytorch
model_name: 124M - 10.879B
tags:
- causal-lm
- from-scratch
- custom-code
- experimental
---
# 124M - 10.879B

Simple LLM, 8K context.

Private inference archive of **all {count} saved checkpoints** from the additional
5B-target, 60%-coding continuation, including the phase seed and qualification
steps. The root weights are the completed **step 66,065** model: 4,999,971,081 phase
next-token targets and 10,878,562,619 lifetime targets. This is a base model, not an
instruction-tuned assistant; generated text can be inaccurate and repetitive.

## Model and training

124,389,632 trainable parameters; 14 layers, width 768, 12 attention heads,
RMSNorm, RoPE (theta 10,000), SwiGLU, and tied 32,768-token embeddings. Configured
context 8,192. PyTorch 2.6, eight H100s, BF16 training with FP32 weights, cropped
padding, cached RoPE, compilation and microbatch two; global batch 32 windows.
AdamW moments were preserved from the EU parent; constant LR 3e-5 for this phase.

The parent had 4,999,981,176 original-mixture targets plus 878,610,362 EU targets.
This phase used 2,629,230,692 fresh unique eligible targets, with no exact-hash
training overlap against either previous snapshot, and at most two passes per
source. Coding supplied 2,999,962,024 targets (~60%). Exhausted book shares were
reallocated only within the original book collections, preserving all category
budgets and existing filters. Exact final source weights are in the data manifest.
Split, near-duplicate, eligibility and contamination audits remain incomplete.

## Validation

Matched expanded validation: 7,620,187 targets across the 12 original sources plus
ECB and EMA, using the existing 2,782-family-selection sample. Coding loss is
1.421 versus 1.559 for the original 5B model and 3.994 for the EU parent. Other
original-source losses largely recover from EU forgetting, but remain slightly
worse than the original 5B model. ECB/EMA losses are 2.494/2.379 versus the EU
parent's 2.065/1.766: most EU specialization was lost. See validation-summary.json.
These are 2K held-out window loss measurements, not proof of generation quality,
factual accuracy, instruction following or full-8K effectiveness.

## Checkpoints and use

Each `checkpoints/step-NNNNNNNNN/` contains FP32 `model.safetensors` and per-step
provenance. `checkpoint-index.json` lists every saved phase checkpoint, phase and
lifetime counters, weight hashes and upload progress. The root files always load
the final model. All safetensors exports are verified against the original weights
and CPU logits. Both tied state-dict keys are stored; model construction restores
the shared parameter. The model is custom PyTorch, not Transformers AutoModel.

Download the repository, then:

```bash
pip install -r requirements.txt
python generate.py --prompt 'def add(a, b):' --tokens 128 --temperature 0.8 --top-k 50
python generate.py --weights checkpoints/step-000001000/model.safetensors --prompt 'Inflation means'
```

These are **inference checkpoints**. Optimizer state, per-rank RNG and exact data
cursors remain on the private Modal training volume; this repository does not
replace those exact-resume files. Private training text and evaluation windows
are not uploaded. The repository remains private.

Source: https://github.com/koogle/lm-exploration/pull/26 . Run: `{RUN}`.
"""


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=86400,
    retries=0,
    max_containers=1,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": run_volume},
)
def publish_all():
    from huggingface_hub import HfApi

    from trainer.distributed_train import atomic_json
    from trainer.hub_checkpoint import export_checkpoint

    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(REPO).private:
        raise ValueError("Checkpoint archive must remain private")
    folder = Path("/runs") / RUN
    state_path = folder / "huggingface-publication.json"
    inventory = sorted(
        Path("/runs").glob(RUN + "-s*/step-*.pt"), key=lambda p: int(p.stem.split("-")[1])
    )
    if not inventory or not any(str(p.relative_to("/runs")) == FINAL for p in inventory):
        raise ValueError("Final checkpoint or archive inventory missing")
    if len({p.name for p in inventory}) != len(inventory):
        raise ValueError("Ambiguous duplicate checkpoint steps")
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "repo_id": REPO,
            "private": True,
            "expected_count": len(inventory),
            "checkpoints": {},
            "completed": False,
        }
    )
    if state["repo_id"] != REPO or state["expected_count"] != len(inventory):
        raise ValueError("Publication inventory changed")
    final_path = Path("/runs") / FINAL
    order = [final_path, *[p for p in inventory if p != final_path]]
    output = Path("/tmp/code-heavy-hf-export")
    try:
        for checkpoint in order:
            key = checkpoint.stem
            if key in state["checkpoints"]:
                continue
            state.update(stage="exporting", current_checkpoint=key, updated_unix=time.time())
            atomic_json(state_path, state)
            run_volume.commit()
            shutil.rmtree(output, ignore_errors=True)
            source = checkpoint.parent
            if not (source / "source").exists():
                source = final_path.parent  # Phase seed has the same verified model definition.
            progress = export_checkpoint(checkpoint, output, source, RUN)
            provenance = json.loads((output / "provenance.json").read_text())
            lifetime = provenance.get("lineage", {}).get("parent_tokens", 0) + progress["tokens"]
            entry = {
                **progress,
                "lifetime_tokens": lifetime,
                "weights_sha256": provenance["weights_sha256"],
                "path": f"checkpoints/{key}/model.safetensors",
                "weight_roundtrip_exact": True,
                "cpu_logits_roundtrip_exact": True,
            }
            if checkpoint == final_path and not state.get("root_revision"):
                shutil.copyfile(
                    "/project/publication/generate_checkpoint_archive.py", output / "generate.py"
                )
                (output / "README.md").write_text(model_card(len(inventory)))
                validation = folder / "expanded-validation.json"
                if not validation.exists():
                    raise ValueError("Final expanded validation missing")
                shutil.copyfile(validation, output / "validation-summary.json")
                index = {
                    "expected_count": len(inventory),
                    "uploaded_count": len(state["checkpoints"]),
                    "final_step": 66065,
                    "checkpoints": state["checkpoints"],
                }
                (output / "checkpoint-index.json").write_text(json.dumps(index, indent=2) + "\n")
                root = api.upload_folder(
                    repo_id=REPO,
                    folder_path=str(output),
                    commit_message="Publish completed Simple LLM 124M, 10.879B lifetime tokens",
                )
                state["root_revision"] = root.oid
                print(
                    json.dumps(
                        {"stage": "final_model_available", "repo_id": REPO, "revision": root.oid}
                    ),
                    flush=True,
                )
            archive = output / "archive"
            archive.mkdir()
            shutil.move(output / "model.safetensors", archive / "model.safetensors")
            shutil.move(output / "provenance.json", archive / "provenance.json")
            state.update(stage="uploading", updated_unix=time.time())
            atomic_json(state_path, state)
            run_volume.commit()
            # Retry transient transfer failures without restarting the export.
            for attempt in range(4):
                try:
                    revision = api.upload_folder(
                        repo_id=REPO,
                        folder_path=str(archive),
                        path_in_repo=f"checkpoints/{key}",
                        commit_message=f"Archive {key}: {progress['tokens']:,} phase targets",
                    )
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
            state["checkpoints"][key] = {**entry, "revision": revision.oid}
            state.update(
                stage="archiving",
                uploaded_count=len(state["checkpoints"]),
                updated_unix=time.time(),
            )
            atomic_json(state_path, state)
            run_volume.commit()
            print(
                json.dumps(
                    {
                        "stage": "checkpoint_archived",
                        "step": progress["step"],
                        "uploaded_count": len(state["checkpoints"]),
                        "expected_count": len(inventory),
                    }
                ),
                flush=True,
            )
        index = {
            "expected_count": len(inventory),
            "uploaded_count": len(state["checkpoints"]),
            "final_step": 66065,
            "checkpoints": state["checkpoints"],
        }
        index_path = output / "checkpoint-index.json"
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        api.upload_file(
            repo_id=REPO,
            path_or_fileobj=str(index_path),
            path_in_repo="checkpoint-index.json",
            commit_message="Complete saved-checkpoint inventory",
        )
        remote = api.model_info(REPO, files_metadata=True)
        files = {f.rfilename: f for f in remote.siblings}
        for entry in state["checkpoints"].values():
            item = files.get(entry["path"])
            if item is None or item.lfs is None or item.lfs.sha256 != entry["weights_sha256"]:
                raise ValueError("Remote checkpoint checksum mismatch")
        root = files.get("model.safetensors")
        if (
            root is None
            or root.lfs.sha256 != state["checkpoints"][final_path.stem]["weights_sha256"]
        ):
            raise ValueError("Default weights are not the final checkpoint")
        if not remote.private:
            raise ValueError("Archive privacy changed")
        state.update(
            stage="completed", completed=True, updated_unix=time.time(), verified_remote_hashes=True
        )
        atomic_json(state_path, state)
        run_volume.commit()
        return {
            "completed": True,
            "repo_id": REPO,
            "private": True,
            "uploaded_count": len(state["checkpoints"]),
            "verified_remote_hashes": True,
            "url": "https://huggingface.co/" + REPO,
        }
    except Exception as exc:
        state.update(
            stage="failed",
            error=f"{type(exc).__name__}: {str(exc)[:1000]}",
            updated_unix=time.time(),
        )
        atomic_json(state_path, state)
        run_volume.commit()
        raise


def verify_inventory(api, state):
    """Compare remote LFS hashes; never infer upload success from local counters."""
    remote = api.model_info(REPO, files_metadata=True)
    files = {f.rfilename: f for f in remote.siblings}
    for entry in state["checkpoints"].values():
        item = files.get(entry["path"])
        if item is None or item.lfs is None or item.lfs.sha256 != entry["weights_sha256"]:
            raise ValueError("Remote checkpoint checksum mismatch: " + entry["path"])
    root = files.get("model.safetensors")
    if root is None or root.lfs.sha256 != state["checkpoints"]["step-000066065"]["weights_sha256"]:
        raise ValueError("Default weights are not the final checkpoint")
    if not remote.private:
        raise ValueError("Archive privacy changed")


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3600,
    retries=0,
    max_containers=4,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": run_volume},
)
def archive_checkpoint(relative_path):
    import re

    from huggingface_hub import HfApi

    from trainer.distributed_train import atomic_json
    from trainer.hub_checkpoint import export_checkpoint

    if not re.fullmatch(RUN + r"-(?:s[0-9]{2}|seed)/step-[0-9]{9}\.pt", relative_path):
        raise ValueError("Checkpoint outside authorized run")
    checkpoint = Path("/runs") / relative_path
    sidecar = Path("/runs") / RUN / "huggingface-checkpoints" / (checkpoint.stem + ".json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    output = Path("/tmp/parallel-hf-export")
    shutil.rmtree(output, ignore_errors=True)
    source = (
        checkpoint.parent
        if (checkpoint.parent / "source").exists()
        else (Path("/runs") / FINAL).parent
    )
    progress = export_checkpoint(checkpoint, output, source, RUN)
    provenance = json.loads((output / "provenance.json").read_text())
    archive = output / "archive"
    archive.mkdir()
    shutil.move(output / "model.safetensors", archive / "model.safetensors")
    shutil.move(output / "provenance.json", archive / "provenance.json")
    api = HfApi()
    for attempt in range(6):
        try:
            revision = api.upload_folder(
                repo_id=REPO,
                folder_path=str(archive),
                path_in_repo=f"checkpoints/{checkpoint.stem}",
                commit_message=f"Archive {checkpoint.stem}: {progress['tokens']:,} phase targets",
            )
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 30))
    entry = {
        **progress,
        "lifetime_tokens": provenance.get("lineage", {}).get("parent_tokens", 0)
        + progress["tokens"],
        "weights_sha256": provenance["weights_sha256"],
        "path": f"checkpoints/{checkpoint.stem}/model.safetensors",
        "revision": revision.oid,
        "weight_roundtrip_exact": True,
        "cpu_logits_roundtrip_exact": True,
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(sidecar, entry)
    run_volume.commit()
    return entry


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=14400,
    retries=0,
    max_containers=1,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": run_volume},
)
def publish_remaining():
    """Resume the serial publisher with four independent bounded upload workers."""
    from huggingface_hub import HfApi

    from trainer.distributed_train import atomic_json

    api = HfApi()
    folder = Path("/runs") / RUN
    state_path = folder / "huggingface-publication.json"
    state = json.loads(state_path.read_text())
    if (
        state["repo_id"] != REPO
        or not state.get("root_revision")
        or not api.model_info(REPO).private
    ):
        raise ValueError("Expected the verified private final-model publication")
    inventory = sorted(Path("/runs").glob(RUN + "-s*/step-*.pt"), key=lambda p: p.name)
    if len(inventory) != state["expected_count"]:
        raise ValueError("Checkpoint inventory changed")
    # Recover uploads whose transfer completed before local bookkeeping failed.
    from huggingface_hub import hf_hub_download

    remote = api.model_info(REPO, files_metadata=True)
    remote_files = {f.rfilename: f for f in remote.siblings}
    for checkpoint in inventory:
        if checkpoint.stem in state["checkpoints"]:
            continue
        prefix = f"checkpoints/{checkpoint.stem}"
        item = remote_files.get(prefix + "/model.safetensors")
        if item is None or prefix + "/provenance.json" not in remote_files:
            continue
        provenance = json.loads(
            Path(
                hf_hub_download(REPO, filename=prefix + "/provenance.json", revision=remote.sha)
            ).read_text()
        )
        if item.lfs is None or item.lfs.sha256 != provenance["weights_sha256"]:
            raise ValueError("Remote recovered checkpoint hash mismatch")
        if not provenance.get("weight_roundtrip_exact") or not provenance.get(
            "cpu_logits_roundtrip_exact"
        ):
            raise ValueError("Recovered export was not verified")
        state["checkpoints"][checkpoint.stem] = {
            "step": provenance["step"],
            "tokens": provenance["tokens"],
            "checkpoint": str(checkpoint),
            "lifetime_tokens": provenance.get("lineage", {}).get("parent_tokens", 0)
            + provenance["tokens"],
            "weights_sha256": provenance["weights_sha256"],
            "path": prefix + "/model.safetensors",
            "revision": remote.sha,
            "weight_roundtrip_exact": True,
            "cpu_logits_roundtrip_exact": True,
        }
    state["uploaded_count"] = len(state["checkpoints"])
    calls = {
        p.stem: archive_checkpoint.spawn(str(p.relative_to("/runs")))
        for p in inventory
        if p.stem not in state["checkpoints"]
    }
    state.update(stage="archiving", parallel_workers=4, updated_unix=time.time())
    state.pop("error", None)
    atomic_json(state_path, state)
    run_volume.commit()
    try:
        for name, call in calls.items():
            state["checkpoints"][name] = call.get()
            state.update(
                uploaded_count=len(state["checkpoints"]),
                current_checkpoint=name,
                updated_unix=time.time(),
            )
            atomic_json(state_path, state)
            run_volume.commit()
            print(
                json.dumps(
                    {"uploaded_count": len(state["checkpoints"]), "expected_count": len(inventory)}
                ),
                flush=True,
            )
        index = {
            "expected_count": len(inventory),
            "uploaded_count": len(state["checkpoints"]),
            "final_step": 66065,
            "checkpoints": state["checkpoints"],
        }
        index_path = Path("/tmp/checkpoint-index.json")
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        api.upload_file(
            repo_id=REPO,
            path_or_fileobj=str(index_path),
            path_in_repo="checkpoint-index.json",
            commit_message="Complete all saved checkpoints and verify their hashes",
        )
        verify_inventory(api, state)
        state.update(
            stage="completed", completed=True, verified_remote_hashes=True, updated_unix=time.time()
        )
        atomic_json(state_path, state)
        run_volume.commit()
        return {
            "completed": True,
            "repo_id": REPO,
            "private": True,
            "uploaded_count": len(state["checkpoints"]),
            "verified_remote_hashes": True,
            "url": "https://huggingface.co/" + REPO,
        }
    except Exception as exc:
        state.update(
            stage="failed",
            error=f"{type(exc).__name__}: {str(exc)[:1000]}",
            updated_unix=time.time(),
        )
        atomic_json(state_path, state)
        run_volume.commit()
        raise


@app.function(
    image=image,
    cpu=4,
    memory=12288,
    timeout=1800,
    retries=0,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": run_volume},
)
def verify_download():
    import subprocess

    from huggingface_hub import HfApi, snapshot_download
    from training_data.subword import digest

    from trainer.distributed_train import atomic_json

    api = HfApi()
    info = api.model_info(REPO, files_metadata=True)
    bundle = Path(
        snapshot_download(
            REPO,
            revision=info.sha,
            allow_patterns=[
                "model.safetensors",
                "config.json",
                "model_100m/*.py",
                "model_10m/*.py",
                "tokenizer/*",
                "generate.py",
                "requirements.txt",
            ],
        )
    )
    expected = next(x.lfs.sha256 for x in info.siblings if x.rfilename == "model.safetensors")
    if digest(bundle / "model.safetensors") != expected:
        raise ValueError("Downloaded weights checksum mismatch")
    result = subprocess.run(
        [
            "python",
            str(bundle / "generate.py"),
            "--prompt",
            "def add(a, b):\n    ",
            "--tokens",
            "16",
            "--temperature",
            "0",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=bundle,
        env={**os.environ, "PYTHONPATH": str(bundle)},
    )
    if result.returncode:
        raise RuntimeError("Downloaded generation CLI failed: " + result.stderr[-2000:])
    report = {
        "completed": True,
        "repo_id": REPO,
        "revision": info.sha,
        "weights_sha256": expected,
        "download_hash_verified": True,
        "standalone_cli_verified": True,
        "smoke_generation": result.stdout.strip(),
        "updated_unix": time.time(),
    }
    atomic_json(Path("/runs") / RUN / "huggingface-download-verification.json", report)
    run_volume.commit()
    return report


@app.function(
    image=image,
    cpu=1,
    memory=4096,
    timeout=900,
    retries=0,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": run_volume},
)
def publish_evaluation():
    import statistics

    from huggingface_hub import HfApi

    from trainer.distributed_train import atomic_json

    api = HfApi()
    if not api.model_info(REPO).private:
        raise ValueError("Generation artifacts require the private model repository")
    output = Path("/tmp/evaluation-upload")
    shutil.rmtree(output, ignore_errors=True)
    (output / "evaluation/generation").mkdir(parents=True)
    root = Path("/runs/simple-llm-code-heavy-generation-20261019")
    models = {}
    sample_report = [
        "# All matched generated samples",
        "",
        "Raw, unverified model continuations, including failures. No generation is a factual claim.",
        "",
    ]
    for label in ["original_5b", "eu_parent", "final_code60"]:
        path = root / (label + ".json")
        data = json.loads(path.read_text())
        if not data.get("completed") or len(data["samples"]) != 50:
            raise ValueError("Incomplete generation comparison")
        shutil.copyfile(path, output / "evaluation/generation" / path.name)
        models[label] = {}
        for mode in data["modes"]:
            samples = [r for r in data["samples"] if r["mode"] == mode]
            models[label][mode] = {
                "samples": len(samples),
                "mean_repeated_token_4gram_fraction": statistics.mean(
                    r["repeated_token_4gram_fraction"] for r in samples
                ),
                "long_loop_samples": sum(r["long_loop"] for r in samples),
            }
        for row in data["samples"]:
            sample_report += [
                f"## {label}: {row['mode']}, seed {row['seed']}, prompt {row['prompt_index']}",
                "",
                "Prompt: " + json.dumps(row["prompt"]),
                "",
                "````text",
                row["text"],
                "````",
                "",
            ]
    (output / "evaluation/all-samples.md").write_text("\n".join(sample_report) + "\n")
    (output / "evaluation/generation-summary.json").write_text(json.dumps(models, indent=2) + "\n")
    state = json.loads((Path("/runs") / RUN / "huggingface-publication.json").read_text())
    text = model_card(state["expected_count"]) + "\n## Matched generation and repetition\n\n"
    text += "150 continuations: ten fixed prompts, 256-token cap; one greedy seed, two seeds for each sampled decoder. FP32, TF32 off; no repetition penalty or n-gram blocking.\n\n"
    text += (
        "Mean repeated generated token four-gram fraction (lower means less exact repetition):\n\n"
    )
    text += "| Decoder | Original 5B | EU parent | Final |\n|---|---:|---:|---:|\n"
    for mode in ["greedy", "low_temperature_topk", "sampled_topk"]:
        text += (
            "| "
            + mode
            + " | "
            + " | ".join(
                f"{models[label][mode]['mean_repeated_token_4gram_fraction']:.1%}"
                for label in models
            )
            + " |\n"
        )
    text += "\nGreedy and T=0.2/k=10 repetition remains severe. T=0.8/k=50 reduces exact repetition in this small matched set, but factual and code outputs are still unreliable. A zero exact-loop flag does not exclude semantic or near repetition. This is not a general generation-quality benchmark.\n\n"
    text += "[All 150 samples, including failures](evaluation/all-samples.md) · [Raw samples and token IDs](evaluation/generation) · [Summary](evaluation/generation-summary.json)\n"
    (output / "README.md").write_text(text)
    shutil.copyfile("/project/publication/generate_checkpoint_archive.py", output / "generate.py")
    revision = api.upload_folder(
        repo_id=REPO,
        folder_path=str(output),
        commit_message="Add all matched generation samples and repetition results",
    )
    result = {
        "completed": True,
        "repo_id": REPO,
        "revision": revision.oid,
        "samples": 150,
        "private": True,
        "updated_unix": time.time(),
    }
    atomic_json(Path("/runs") / RUN / "huggingface-evaluation-publication.json", result)
    run_volume.commit()
    return result
