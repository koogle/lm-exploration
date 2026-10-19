"""Archive every saved code-heavy phase checkpoint on private Hugging Face."""

import json
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
