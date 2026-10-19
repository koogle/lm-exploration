"""Durable, CPU-only publication of every committed overnight inference checkpoint."""

import json
import re
import tempfile
import time
from pathlib import Path

import modal

from modal_overnight import image, run_volume

image = image.add_local_file("modal_overnight.py", "/project/modal_overnight.py")
app = modal.App("simple-llm-live-publication")
REPO = "jfrick/124M-10.879B"


def checked_checkpoint(run_id, checkpoint):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,60}", run_id):
        raise ValueError("Invalid run ID")
    if not re.fullmatch(re.escape(run_id) + r"(?:-s[0-9]{2}|-seed)/step-[0-9]{9}\.pt", checkpoint):
        raise ValueError("Expected a trusted checkpoint from this run")
    return Path("/runs") / checkpoint


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=2400,
    retries=2,
    max_containers=2,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": run_volume},
)
def publish_checkpoint(run_id, checkpoint):
    import shutil

    from huggingface_hub import HfApi
    from training_data.subword import digest

    from trainer.distributed_train import atomic_json
    from trainer.hub_checkpoint import export_checkpoint

    path = checked_checkpoint(run_id, checkpoint)
    api = HfApi()
    if not api.model_info(REPO).private:
        raise ValueError("Refusing public publication")
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        source = (
            Path("/runs") / (run_id + "-s00") if path.parent.name.endswith("-seed") else path.parent
        )
        export_checkpoint(path, output, source, run_id)
        provenance = json.loads((output / "provenance.json").read_text())
        lifetime = provenance.get("lineage", {}).get("parent_tokens", 0) + provenance["tokens"]
        title = f"124M - {lifetime / 1e9:.3f}B"
        (output / "README.md").write_text(
            f"---\nlibrary_name: pytorch\nmodel_name: {title}\n---\n# {title}\n\n"
            f"Intermediate Simple LLM base-model checkpoint, step {provenance['step']:,}. "
            f"{lifetime:,} lifetime targets; {provenance['tokens']:,} targets in this "
            "additional 10B continuation. The parent root model is preserved.\n\n"
            "8K context, eight H100s, compilation, cached RoPE, cropped padding and "
            "microbatch two; global batch32; constant LR3e-5 with preserved Adam moments. "
            "Code60% (fresh files, one pass), EU5%, other sources35% (up to two phase passes). "
            "Frozen stratified held-out evaluation; data-manifest.json records the "
            "operational gates and remaining contamination/deduplication limitations. "
            "Not instruction-tuned; output may be incorrect and repetitive.\n\n"
            "FP32 inference weights are round-trip verified. Full optimizer/RNG/cursors "
            "remain on Modal. Install requirements.txt and run "
            '`python generate.py --prompt "def add(a, b):" --tokens 64` from this folder.\n'
        )
        shutil.copyfile(
            "/project/publication/generate_checkpoint_archive.py", output / "generate.py"
        )
        prefix = f"continuations/{run_id}/checkpoints/{path.stem}"
        revision = api.upload_folder(
            repo_id=REPO,
            folder_path=str(output),
            path_in_repo=prefix,
            commit_message=f"Archive {run_id} step {provenance['step']}",
        )
        remote = api.model_info(REPO, revision=revision.oid, files_metadata=True)
        actual = next(
            f.lfs.sha256 for f in remote.siblings if f.rfilename == prefix + "/model.safetensors"
        )
        expected = digest(output / "model.safetensors")
        if actual != expected:
            raise ValueError("Remote checkpoint hash mismatch")
        entry = {
            "step": provenance["step"],
            "phase_tokens": provenance["tokens"],
            "lifetime_tokens": lifetime,
            "checkpoint": checkpoint,
            "repo_id": REPO,
            "path": prefix + "/model.safetensors",
            "revision": revision.oid,
            "sha256": expected,
            "weight_roundtrip_exact": True,
            "cpu_logits_roundtrip_exact": True,
            "remote_hash_verified": True,
        }
    sidecar = Path("/runs") / run_id / "hf-live-checkpoints" / (path.stem + ".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(sidecar, entry)
    run_volume.commit()
    return entry


@app.function(
    image=image,
    cpu=0.25,
    memory=2048,
    timeout=3600,
    retries=0,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": run_volume},
)
def watch(run_id):
    """Chain 45-minute CPU watchers while preparation/training proceeds elsewhere."""
    from huggingface_hub import HfApi

    from trainer.distributed_train import atomic_json

    checked_checkpoint(run_id, run_id + "-s00/step-000000002.pt")
    folder = Path("/runs") / run_id
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "hf-live-publication.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"repo_id": REPO, "run_id": run_id, "checkpoints": {}, "failures": {}}
    )
    api = HfApi()
    if state["repo_id"] != REPO or not api.model_info(REPO).private:
        raise ValueError("Private publication identity mismatch")

    def save():
        state.update(uploaded_count=len(state["checkpoints"]), updated_unix=time.time())
        atomic_json(state_path, state)
        run_volume.commit()

    def index():
        api.upload_file(
            repo_id=REPO,
            path_or_fileobj=json.dumps(
                {
                    "run_id": run_id,
                    "repo_id": REPO,
                    "target_new_tokens": 10_000_000_000,
                    "stage": state.get("stage"),
                    "uploaded_count": len(state["checkpoints"]),
                    "checkpoints": state["checkpoints"],
                },
                indent=2,
            ).encode(),
            path_in_repo=f"continuations/{run_id}/checkpoint-index.json",
            commit_message=f"Update {run_id} checkpoint inventory",
        )

    state["stage"] = "waiting_for_checkpoints"
    save()
    index()
    stop = time.monotonic() + 2700
    while time.monotonic() < stop:
        run_volume.reload()
        # Recover completed upload workers if a prior watcher ended before bookkeeping.
        for sidecar in (folder / "hf-live-checkpoints").glob("step-*.json"):
            state["checkpoints"][sidecar.stem] = json.loads(sidecar.read_text())
        pending = state.get("pending")
        if pending:
            try:
                entry = modal.FunctionCall.from_id(pending["call_id"]).get(timeout=0)
                state["checkpoints"][pending["step"]] = entry
                state.pop("pending", None)
                state.pop("last_error", None)
                save()
                index()
            except TimeoutError:
                pass
            except Exception as exc:  # noqa: BLE001 -- persist/retry upload failures separately
                step = pending["step"]
                state["failures"][step] = state["failures"].get(step, 0) + 1
                state["last_error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
                state.pop("pending", None)
                if state["failures"][step] >= 3:
                    state["stage"] = "publication_failed"
                    save()
                    return state
        candidates = sorted(
            Path("/runs").glob(run_id + "-s[0-9][0-9]/step-*.pt"), key=lambda p: p.name
        )
        seed = Path("/runs") / (run_id + "-seed/step-000000000.pt")
        source = Path("/runs") / (run_id + "-s00")
        if (
            seed.exists()
            and (source / "source/model_100m/model.py").exists()
            and (source / "data-manifest.json").exists()
        ):
            candidates.insert(0, seed)
        remaining = [p for p in candidates if p.stem not in state["checkpoints"]]
        if not state.get("pending") and remaining:
            path = remaining[0]
            checkpoint = str(path.relative_to("/runs"))
            call = publish_checkpoint.spawn(run_id, checkpoint)
            state.update(
                stage="publishing",
                pending={"step": path.stem, "call_id": call.object_id, "checkpoint": checkpoint},
            )
        workflow = (
            json.loads((folder / "workflow.json").read_text())
            if (folder / "workflow.json").exists()
            else {}
        )
        terminal = workflow.get("stage") in [
            "training_completed",
            "completed",
            "failed",
            "data_gate_failed",
        ]
        if terminal and not remaining and not state.get("pending"):
            state["stage"] = (
                "complete"
                if workflow["stage"] in ["training_completed", "completed"]
                else "training_stopped"
            )
            save()
            index()
            return state
        save()
        time.sleep(30)
    next_call = watch.spawn(run_id)
    state["continuation_call_id"] = next_call.object_id
    save()
    return {
        "continuation_call_id": next_call.object_id,
        "uploaded_count": len(state["checkpoints"]),
    }
