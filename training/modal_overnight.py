"""Additional 10B continuation, only after 6B fresh-code capacity/variety checks."""

import json
import time
from fractions import Fraction
from pathlib import Path

import modal

from distributed_training.modal_train import validate_paths, worker
from modal_eu_continuation import data_volume, run_volume

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", "numpy", "pydantic>=2.11,<3", "tokenizers==0.22.2")
    .pip_install("huggingface_hub==0.36.0", "pyarrow==19.0.1")
    .pip_install("matplotlib==3.10.8", "safetensors", "pyahocorasick==2.1.0")
    .env(
        {
            "PYTHONPATH": "/project",
            "TOKENIZERS_PARALLELISM": "true",
            "RAYON_NUM_THREADS": "8",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    .add_local_dir("100m", "/project/model_100m")
    .add_local_dir("10m", "/project/model_10m")
    .add_local_dir("trainer", "/project/trainer")
    .add_local_dir("distributed_training", "/project/distributed_training")
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("tokenizer", "/project/tokenizer")
    .add_local_dir("tests", "/project/tests")
    .add_local_dir("publication", "/project/publication")
    .add_local_file("generate.py", "/project/generate.py")
)
image = image.add_local_file(
    "modal_eu_continuation.py", "/project/modal_eu_continuation.py"
).add_local_dir("analysis/loss-curves", "/project/loss_curves")
app = modal.App("simple-llm-overnight-10b")
VERSION = "readiness-20261020-v2"
BASE = Path("/data/scholarly-code") / VERSION
PRIOR = Path("/data/scholarly-code/code-heavy-fresh-20261019/staging/raw.sqlite")
EU = Path("/data/scholarly-code/eu-continuation-20261018/snapshots/ecb-ema-one-pass-v1/raw.sqlite")
OLD_MANIFEST = Path(
    "/data/scholarly-code/code-heavy-fresh-20261019/snapshots/code60-5b-books-v2/manifest.json"
)
PARENT = "simple-llm-code60-fresh-5b-20261019-r02-s02/step-000066065.pt"
SNAPSHOT = VERSION + "/snapshots/code60-onepass-10b"
TARGET = 10_000_000_000
SETTINGS = {
    "optimizer": "adamw",
    "world_size": 8,
    "global_windows": 32,
    "microbatch": 2,
    "context": 8192,
    "warmup_tokens": 1,
    "peak_lr": 3e-5,
    "min_lr_ratio": 1.0,
    "checkpoint_every": 2000,
    "evaluate_every": 2000,
    "allow_unreviewed_data": True,
    "prefetch_batches": 4,
    "crop_padding": True,
    "cache_rope": True,
    "compile_model": True,
    "loss_chunk_size": 0,
    "n_layer": 14,
    "n_head": 12,
    "n_embd": 768,
}


def mark(run_id, stage, **details):
    from distributed_training.train import atomic_json

    folder = Path("/runs") / run_id
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "workflow.json", {"stage": stage, "updated_unix": time.time(), **details})
    run_volume.commit()


@app.function(
    image=image,
    cpu=16,
    memory=32768,
    timeout=10800,
    retries=0,
    max_containers=1,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/data": data_volume},
)
def prepare_original():
    from training_data.cached_shards import CachedShards
    from training_data.fresh_raw_corpus import FreshRawCorpus
    from training_data.original_refresh import fork_original
    from training_data.overnight_snapshot import weights_from

    from distributed_training.train import atomic_json
    from trainer.continued_pretraining import token_targets

    class Refresh(FreshRawCorpus):
        def report(self, target):
            result = super().report(target)
            for source in result["sources"].values():
                if source["upstream_exhausted"]:
                    source.update(minimum_unique_train_tokens=0, capacity_sufficient=True)
            result["ready"] = all(self.enough(s) for s in result["sources"].values())
            return result

    folder = BASE / "original-refresh"
    recipe = fork_original(folder, PRIOR, EU)
    all_targets = token_targets(weights_from(OLD_MANIFEST), TARGET)
    targets = {
        s["name"]: 0 if s["name"] == "code_python" else all_targets[s["name"]]
        for s in recipe["sources"]
    }
    target = sum(targets.values())
    weights = {n: Fraction(v * 100, target) for n, v in targets.items()}
    corpus = Refresh(
        folder, recipe, weights, targets, workers=6, batch_size=128, reader=CachedShards()
    )
    started = time.monotonic()
    try:
        while True:
            result = corpus.prepare_parallel(target, max_seconds=20)
            progress = corpus.report(target)
            atomic_json(folder / "progress.json", progress)
            data_volume.commit()
            print(
                json.dumps(
                    {
                        "stage": "fresh_original",
                        "fresh_tokens": sum(
                            s["train_loss_tokens"] for s in progress["sources"].values()
                        ),
                    }
                ),
                flush=True,
            )
            if result["ready"] or time.monotonic() - started >= 6000:
                return progress
    finally:
        corpus.close()
        data_volume.commit()


@app.function(
    image=image,
    cpu=16,
    memory=32768,
    timeout=86400,
    retries=0,
    max_containers=1,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/data": data_volume},
)
def audit_augmented(audit_round=0):
    from training_data.readiness_audit import audit

    prepare_patches.remote()
    data_volume.reload()
    folder = BASE / f"combined-refresh-audit-r{audit_round}"
    return audit(
        [
            ("prior", PRIOR, False),
            ("eu", EU, False),
            ("fresh_code", BASE / "multilingual/raw.sqlite", True),
            ("fresh_original", BASE / "original-refresh/raw.sqlite", True),
            ("fresh_patches", BASE / "patches/raw.sqlite", True),
        ],
        folder,
        set(),
        [],
        commit=data_volume.commit,
        benchmark_screen=False,
        tokenizer_diagnostics=False,
        resume=True,
    )


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=21600,
    retries=2,
    max_containers=1,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/data": data_volume},
)
def prepare_patches():
    from training_data.hf_patches import prepare

    data_volume.reload()
    return prepare(
        BASE / "patches",
        [
            {
                "repo": "jfrick/mit-claude-codex-code-20261019",
                "revision": "b27f9cef49fa44643247d8a97173201fd47896c8",
            },
            {
                "repo": "jfrick/curated-oss-code-20261019",
                "revision": "1bda7edae2fac3b6d646422377817ee53893bc3c",
            },
        ],
        BASE / "parallel-code/exclusions.sqlite",
        commit=data_volume.commit,
    )


@app.function(
    image=image,
    cpu=16,
    memory=32768,
    timeout=86400,
    retries=0,
    max_containers=1,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/data": data_volume},
)
def collect_more_code(audit_round):
    return modal.Function.from_name("simple-llm-parallel-code", "collect").remote(audit_round)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=21600,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume},
)
def freeze_snapshot(audit_round=0):
    from training_data.overnight_snapshot import freeze

    data_volume.reload()
    result = freeze(
        BASE / f"combined-refresh-audit-r{audit_round}",
        Path("/data/scholarly-code") / SNAPSHOT,
        OLD_MANIFEST,
        {
            "prior": PRIOR,
            "eu": EU,
            "fresh_code": BASE / "multilingual/raw.sqlite",
            "fresh_original": BASE / "original-refresh/raw.sqlite",
            "fresh_patches": BASE / "patches/raw.sqlite",
        },
    )
    data_volume.commit()
    return {
        "snapshot": SNAPSHOT,
        "sources": result["sources"],
        "code_languages": result["code_languages"],
        "code_path_tasks": result["code_path_tasks"],
        "heldout": result["heldout"],
    }


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=7200,
    retries=0,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def start_prepared_run(run_id, provenance):
    """User-authorized experimental launch without the remaining corpus audit."""
    from training_data.prepared_reference import build
    from training_data.raw_token_stream import RawTokenMixture

    from trainer.representative_validation import load_heldout

    data_volume.reload()
    mark(run_id, "opening_prepared_data", global_audit_bypassed=True)
    destination = Path("/data/scholarly-code") / SNAPSHOT
    prepared = build(
        destination,
        OLD_MANIFEST,
        BASE / "original-refresh/raw.sqlite",
        BASE / "multilingual/raw.sqlite",
        BASE / "patches/raw.sqlite",
        EU,
        BASE / "multilingual/progress.json",
        BASE / "patches/report.json",
    )
    data = RawTokenMixture(destination / "manifest.json", context=8192, allow_unreviewed=True)
    validation = load_heldout(destination / "manifest.json", data)
    for name in prepared["sources"]:
        next(data.blocks(name))
    data.db.close()
    data_volume.commit()
    training = train.spawn(run_id, provenance)
    mark(
        run_id,
        "training_scheduled",
        training_call_id=training.object_id,
        readiness_mode=prepared["readiness_mode"],
        validation_windows=len(validation),
    )
    return {"training_call_id": training.object_id, "validation_windows": len(validation)}


@app.function(
    image=image,
    cpu=4,
    memory=12288,
    timeout=1800,
    retries=0,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def initialize(run_id):
    import torch

    from distributed_training.config import DistributedConfig
    from distributed_training.run_validation import validate_overnight_run
    from trainer.continued_pretraining import migrate_phase

    config = DistributedConfig(**SETTINGS, target_tokens=TARGET)
    validate_overnight_run(config)
    torch.set_num_threads(4)
    manifest = Path("/data/scholarly-code") / SNAPSHOT / "manifest.json"
    if not json.loads(manifest.read_text()).get("readiness_gate_passed"):
        raise ValueError("No passed readiness gate")
    lineage = migrate_phase(
        Path("/runs") / PARENT,
        Path("/runs") / (run_id + "-seed/step-000000000.pt"),
        manifest,
        config,
        {
            "target_tokens": TARGET,
            "code_share": 60,
            "eu_share": 5,
            "code_max_passes": 1,
            "fresh_code_required": True,
            "snapshot": SNAPSHOT,
        },
    )
    run_volume.commit()
    return {"seed": run_id + "-seed/step-000000000.pt", "lineage": lineage}


@app.function(
    image=image,
    gpu="H100:8",
    cpu=32,
    memory=98304,
    timeout=21600,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def segment(run_id, provenance, resume, max_steps=None):
    from distributed_training.config import DistributedConfig
    from distributed_training.run_validation import validate_overnight_run

    config = DistributedConfig(**SETTINGS, target_tokens=TARGET)
    validate_overnight_run(config)
    import torch

    if torch.cuda.device_count() != 8:
        raise ValueError("Expected eight H100s")
    return worker.local(
        run_id,
        SNAPSHOT,
        config.as_dict(),
        provenance,
        resume,
        18000,
        max_steps,
    )


@app.function(
    image=image,
    cpu=1,
    memory=4096,
    timeout=82800,
    retries=0,
    volumes={"/runs": run_volume, "/data": data_volume},
)
def train(run_id, provenance):
    try:
        seed = initialize.remote(run_id)
        resume = seed["seed"]
        previous = None
        for index in range(8):
            segment_id = f"{run_id}-s{index:02d}"
            mark(
                run_id,
                "qualifying" if index < 2 else "training",
                segment=segment_id,
                phase_target=TARGET,
                parent_tokens=seed["lineage"]["parent_tokens"],
                snapshot=SNAPSHOT,
            )
            result = segment.remote(segment_id, provenance, resume, 2 if index < 2 else None)
            if previous and previous["final_validation"] != result["initial_validation"]:
                raise ValueError("Fresh-process restore validation mismatch")
            if result.get("data_error") or result["stop_reason"] not in [
                "step_limit",
                "time_limit",
                "token_limit",
            ]:
                raise ValueError("Training did not stop cleanly: " + str(result.get("stop_reason")))
            run_volume.reload()
            (Path("/runs") / run_id / f"segment-{index:02d}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            run_volume.commit()
            if TARGET - result["total_loss_tokens"] < 32 * 8192:
                checkpoint = str(Path(result["final_checkpoint"]).relative_to("/runs"))
                completion = finish.spawn(run_id, checkpoint)
                mark(
                    run_id,
                    "training_completed",
                    actual_new_tokens=result["total_loss_tokens"],
                    lifetime_tokens=result["lifetime_loss_tokens"],
                    final_checkpoint=checkpoint,
                    finalization_call_id=completion.object_id,
                )
                return result
            previous = result
            resume = str(Path(result["final_checkpoint"]).relative_to("/runs"))
        raise ValueError("Segment budget exhausted")
    except Exception as exc:
        mark(run_id, "failed", error=f"{type(exc).__name__}: {str(exc)[:1500]}")
        raise


@app.function(
    image=image, cpu=1, memory=2048, timeout=3600, retries=0, volumes={"/runs": run_volume}
)
def wait_for_data(run_id, code_call, original_call, provenance, audit_call=None, audit_round=0):
    """Chain bounded CPU waiters; no GPU is allocated while preparation runs."""
    validate_paths(run_id, SNAPSHOT, PARENT)
    stop = time.monotonic() + 2700
    try:
        while time.monotonic() < stop:
            try:
                if audit_call is None:
                    mark(
                        run_id,
                        "collecting_fresh_code",
                        collection_call_id=code_call,
                        target_new_code_tokens=6_000_000_000,
                    )
                    collection = modal.FunctionCall.from_id(code_call).get(timeout=30)
                    if collection.get("audit_call_id"):
                        modal.FunctionCall.from_id(collection["audit_call_id"]).cancel(
                            terminate_containers=True
                        )
                    modal.FunctionCall.from_id(original_call).get(timeout=30)
                    audit_call = audit_augmented.spawn(audit_round).object_id
                mark(
                    run_id,
                    "auditing_fresh_code",
                    audit_call_id=audit_call,
                    target_new_code_tokens=6_000_000_000,
                )
                assessment = modal.FunctionCall.from_id(audit_call).get(timeout=30)
                if assessment["stage"] != "audited_not_training_ready":
                    raise ValueError("Unexpected audit result")
                mark(run_id, "checking_capacity_and_freezing")
                preparation = freeze_and_schedule.spawn(
                    run_id, original_call, provenance, audit_round
                )
                mark(
                    run_id,
                    "freezing_snapshot",
                    preparation_call_id=preparation.object_id,
                    collection_call_id=code_call,
                    audit_call_id=audit_call,
                )
                return {"preparation_call_id": preparation.object_id}
            except TimeoutError:
                continue
        next_call = wait_for_data.spawn(
            run_id, code_call, original_call, provenance, audit_call, audit_round
        )
        mark(
            run_id,
            "preparation_pending",
            continuation_call_id=next_call.object_id,
            collection_call_id=code_call,
            audit_call_id=audit_call,
        )
        return {"continuation_call_id": next_call.object_id}
    except Exception as exc:
        mark(run_id, "data_gate_failed", error=f"{type(exc).__name__}: {str(exc)[:1500]}")
        raise


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=21600,
    retries=0,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def freeze_and_schedule(run_id, original_call, provenance, audit_round=0):
    try:
        prepared = freeze_snapshot.local(audit_round)
        folder = Path("/runs") / run_id
        (folder / "data-readiness.json").write_text(json.dumps(prepared, indent=2) + "\n")
        run_volume.commit()
        training = train.spawn(run_id, provenance)
        mark(run_id, "training_scheduled", training_call_id=training.object_id, snapshot=SNAPSHOT)
        return {"training_call_id": training.object_id, "prepared": prepared}
    except ValueError as exc:
        snapshot_path = Path("/data/scholarly-code") / SNAPSHOT
        capacity_failure = any(
            message in str(exc)
            for message in [
                "new eligible code",
                "eight languages",
                "path-task families",
                "Cannot retain",
                "Selected snapshot lost required code variety",
                "held-out parents",
            ]
        )
        if audit_round < 8 and capacity_failure and not (snapshot_path / "raw.sqlite").exists():
            collection = collect_more_code.spawn(audit_round + 1)
            continuation = wait_for_data.spawn(
                run_id, collection.object_id, original_call, provenance, None, audit_round + 1
            )
            mark(
                run_id,
                "collecting_more_after_filtering",
                reason=str(exc),
                audit_round=audit_round + 1,
                collection_call_id=collection.object_id,
                continuation_call_id=continuation.object_id,
            )
            return {"continuation_call_id": continuation.object_id, "capacity_retry": True}
        mark(run_id, "data_gate_failed", error=f"{type(exc).__name__}: {str(exc)[:1500]}")
        raise
    except Exception as exc:
        mark(run_id, "data_gate_failed", error=f"{type(exc).__name__}: {str(exc)[:1500]}")
        raise


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=14400,
    retries=0,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/runs": run_volume, "/data": data_volume},
)
def finish(run_id, checkpoint):
    """Final matched evaluation and small plot uploads, independent of storage quota."""
    import importlib.util

    from huggingface_hub import HfApi, hf_hub_download

    from distributed_training.train import atomic_json

    folder = Path("/runs") / run_id
    results = {}
    try:
        scorer = modal.Function.from_name("simple-llm-eu-expanded-validation", "score")
        results["expanded_validation"] = scorer.remote(
            "simple-llm-eu-expanded-validation-20261019", checkpoint, run_id
        )
    except Exception as exc:  # noqa: BLE001 -- retain finalization errors independently
        results["expanded_validation_error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
    # This uses new representative held-outs throughout; old one-window diagnostics
    # are never substituted for them in the overnight curves.
    spec = importlib.util.spec_from_file_location(
        "curve_builder", "/project/loss_curves/build_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PHASES = [("Fresh multilingual code — overnight 10B", run_id)]
    output = folder / "training-curves"
    try:
        run_volume.reload()
        module.build(Path("/runs"), output)
        results["curves"] = str(output)
        api = HfApi()
        repo = "jfrick/124M-10.879B"
        if not api.model_info(repo).private:
            raise ValueError("Private run cannot publish to public model")
        revision = api.upload_folder(
            repo_id=repo,
            folder_path=str(output),
            path_in_repo="evaluation/" + run_id,
            commit_message="Add overnight continuation curves and metrics",
        )
        from training_data.subword import digest

        for name in ["loss-curves.svg", "optimizer-curves.svg", "provenance.json"]:
            downloaded = Path(
                hf_hub_download(
                    repo,
                    "evaluation/" + run_id + "/" + name,
                    revision=revision.oid,
                    force_download=True,
                )
            )
            if digest(downloaded) != digest(output / name):
                raise ValueError("Published curve hash mismatch")
        results["hf_curves"] = {
            "repo": repo,
            "revision": revision.oid,
            "verified": True,
            "format": "SVG",
        }
    except Exception as exc:  # noqa: BLE001 -- retain finalization errors independently
        results["curves_or_upload_error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
    try:
        import shutil

        from training_data.subword import digest

        from trainer.hub_checkpoint import export_checkpoint

        checkpoint_path = Path("/runs") / checkpoint
        checkpoint_source = checkpoint_path.parent
        report = json.loads((checkpoint_source / "report.json").read_text())
        lifetime = report["lifetime_loss_tokens"]
        model_name = f"124M - {lifetime / 1e9:.3f}B"
        repo = f"jfrick/124M-{lifetime / 1e9:.3f}B"
        bundle = folder / "hf-export"
        export_checkpoint(checkpoint_path, bundle, checkpoint_source, run_id)
        shutil.copyfile(
            "/project/publication/generate_checkpoint_archive.py", bundle / "generate.py"
        )
        (bundle / "README.md").write_text(
            f"---\nlibrary_name: pytorch\nmodel_name: {model_name}\n---\n# {model_name}\n\n"
            f"Simple LLM base model; {lifetime:,} lifetime next-token targets; "
            f"{report['total_loss_tokens']:,} targets in this continuation.\n\n"
            "124,389,632 parameters, 8K context, frozen 32,768-token BPE. "
            "Eight H100s, cropped padding, cached RoPE, torch.compile, microbatch two, "
            "32 global windows, BF16 computation with FP32 weights. AdamW moments/RNG "
            "preserved; constant learning rate 3e-5. Code 60%, EU 5%, previous non-code "
            "sources scaled to 35%. Code documents are fresh after exact/whitespace "
            "deduplication and consumed at most once; non-code may reuse older data "
            "at most twice in this phase. See data-manifest.json for capacities and limitations.\n\n"
            "HumanEval/MBPP lexical screening and parent split checks do not prove "
            "semantic contamination absent. This is not instruction-tuned; generation "
            "may remain incorrect and repetitive. Full optimizer checkpoints stay on Modal.\n\n"
            "[Training and validation curves](evaluation/training-curves/README.md). "
            "Custom PyTorch code: install requirements.txt and run "
            "`python generate.py --prompt 'def add(a, b):' --tokens 64`.\n"
        )
        if output.exists():
            shutil.copytree(output, bundle / "evaluation/training-curves", dirs_exist_ok=True)
        api = HfApi()
        api.create_repo(repo, private=True, exist_ok=True)
        if not api.model_info(repo).private:
            raise ValueError("Refusing public model publication")
        revision = api.upload_folder(
            repo_id=repo,
            folder_path=str(bundle),
            commit_message="Publish completed fresh-code continuation and loss curves",
        )
        remote = api.model_info(repo, revision=revision.oid, files_metadata=True)
        remote_hash = next(
            x.lfs.sha256 for x in remote.siblings if x.rfilename == "model.safetensors"
        )
        if remote_hash != digest(bundle / "model.safetensors"):
            raise ValueError("Remote model hash mismatch")
        results["hf_model"] = {
            "repo": repo,
            "revision": revision.oid,
            "weights_hash_verified": True,
            "private": True,
        }
    except Exception as exc:  # noqa: BLE001 -- keep checkpoints despite HF quota failures
        results["model_upload_error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
    atomic_json(folder / "finalization.json", results)
    run_volume.commit()
    return results
