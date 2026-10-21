"""Fresh-only 5B-token continuation, CPU preparation followed by eight H100s."""

import json
import time
from pathlib import Path

import modal

from distributed.modal_train import validate_paths, worker
from modal_eu_continuation import data_volume, image, run_volume, snapshot_path

image = image.add_local_file("modal_eu_continuation.py", "/project/modal_eu_continuation.py")
app = modal.App("simple-llm-code-heavy-5b")
ORIGINAL = Path(
    "/data/scholarly-code/scholarly-small-5b-raw-20261017/snapshots/scholarly-modern-10m-5b-20261017/manifest.json"
)
EU = Path("/data/scholarly-code/eu-continuation-20261018/snapshots/ecb-ema-one-pass-v1/raw.sqlite")
PARENT = "simple-llm-124m-eu-onepass-20261018-s02/step-000003838.pt"
SNAPSHOT = "code-heavy-fresh-20261019/snapshots/code60-5b-books-v2"
VALIDATION = "simple-llm-eu-expanded-validation-20261019"
TARGET = 5_000_000_000
SETTINGS = {
    "optimizer": "adamw",
    "data_loader": "raw",
    "world_size": 8,
    "global_windows": 32,
    "microbatch": 2,
    "context": 8192,
    "target_tokens": TARGET,
    "warmup_tokens": 1,
    "peak_lr": 3e-5,
    "min_lr_ratio": 1.0,
    "checkpoint_every": 1000,
    "evaluate_every": 500,
    "allow_unreviewed_data": True,
    "prefetch_batches": 4,
    "crop_padding": True,
    "cache_rope": True,
    "compile_model": True,
    "n_layer": 14,
    "n_head": 12,
    "n_embd": 768,
}


def mark(folder, stage, **details):
    from trainer.loop import atomic_json

    atomic_json(folder / "workflow.json", {"stage": stage, "updated_unix": time.time(), **details})
    run_volume.commit()


@app.function(
    image=image,
    cpu=16,
    memory=32768,
    timeout=14400,
    retries=0,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/data": data_volume},
    max_containers=1,
)
def prepare():
    import shutil

    from training_data.cached_shards import CachedShards
    from training_data.fresh_book_pool import FreshBookPoolCorpus
    from training_data.fresh_raw_corpus import initialize_fresh

    from trainer.continued_pretraining import coding_double_mix, token_targets
    from trainer.loop import atomic_json

    started = time.monotonic()
    destination = snapshot_path(SNAPSHOT)
    if (destination / "manifest.json").exists():
        return json.loads((destination / "manifest.json").read_text())
    staging = destination.parent.parent / "staging"
    (staging / "failure.json").unlink(missing_ok=True)
    print(json.dumps({"stage": "initializing_fresh_exclusions"}), flush=True)
    recipe = initialize_fresh(staging, ORIGINAL, [EU])
    manifest = json.loads(ORIGINAL.read_text())
    weights = coding_double_mix(manifest["sources"])
    corpus = FreshBookPoolCorpus(
        staging,
        recipe,
        weights,
        token_targets(weights, TARGET),
        workers=6,
        batch_size=128,
        reader=CachedShards(),
    )
    try:
        atomic_json(
            staging / "progress.json", {**corpus.report(TARGET), "updated_unix": time.time()}
        )
        data_volume.commit()
        while True:
            result = corpus.prepare_parallel(TARGET, max_seconds=20)
            status = corpus.report(TARGET)
            atomic_json(staging / "progress.json", {**status, "updated_unix": time.time()})
            data_volume.commit()
            print(
                json.dumps(
                    {
                        "stage": "preparing",
                        "fresh_tokens": sum(
                            s["train_loss_tokens"] for s in status["sources"].values()
                        ),
                        "source_tokens": {
                            n: s["train_loss_tokens"] for n, s in status["sources"].items()
                        },
                        **result,
                    }
                ),
                flush=True,
            )
            if result["ready"]:
                break
            if time.monotonic() - started >= 10800:
                return {"preparation_pending": True, "progress": status}
        temporary = destination.with_name(destination.name + "-partial")
        if temporary.exists():
            shutil.rmtree(temporary)
        frozen = corpus.export(temporary)
        temporary.rename(destination)
        data_volume.commit()
        return frozen
    except Exception as exc:
        atomic_json(
            staging / "failure.json",
            {"error": f"{type(exc).__name__}: {exc}", "updated_unix": time.time()},
        )
        raise
    finally:
        corpus.close()
        data_volume.commit()


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

    from distributed.config import DistributedConfig
    from trainer.continued_pretraining import migrate_phase

    torch.set_num_threads(4)
    folder = Path("/runs") / (run_id + "-seed")
    lineage = migrate_phase(
        Path("/runs") / PARENT,
        folder / "step-000000000.pt",
        snapshot_path(SNAPSHOT) / "manifest.json",
        DistributedConfig(**SETTINGS),
        {
            "fresh_only": True,
            "coding_share": 60,
            "target_tokens": TARGET,
            "book_allocation": "fresh-book-pool-v1",
        },
    )
    run_volume.commit()
    return {"seed": str(folder.relative_to("/runs") / "step-000000000.pt"), "lineage": lineage}


@app.function(
    image=image,
    gpu="H100:8",
    cpu=32,
    memory=98304,
    timeout=25200,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def segment(run_id, provenance, resume, max_steps=None):
    import torch

    if torch.cuda.device_count() != 8:
        raise RuntimeError("Expected eight GPUs")
    return worker.local(
        run_id,
        SNAPSHOT,
        SETTINGS,
        provenance,
        resume,
        21600,
        max_steps,
        dataset_root="scholarly-code",
    )


@app.function(
    image=image, cpu=1, memory=4096, timeout=10800, retries=0, volumes={"/runs": run_volume}
)
def assessment(run_id, checkpoint):
    from trainer.eu_expanded_validation import paired_summary

    scorer = modal.Function.from_name("simple-llm-eu-expanded-validation", "score")
    label = run_id + "-final"
    timing = scorer.remote(VALIDATION, checkpoint, label)
    run_volume.reload()
    folder = Path("/runs") / VALIDATION
    scores = json.loads((folder / f"scores-{label}.json").read_text())
    sample = json.loads((folder / "sample-summary.json").read_text())
    result = {"timing": timing}
    for baseline in ("before", "after"):
        result[baseline] = paired_summary(
            json.loads((folder / f"scores-{baseline}.json").read_text()), scores, sample
        )
    (Path("/runs") / run_id / "expanded-validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    run_volume.commit()
    return {"completed": True, "total_tokens": timing["total_tokens"]}


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=82800,
    retries=0,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def start(run_id, provenance):
    """Prepare in restartable CPU chunks; hand off to a separate training controller."""
    validate_paths(run_id, SNAPSHOT, PARENT)
    folder = Path("/runs") / run_id
    folder.mkdir(exist_ok=False)
    (folder / "settings.json").write_text(json.dumps(SETTINGS, indent=2) + "\n")
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    mark(folder, "preparing", snapshot=SNAPSHOT, target_tokens=TARGET)
    try:
        for chunk in range(5):
            call = prepare.spawn()
            mark(
                folder,
                "preparing",
                snapshot=SNAPSHOT,
                target_tokens=TARGET,
                prepare_call_id=call.object_id,
                chunk=chunk,
            )
            manifest = call.get(timeout=14400)
            if manifest.get("preparation_pending"):
                continue
            if manifest.get("freshness", {}).get("prior_hash_overlap") != 0:
                raise ValueError("Missing fresh-data disjointness audit")
            training = train.spawn(run_id, provenance)
            return {"training_call_id": training.object_id}
        raise RuntimeError("Bounded preparation budget exhausted; progress retained")
    except Exception as exc:
        run_volume.reload()
        mark(folder, "failed", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
        raise


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=82800,
    retries=0,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def train(run_id, provenance):
    """Qualify actual new-data start/restore, then bounded 5B continuation."""
    validate_paths(run_id, SNAPSHOT, PARENT)
    folder = Path("/runs") / run_id
    previous = None
    try:
        seed = initialize.remote(run_id)
        resume = seed["seed"]
        for index in range(5):
            segment_id = f"{run_id}-s{index:02d}"
            mark(
                folder,
                "qualifying" if index < 2 else "training",
                segment=segment_id,
                target_tokens=TARGET,
                phase_tokens=previous["total_loss_tokens"] if previous else 0,
                parent_tokens=seed["lineage"]["parent_tokens"],
                snapshot=SNAPSHOT,
            )
            report = segment.remote(segment_id, provenance, resume, 2 if index < 2 else None)
            if previous and previous["final_validation"] != report["initial_validation"]:
                raise RuntimeError("Fresh-process restore validation mismatch")
            if report.get("data_error") or report["stop_reason"] not in (
                "step_limit",
                "time_limit",
                "token_limit",
            ):
                raise RuntimeError(f"Unexpected training stop: {report.get('stop_reason')}")
            if previous and report["total_loss_tokens"] <= previous["total_loss_tokens"]:
                raise RuntimeError("No training progress")
            run_volume.reload()
            (folder / f"segment-{index:02d}.json").write_text(json.dumps(report, indent=2) + "\n")
            if (
                TARGET - report["total_loss_tokens"]
                < SETTINGS["global_windows"] * SETTINGS["context"]
            ):
                final = str(Path(report["final_checkpoint"]).relative_to("/runs"))
                evaluation = assessment.spawn(run_id, final)
                mark(
                    folder,
                    "completed",
                    target_tokens=TARGET,
                    phase_tokens=report["total_loss_tokens"],
                    lifetime_tokens=report["lifetime_loss_tokens"],
                    final_checkpoint=final,
                    unconsumed_tail_tokens=TARGET - report["total_loss_tokens"],
                    assessment_call_id=evaluation.object_id,
                )
                return report
            previous = report
            resume = str(Path(report["final_checkpoint"]).relative_to("/runs"))
        raise RuntimeError("Reached bounded segment limit")
    except Exception as exc:
        run_volume.reload()
        mark(
            folder,
            "failed",
            error=f"{type(exc).__name__}: {str(exc)[:1000]}",
            phase_tokens=previous["total_loss_tokens"] if previous else 0,
        )
        raise


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=180,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def inspect(run_id):
    # Warm containers otherwise retain the first mounted volume snapshot.
    data_volume.reload()
    run_volume.reload()
    validate_paths(run_id, SNAPSHOT, "")
    result = {}
    folder = Path("/runs") / run_id
    for name in ("workflow.json", "expanded-validation.json"):
        if (folder / name).exists():
            result[name] = json.loads((folder / name).read_text())
    staging = snapshot_path(SNAPSHOT).parent.parent / "staging"
    for name in ("progress.json", "failure.json"):
        if (staging / name).exists():
            result[name] = json.loads((staging / name).read_text())
    return result
