"""Bounded batch-one sweep. Every timed variant must pass the reference probe."""

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import atomic_json
from megakernel.build_sweep import rank_trials, schedule


def run(root, attempt, manifest):
    import modal
    import torch

    from megakernel.model import load_teacher
    from megakernel.probe import run as probe

    out = Path(root) / "attempts" / attempt
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(Path(root) / "manifest.json", manifest)
    volume = modal.Volume.from_name("lm-exploration-teacher-rollouts")
    builds = Path("/experiment/megakernel/builds")
    report = {
        "status": "loading",
        "actual_billed_usd": None,
        "build": json.loads((builds / "build.json").read_text()),
        "trials": [],
        "protocol": "batch-one greedy 128-token cap; forward/reverse order",
    }
    started = time.monotonic()

    def save():
        report["pipeline_seconds"] = time.monotonic() - started
        report["eligible_ranking"] = rank_trials(report["trials"])
        atomic_json(out / "report.json", report)
        volume.commit()

    save()
    try:
        teacher = load_teacher("/runs/teacher-hf-cache")
        report["model_load_and_download_seconds"] = time.monotonic() - started
        report["status"] = "running"
        save()
        for round_id, variant in schedule():
            name = f"round-{round_id}-{variant['id']}"
            torch.cuda.reset_peak_memory_stats()
            # Probe reset checks also warm up each library before timing.
            result = probe(
                out / "trials",
                name,
                manifest,
                teacher=teacher,
                library=builds / f"{variant['id']}.so",
            )
            report["trials"].append(
                {"round": round_id, "variant": variant, "result": result}
            )
            gc.collect()
            torch.cuda.empty_cache()
            save()
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    return report


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], json.loads(Path(sys.argv[3]).read_text()))
