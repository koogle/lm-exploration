"""CPU-only data collection/auditing. There is intentionally no training entrypoint."""

from pathlib import Path

import modal

from modal_eu_continuation import data_volume, image

app = modal.App("simple-llm-data-readiness")
BASE = Path("/data/scholarly-code/readiness-20261020-v1")
PRIOR = Path("/data/scholarly-code/code-heavy-fresh-20261019/staging/raw.sqlite")
EU = Path("/data/scholarly-code/eu-continuation-20261018/snapshots/ecb-ema-one-pass-v1/raw.sqlite")


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
def collect_code():
    from training_data.overnight_corpus import prepare_code

    result = prepare_code(
        BASE / "multilingual",
        PRIOR,
        EU,
        target=6_500_000_000,
        seconds=80000,
        commit=data_volume.commit,
    )
    assessment = audit_fresh.spawn()
    return {"collection": result, "audit_call_id": assessment.object_id}


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
def audit_existing():
    from training_data.readiness_audit import audit, benchmark_index

    folder = BASE / "existing-audit"
    index, pins = benchmark_index(folder)
    return audit(
        [("prior", PRIOR, False), ("eu", EU, False)], folder, index, pins, commit=data_volume.commit
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
def audit_fresh():
    from training_data.readiness_audit import audit, benchmark_index

    folder = BASE / "combined-audit"
    index, pins = benchmark_index(folder)
    return audit(
        [
            ("prior", PRIOR, False),
            ("eu", EU, False),
            ("fresh_code", BASE / "multilingual/raw.sqlite", True),
        ],
        folder,
        index,
        pins,
        commit=data_volume.commit,
    )
