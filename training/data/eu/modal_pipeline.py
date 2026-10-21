"""Bounded CPU collectors: one serial writer per organization, HF-backed resume."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import modal
from filelock import FileLock

ROOT = Path(__file__).resolve().parent
app = modal.App("lm-exploration-eu-corpus")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements-modal.txt"))
    .env({"PYTHONPATH": "/root/eu"})
)
ca_bundle = os.environ.get("SSL_CERT_FILE")
if ca_bundle and Path(ca_bundle).is_file():
    image = image.env({"SSL_CERT_FILE": "/root/eu-ca.pem", "REQUESTS_CA_BUNDLE": "/root/eu-ca.pem"})
    image = image.add_local_file(ca_bundle, "/root/eu-ca.pem")
for name in (
    "pipeline.py",
    "document_queue.py",
    "campaign.py",
    "hf_store.py",
    "eu_corpus.py",
    "sources.json",
    "pipeline_sources.json",
):
    image = image.add_local_file(ROOT / name, f"/root/eu/{name}")
image = image.add_local_dir(ROOT / "adapters", "/root/eu/adapters", ignore=["__pycache__/"])
secret = modal.Secret.from_name("eu-corpus-huggingface", required_keys=["HF_TOKEN"])


@app.function(image=image, secrets=[secret], cpu=1, memory=2048, timeout=120, max_containers=1)
def verify_access(source_id="both"):
    import httpx
    from huggingface_hub import HfApi

    identity = HfApi().whoami()["name"]
    outcomes = {}
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        hosts = {
            "ecb": "www.ecb.europa.eu",
            "ema": "www.ema.europa.eu",
            "eurlex": "publications.europa.eu",
            "commission-press": "ec.europa.eu",
        }
        selected = ["ecb", "ema"] if source_id == "both" else [source_id]
        for host in [hosts[name] for name in selected]:
            response = client.get(f"https://{host}/robots.txt")
            outcomes[host] = response.status_code
    return {"hf_owner": identity, "robots_status": outcomes}


@app.function(
    image=image, secrets=[secret], cpu=1, memory=2048, timeout=3600, max_containers=1, retries=0
)
def collect_ecb(
    run_id: str, max_documents: int = 1000, work_seconds: int = 3000, refresh: bool = False
):
    from campaign import wait_for_dispatch
    from pipeline import run_source

    wait_for_dispatch("ecb", run_id)
    return run_source("ecb", run_id, max_documents, work_seconds, refresh)


@app.function(
    image=image, secrets=[secret], cpu=1, memory=2048, timeout=3600, max_containers=1, retries=0
)
def collect_ema(
    run_id: str, max_documents: int = 1000, work_seconds: int = 3000, refresh: bool = False
):
    from campaign import wait_for_dispatch
    from pipeline import run_source

    wait_for_dispatch("ema", run_id)
    return run_source("ema", run_id, max_documents, work_seconds, refresh)


@app.function(
    image=image, secrets=[secret], cpu=0.125, memory=512, timeout=25200, max_containers=1, retries=0
)
def campaign_ecb(
    campaign_id: str,
    rounds: int,
    max_documents: int,
    work_seconds: int,
    after_campaign: str = "",
    predecessor_wait_seconds: int = 1800,
):
    from campaign import run_campaign

    return run_campaign(
        "ecb",
        campaign_id,
        collect_ecb.remote,
        rounds,
        max_documents,
        work_seconds,
        after_campaign=after_campaign,
        predecessor_wait_seconds=predecessor_wait_seconds,
        spawn_callable=collect_ecb.spawn,
        call_loader=modal.FunctionCall.from_id,
    )


@app.function(
    image=image, secrets=[secret], cpu=0.125, memory=512, timeout=25200, max_containers=1, retries=0
)
def campaign_ema(
    campaign_id: str,
    rounds: int,
    max_documents: int,
    work_seconds: int,
    after_campaign: str = "",
    predecessor_wait_seconds: int = 1800,
):
    from campaign import run_campaign

    return run_campaign(
        "ema",
        campaign_id,
        collect_ema.remote,
        rounds,
        max_documents,
        work_seconds,
        after_campaign=after_campaign,
        predecessor_wait_seconds=predecessor_wait_seconds,
        spawn_callable=collect_ema.spawn,
        call_loader=modal.FunctionCall.from_id,
    )


@app.function(
    image=image, secrets=[secret], cpu=1, memory=2048, timeout=3600, max_containers=1, retries=0
)
def collect_eurlex(
    run_id: str, max_documents: int = 1000, work_seconds: int = 3000, refresh: bool = False
):
    from campaign import wait_for_dispatch
    from pipeline import run_source

    wait_for_dispatch("eurlex", run_id)
    return run_source("eurlex", run_id, max_documents, work_seconds, refresh)


@app.function(
    image=image, secrets=[secret], cpu=0.125, memory=512, timeout=25200, max_containers=1, retries=0
)
def campaign_eurlex(
    campaign_id: str,
    rounds: int,
    max_documents: int,
    work_seconds: int,
    after_campaign: str = "",
    predecessor_wait_seconds: int = 1800,
):
    from campaign import run_campaign

    return run_campaign(
        "eurlex",
        campaign_id,
        collect_eurlex.remote,
        rounds,
        max_documents,
        work_seconds,
        after_campaign=after_campaign,
        predecessor_wait_seconds=predecessor_wait_seconds,
        spawn_callable=collect_eurlex.spawn,
        call_loader=modal.FunctionCall.from_id,
    )


@app.function(
    image=image, secrets=[secret], cpu=1, memory=2048, timeout=3600, max_containers=1, retries=0
)
def collect_commission_press(
    run_id: str, max_documents: int = 1000, work_seconds: int = 3000, refresh: bool = False
):
    from campaign import wait_for_dispatch
    from pipeline import run_source

    wait_for_dispatch("commission-press", run_id)
    return run_source("commission-press", run_id, max_documents, work_seconds, refresh)


@app.function(
    image=image, secrets=[secret], cpu=0.125, memory=512, timeout=25200, max_containers=1, retries=0
)
def campaign_commission_press(
    campaign_id: str,
    rounds: int,
    max_documents: int,
    work_seconds: int,
    after_campaign: str = "",
    predecessor_wait_seconds: int = 1800,
):
    from campaign import run_campaign

    return run_campaign(
        "commission-press",
        campaign_id,
        collect_commission_press.remote,
        rounds,
        max_documents,
        work_seconds,
        after_campaign=after_campaign,
        predecessor_wait_seconds=predecessor_wait_seconds,
        spawn_callable=collect_commission_press.spawn,
        call_loader=modal.FunctionCall.from_id,
    )


@app.local_entrypoint()
def main(
    run_id: str = "",
    source: str = "both",
    max_documents: int = 1000,
    work_seconds: int = 3000,
    refresh: bool = False,
    check_only: bool = False,
    rounds: int = 1,
    after_campaign: str = "",
    predecessor_wait_seconds: int = 1800,
):
    from pipeline import load_mapping

    mapping = load_mapping(ROOT / "pipeline_sources.json")
    if source not in {"both", "ecb", "ema", "eurlex", "commission-press"}:
        raise ValueError("Unknown source")
    if not 1 <= max_documents <= mapping["limits"]["max_documents_per_source"]:
        raise ValueError("max_documents outside configured bound")
    if not 1 <= work_seconds <= mapping["limits"]["work_seconds"]:
        raise ValueError("work_seconds outside configured bound")
    if not 1 <= rounds <= mapping["limits"]["campaign_max_rounds"]:
        raise ValueError("Campaign rounds outside configured bound")
    if rounds > 1 and refresh:
        raise ValueError("Refresh inventories in a single round before starting a campaign")
    if after_campaign and (source == "both" or rounds == 1):
        raise ValueError("A predecessor requires one source and a multi-round campaign")
    if not 1 <= predecessor_wait_seconds <= 10800:
        raise ValueError("Predecessor wait must be 1–10800 seconds")
    access = verify_access.remote(source)
    print(json.dumps(access), flush=True)
    if access["hf_owner"] != mapping["owner"] or any(
        v != 200 for v in access["robots_status"].values()
    ):
        raise RuntimeError("Remote credential/source access check failed")
    if check_only:
        return
    run_id = run_id or datetime.now(UTC).strftime("eu-%Y%m%dT%H%M%S")
    # Dedicated ingestion ledger: separate from training/latent-workspace budgets.
    # This is a conservative compute reservation, not a measured bill.
    ledger_path = ROOT / "artifacts" / "modal-ingestion-budget.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(ledger_path) + ".lock"):
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"runs": []}
        if any(r["run_id"] == run_id for r in ledger["runs"]):
            raise ValueError(
                "run_id already reserved locally; choose a new ID to resume HF progress"
            )
        sources = ["ecb", "ema"] if source == "both" else [source]
        reservation = {
            "run_id": run_id,
            "sources": sources,
            "reserved_at": datetime.now(UTC).isoformat(),
            "max_documents_each": max_documents,
            "rounds": rounds,
            "after_campaign": after_campaign,
            "predecessor_wait_seconds": predecessor_wait_seconds if after_campaign else 0,
            "max_attempts_each_campaign": max_documents * rounds,
            "worker_seconds_each": 3600,
            "reserved_vcpu_seconds": len(sources) * (3600 * rounds + (3150 if rounds > 1 else 0)),
            "reserved_gib_seconds": len(sources)
            * (2 * 3600 * rounds + (12600 if rounds > 1 else 0)),
        }
        ledger["runs"].append(reservation)
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    functions = {
        "ecb": collect_ecb,
        "ema": collect_ema,
        "eurlex": collect_eurlex,
        "commission-press": collect_commission_press,
    }
    if rounds > 1:
        supervisors = {
            "ecb": campaign_ecb,
            "ema": campaign_ema,
            "eurlex": campaign_eurlex,
            "commission-press": campaign_commission_press,
        }
        calls = {
            s: supervisors[s].spawn(
                run_id,
                rounds,
                max_documents,
                work_seconds,
                after_campaign,
                predecessor_wait_seconds,
            )
            for s in sources
        }
    else:
        calls = {
            s: functions[s].spawn(run_id, max_documents, work_seconds, refresh) for s in sources
        }
    launch = {
        "run_id": run_id,
        "rounds": rounds,
        "after_campaign": after_campaign,
        "predecessor_wait_seconds": predecessor_wait_seconds if after_campaign else 0,
        "calls": {s: c.object_id for s, c in calls.items()},
        "datasets": {s: mapping["sources"][s]["repo_id"] for s in sources},
    }
    (ROOT / "artifacts" / f"modal-launch-{run_id}.json").write_text(
        json.dumps(launch, indent=2) + "\n"
    )
    print(json.dumps(launch), flush=True)
    # Keep this coordinator alive for both workers. Detached local entrypoints only
    # guarantee survival of the last triggered function; HF checkpoints survive either.
    for s, call in calls.items():
        try:
            result = call.get()
        except Exception as error:  # noqa: BLE001 - retain per-source failure and collect sibling result
            result = {"source_id": s, "run_id": run_id, "status": "failed", "error": str(error)}
        (ROOT / "artifacts" / f"modal-result-{run_id}-{s}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)
