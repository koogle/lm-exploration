"""Eight disjoint CPU shard workers; merge once before existing readiness gates."""

import time

import modal

from modal_overnight import BASE, PRIOR, data_volume, image

image = image.add_local_file("modal_overnight.py", "/project/modal_overnight.py")
app = modal.App("simple-llm-parallel-code")
FOLDER = BASE / "parallel-code"
WORKERS = 8


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=7200,
    retries=0,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/data": data_volume},
)
def exclusions():
    from training_data.parallel_code import build_exclusions

    result = build_exclusions(BASE / "multilingual/raw.sqlite", PRIOR, FOLDER)
    data_volume.commit()
    return result


@app.function(
    image=image,
    cpu=8,
    memory=24576,
    timeout=86400,
    retries=2,
    max_containers=8,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    volumes={"/data": data_volume},
)
def partition(worker, identity, target):
    from training_data.parallel_code import prepare_partition

    return prepare_partition(
        FOLDER / f"worker-{worker:02d}",
        FOLDER / "exclusions.sqlite",
        identity,
        worker,
        WORKERS,
        target,
        commit=data_volume.commit,
    )


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=21600,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume},
)
def merge():
    from training_data.parallel_code import merge_partitions

    from trainer.distributed_train import atomic_json

    result = merge_partitions(
        BASE / "multilingual/raw.sqlite",
        [FOLDER / f"worker-{i:02d}/raw.sqlite" for i in range(WORKERS)],
        commit=data_volume.commit,
    )
    atomic_json(BASE / "multilingual/progress.json", {"stage": "parallel_code_merged", **result})
    data_volume.commit()
    return result


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume},
)
def collect(audit_round=0):
    """Stage >8.8B pre-merge targets so final global dedup has room to remove clones."""
    from trainer.distributed_train import atomic_json

    identity = exclusions.remote()
    target = 1_100_000_000 + audit_round * 350_000_000
    calls = [partition.spawn(i, identity, target) for i in range(WORKERS)]
    results = {}
    atomic_json(
        FOLDER / f"calls-r{audit_round}.json",
        {"workers": [c.object_id for c in calls], "per_worker_target": target},
    )
    data_volume.commit()
    try:
        while len(results) < WORKERS:
            for worker, call in enumerate(calls):
                if worker in results:
                    continue
                try:
                    results[worker] = call.get(timeout=0)
                except TimeoutError:
                    pass
            atomic_json(
                FOLDER / "coordinator.json",
                {
                    "stage": "collecting",
                    "workers": WORKERS,
                    "completed_workers": sorted(results),
                    "per_worker_target": target,
                    "updated_unix": time.time(),
                },
            )
            data_volume.commit()
            if len(results) < WORKERS:
                time.sleep(30)
        result = merge.remote()
        atomic_json(
            FOLDER / "coordinator.json",
            {"stage": "merged", "collection": result, "updated_unix": time.time()},
        )
        data_volume.commit()
        return {"collection": result}
    except Exception as exc:
        atomic_json(
            FOLDER / "coordinator.json",
            {
                "stage": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:1200]}",
                "updated_unix": time.time(),
            },
        )
        data_volume.commit()
        raise
