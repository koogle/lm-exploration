"""Bounded CPU benchmark of the optimized preparation implementation."""

import json
import os
from pathlib import Path

import modal

from modal_scripts.modal_prepare_simple_llm import image

image = image.add_local_dir("modal_scripts", "/project/modal_scripts")
app = modal.App("lm-exploration-preparation-benchmark")


@app.function(image=image, cpu=16, memory=16384, timeout=900, retries=0, max_containers=1)
def measure(recipe):
    os.environ["RAYON_NUM_THREADS"] = "8"
    from training_data.benchmark_preparation import benchmark

    print("Starting controlled CPU benchmarks", flush=True)
    result = {
        "cpu_request": 16,
        "tokenizer_threads": 8,
        "source_workers": 3,
        "trials": [benchmark(), benchmark()],
    }
    import time
    from concurrent.futures import ThreadPoolExecutor

    from training_data.cached_shards import CachedShards
    from training_data.simple_llm import validate_recipe

    # The first two recipe sources stand in for the live corpus.
    sources = validate_recipe(recipe)["sources"][:2]
    for source in sources:
        source["weight_percent"] = 50

    print("Fetching pinned real-source samples", flush=True)

    def collect(source):
        print(f"Sampling {source['name']}", flush=True)
        rows = CachedShards()(source, source["files"][0])
        try:
            return [next(rows)[1] for _ in range(256)]
        finally:
            rows.close()

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        samples = list(pool.map(collect, sources))
    result["live_sample_fetch_seconds"] = time.perf_counter() - started
    print("Real-source samples downloaded; benchmarking", flush=True)
    for source in sources:
        source["files"] = [source["files"][0]]
    result["live_sample"] = benchmark(
        sampled_sources=sources, sampled_rows={s["name"]: r for s, r in zip(sources, samples)}
    )
    result["live_sample"]["source_names"] = [s["name"] for s in sources]
    return result


@app.local_entrypoint()
def main(recipe: str, output: str = "artifacts/unpublished/preparation-speedup.json"):
    report = measure.remote(json.loads(Path(recipe).read_text()))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
