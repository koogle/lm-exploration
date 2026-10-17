"""Bounded CPU benchmark of the optimized preparation implementation."""

import json
import os
from pathlib import Path

import modal

from modal_prepare_scholarly import image

image = image.add_local_dir("trainer", "/project/trainer").add_local_file(
    "modal_prepare_scholarly.py", "/project/modal_prepare_scholarly.py"
)
app = modal.App("lm-exploration-preparation-benchmark")


@app.function(image=image, cpu=16, memory=16384, timeout=900, retries=0, max_containers=1)
def measure():
    os.environ["RAYON_NUM_THREADS"] = "8"
    from trainer.benchmark_preparation import benchmark

    result = {
        "cpu_request": 16,
        "tokenizer_threads": 8,
        "source_workers": 3,
        "trials": [benchmark(), benchmark()],
    }
    import time
    from concurrent.futures import ThreadPoolExecutor

    from training_data.cached_shards import CachedShards
    from training_data.scholarly import load_recipe

    sources = [
        s
        for s in load_recipe("/project/tokenizer/recipe-v1.json")["sources"]
        if s["name"] in ("code_python", "finemath_4plus")
    ]
    for source in sources:
        source["weight_percent"] = 50

    def collect(source):
        rows = CachedShards()(source, source["files"][5 if source["name"] == "code_python" else 0])
        try:
            return [next(rows)[1] for _ in range(256)]
        finally:
            rows.close()

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        samples = list(pool.map(collect, sources))
    result["live_sample_fetch_seconds"] = time.perf_counter() - started
    for source in sources:
        source["files"] = [source["files"][5 if source["name"] == "code_python" else 0]]
    result["live_sample"] = benchmark(
        sampled_sources=sources, sampled_rows={s["name"]: r for s, r in zip(sources, samples)}
    )
    result["live_sample"]["source_names"] = [s["name"] for s in sources]
    return result


@app.local_entrypoint()
def main():
    report = measure.remote()
    Path("results/preparation-speedup-20261017.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
