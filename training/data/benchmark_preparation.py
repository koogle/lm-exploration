"""Controlled CPU benchmark: same tokenizer, rows, SQLite durability and outputs."""

import hashlib
import random
import tempfile
import time
from pathlib import Path

from training_data.fast_raw_corpus import FastRawCorpus
from training_data.persistent_stream import PersistentStream
from training_data.raw_corpus import RawCorpus


def benchmark(rows_per_source=1024, words_per_row=256, sampled_rows=None, sampled_sources=None):
    rng = random.Random(42)
    sources = [
        {
            "name": name,
            "weight_percent": 50,
            "files": ["fixture"],
            "repo": "synthetic-benchmark",
            "revision": "0" * 40,
        }
        for name in ("fineweb_edu", "finemath_4plus")
    ]
    vocabulary = [
        "function",
        "return",
        "matrix",
        "probability",
        "learning",
        "variable",
        "import",
        "theorem",
    ]
    rows = {
        s["name"]: [
            {
                "id": str(i),
                "text": f"{s['name']} {i}: "
                + " ".join(
                    rng.choice(vocabulary) + str(rng.randrange(10000)) for _ in range(words_per_row)
                ),
            }
            for i in range(rows_per_source)
        ]
        for s in sources
    }

    if sampled_rows is not None:
        rows, sources = sampled_rows, sampled_sources
        rows_per_source = len(next(iter(rows.values())))
    recipe = {"sources": sources}

    def reader(source, shard, start):
        yield from enumerate(rows[source["name"]][start:], start)

    def fingerprint(db):
        h = hashlib.sha256()
        for table in ("documents", "cursors", "decisions"):
            for row in db.execute(f"SELECT * FROM {table} ORDER BY 1,2"):
                h.update(repr(row).encode())
        return h.hexdigest()

    results = {}
    with tempfile.TemporaryDirectory() as temporary:
        for mode in ("serial", "batched_parallel"):
            folder = Path(temporary) / mode
            corpus = (
                RawCorpus(folder, recipe)
                if mode == "serial"
                else FastRawCorpus(folder, recipe, workers=3, batch_size=64, reader=reader)
            )
            streams = {s["name"]: PersistentStream(reader) for s in sources}
            start = time.perf_counter()
            if mode == "serial":
                for _ in range(rows_per_source // 64 + 2):
                    for source in sources:
                        corpus.prepare(
                            source["name"], max_records=64, reader=streams[source["name"]]
                        )
                    corpus.report(5_000_000_000)
            else:
                pending = {}
                while True:
                    active = [s for s in sources if corpus.cursor(s["name"])[0] < 1]
                    if not active:
                        break
                    for source in active:
                        pending[source["name"]] = corpus.pool.submit(
                            corpus._fetch, source, corpus.cursor(source["name"])
                        )
                    for source in active:
                        corpus._write(source, pending.pop(source["name"]).result())
                    corpus.report(5_000_000_000)
            seconds = time.perf_counter() - start
            tokens = corpus.db.execute(
                "SELECT SUM(loss_tokens) FROM documents WHERE conflict=0"
            ).fetchone()[0]
            results[mode] = {
                "seconds": seconds,
                "rows_per_second": 2 * rows_per_source / seconds,
                "tokens_per_second": tokens / seconds,
                "counted_tokens": tokens,
                "fingerprint": fingerprint(corpus.db),
            }
            for stream in streams.values():
                stream.close()
            corpus.close()
    if results["serial"]["fingerprint"] != results["batched_parallel"]["fingerprint"]:
        raise ValueError("Benchmark outputs differed")
    results["speedup"] = results["serial"]["seconds"] / results["batched_parallel"]["seconds"]
    results["scope"] = (
        "Controlled CPU tokenizer/SQLite benchmark; excludes upstream download and Modal Volume commit"
    )
    return results
