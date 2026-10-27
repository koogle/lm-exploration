"""Resume the matched KDA/RoPE checkpoints to one billion total positions."""

import json
import time
from pathlib import Path

import modal

from .long_study import STEPS, TOKENS_PER_STEP
from .modal_study import SNAP, app, data, image, registry, runs

TOTAL_STEPS = 1_000_000_000 // TOKENS_PER_STEP
SOURCE = "kda-100m-1793059841"
VARIANTS = ["baseline", "rope_triton"]


@app.function(timeout=39600, retries=0)
def supervise_continuation(run, deadline):
    assert deadline - time.time() <= 36000
    registry[run + "/ready"] = True
    while time.time() < deadline:
        if registry.get(run + "/done", False):
            return
        time.sleep(min(10, max(0, deadline - time.time())))
    for call in registry.get(run + "/calls", []):
        try:
            modal.FunctionCall.from_id(call).cancel()
        except Exception as error:  # noqa: BLE001
            print("Cancellation failed:", error, flush=True)
    registry[run + "/expired"] = True


@app.function(
    image=image,
    cpu=16,
    memory=32768,
    timeout=7200,
    retries=0,
    volumes={"/runs": runs, "/data": data},
)
def prepare_continuation(run, deadline):
    import hashlib
    import random
    import shutil
    import sqlite3
    import zlib

    import numpy as np
    from training_data.subword import load_frozen

    assert time.time() < deadline - 600
    runs.reload()
    origin, folder = Path("/runs") / SOURCE, Path("/runs") / run
    folder.mkdir(parents=True, exist_ok=True)
    old = json.loads((origin / "data.json").read_text())
    excluded = set(old["train"]["document_hashes"]) | set(old["validation"]["document_hashes"])
    tokenizer = load_frozen()
    db = sqlite3.connect(f"file:/data/{SNAP}/raw.sqlite?mode=ro&immutable=1", uri=True)
    sources = old["sources"]
    needed = (TOTAL_STEPS - STEPS) * TOKENS_PER_STEP + 1
    per = (needed + len(sources) - 1) // len(sources)
    documents, counts, seen = [], {}, set()
    for source in sources:
        count = 0
        cursor = db.execute(
            "SELECT hash,text FROM documents WHERE source=? AND split='train' AND conflict=0 ORDER BY rowid",
            (source,),
        )
        while count < per:
            assert time.time() < deadline - 300
            rows = cursor.fetchmany(128)
            if not rows:
                break
            rows = [(h, raw) for h, raw in rows if h not in excluded and h not in seen]
            encoded = tokenizer.encode_batch(
                [zlib.decompress(raw).decode() for _, raw in rows], add_special_tokens=False
            )
            for (h, _), enc in zip(rows, encoded):
                if h in seen:
                    continue
                ids = np.array([1, *enc.ids, 2], dtype=np.int32)
                documents.append((h, ids))
                seen.add(h)
                count += len(ids)
                if count >= per:
                    break
        assert count >= per, (source, count, per)
        counts[source] = count
        print("Prepared fresh tokens", source, count, flush=True)
    assert not seen & excluded
    random.Random(2027).shuffle(documents)
    total = sum(counts.values())
    arr = np.lib.format.open_memmap(folder / "train.npy", mode="w+", dtype=np.int32, shape=(total,))
    digest = hashlib.sha256()
    offset = 0
    for _, ids in documents:
        arr[offset : offset + len(ids)] = ids
        digest.update(ids.tobytes())
        offset += len(ids)
    arr.flush()
    del arr
    shutil.copyfile(origin / "validation.npy", folder / "validation.npy")
    val = np.load(folder / "validation.npy", mmap_mode="r")
    assert hashlib.sha256(val.tobytes()).hexdigest() == old["validation"]["tokens_sha256"]
    info = {
        "source_run": SOURCE,
        "snapshot": SNAP,
        "training_ready": old["training_ready"],
        "manifest_sha256": old["manifest_sha256"],
        "sources": sources,
        "train": {
            "tokens": total,
            "sources": counts,
            "tokens_sha256": digest.hexdigest(),
            "document_hashes": [h for h, _ in documents],
        },
        "validation": old["validation"],
        "excluded_previous_documents": len(excluded),
        "overlap_with_previous_train_or_validation": 0,
    }
    (folder / "data.json").write_text(json.dumps(info))
    runs.commit()
    return info


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=32400,
    retries=0,
    max_containers=4,
    volumes={"/runs": runs},
)
def train_continuation(run, variant, seed, deadline):
    from model_hybrid.long_study import train_impl

    assert variant in VARIANTS
    return train_impl(run, variant, seed, deadline, source=SOURCE, steps=TOTAL_STEPS, runtime=31800)
