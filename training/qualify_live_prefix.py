"""Opt-in real-upstream data-only smoke test; no model training."""

import json
import tempfile
import time
from pathlib import Path

import torch
from training_data.live_stream import LiveTokenMixture
from training_data.prefetch import PrefetchedBatches
from training_data.scholarly import SPLIT_VERSION, load_recipe
from training_data.subword import TOKENIZER_SHA256


def main():
    source = next(
        s for s in load_recipe("tokenizer/recipe-v1.json")["sources"] if s["name"] == "code_python"
    )
    source["weight_percent"] = 100
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "format": "scholarly-live-v1",
                    "recipe": {"sources": [source]},
                    "max_passes": 2,
                    "split_policy": SPLIT_VERSION,
                    "tokenizer_sha256": TOKENIZER_SHA256,
                    "max_seen_documents": 10000,
                }
            )
        )

        def factory():
            return LiveTokenMixture(path, context=256, allow_unreviewed=True)

        started = time.monotonic()
        data = PrefetchedBatches(factory, 2, depth=3)
        for _ in range(2):
            data.next_batch(2)
            data.acknowledge()
        saved = data.state_dict()
        expected = data.next_batch(2)
        stats = data.stats()
        data.close()
        resumed = PrefetchedBatches(factory, 2, depth=3, state=saved)
        actual = resumed.next_batch(2)
        assert all(torch.equal(x, y) for x, y in zip(expected, actual))
        resumed.acknowledge()
        report = {
            "source": source["name"],
            "revision": source["revision"],
            "read_without_full_corpus_preparation": True,
            "unacknowledged_batch_restored_exactly": True,
            "consumed_loss_tokens": sum(resumed.exposures.values()),
            "elapsed_seconds": time.monotonic() - started,
            "prefetch": stats,
            "model_training": False,
        }
        resumed.close()
        Path("results/live-prefix-check-20261018.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report))


if __name__ == "__main__":
    main()
