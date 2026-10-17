import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

from training_data.scholarly import adapt, load_recipe

from tokenizer_research import CorpusBuilder, load_spec, records, train_candidate

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    spec = load_spec(ROOT / "tokenizer/research/learning-curve-v1.json")
    spec.update(
        sizes_mib=[1 / 128, 1 / 64],
        evaluation_mib=1 / 256,
        vocab_sizes=[512, 1024],
        max_document_bytes=2048,
        max_parent_bytes=4096,
        chunk_bytes=512,
    )
    recipe = load_recipe(ROOT / spec["source_recipe"])
    recipe["sources"] = [copy.deepcopy(recipe["sources"][0])]
    source = recipe["sources"][0]
    source.update(weight_percent=100, files=["fixture.parquet"])
    rng = random.Random(42)
    rows = [
        {
            "id": str(i),
            "text": f"Document {i}: "
            + "".join(
                rng.choice("abcdefghijklmnopqrstuvwxyz 0123456789αβé\n") for _ in range(1500)
            ),
        }
        for i in range(200)
    ]
    builder = CorpusBuilder(root, spec, recipe)
    reader = lambda s, f, start: enumerate(rows[start:], start)
    return builder, reader, rows


class LearningCurveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_nested_utf8_samples_and_shared_evaluation(self):
        builder, reader, _ = fixture(self.root / "data")
        self.addCleanup(builder.close)
        builder.collect("fineweb_edu", reader=reader)
        manifest = builder.freeze()
        path = self.root / "data/frozen"
        small = "".join(t for _, t, _ in records(path, manifest, "train", {"fineweb_edu": 8192}))
        large = "".join(t for _, t, _ in records(path, manifest, "train", {"fineweb_edu": 16384}))
        self.assertTrue(large.startswith(small))
        self.assertLessEqual(8192 - len(small.encode()), 3)
        train_parents = {p for _, _, p in records(path, manifest, "train")}
        eval_parents = {p for _, _, p in records(path, manifest, "validation")}
        self.assertFalse(train_parents & eval_parents)
        self.assertFalse(builder.db.execute('SELECT 1 FROM chunks WHERE split="test"').fetchone())
        config = ROOT / "tokenizer/scholarly-code-v1/config.json"
        a = train_candidate(path, self.root / "a", 1 / 128, 512, config)
        b = train_candidate(path, self.root / "b", 1 / 64, 1024, config)
        self.assertEqual(a["corpus_manifest_sha256"], b["corpus_manifest_sha256"])
        self.assertEqual(a["metrics"]["bytes"], b["metrics"]["bytes"])
        self.assertTrue(a["correctness_passed"] and b["correctness_passed"])
        self.assertEqual(a["actual_vocab_size"], 512)
        self.assertEqual(b["actual_vocab_size"], 1024)
        self.assertGreater(
            b["actual_training_bytes"]["fineweb_edu"], a["actual_training_bytes"]["fineweb_edu"]
        )

    def test_resume_and_identity_guard(self):
        builder, reader, _ = fixture(self.root / "data")
        builder.collect("fineweb_edu", max_records=3, reader=reader)
        cursor = builder.db.execute("SELECT row FROM cursors").fetchone()[0]
        self.assertEqual(cursor, 3)
        spec, recipe = builder.spec, builder.recipe
        builder.close()
        resumed = CorpusBuilder(self.root / "data", spec, recipe)
        self.addCleanup(resumed.close)
        resumed.collect("fineweb_edu", reader=reader)
        self.assertTrue(resumed.complete("fineweb_edu"))
        changed = copy.deepcopy(spec)
        changed["seed"] += 1
        with self.assertRaises(ValueError):
            CorpusBuilder(self.root / "data", changed, recipe)

    def test_interrupted_uncommitted_batch_replays_from_durable_cursor(self):
        builder, reader, rows = fixture(self.root / "data")
        self.addCleanup(builder.close)
        builder.collect("fineweb_edu", max_records=3, reader=reader)
        before = builder.counts("fineweb_edu")

        def interrupted(source, shard, start):
            yield start, rows[start]
            raise OSError("simulated network interruption")

        with self.assertRaises(OSError):
            builder.collect("fineweb_edu", reader=interrupted)
        self.assertEqual(builder.db.execute("SELECT row FROM cursors").fetchone()[0], 3)
        self.assertEqual(builder.counts("fineweb_edu"), before)
        builder.collect("fineweb_edu", reader=reader)
        self.assertTrue(builder.complete("fineweb_edu"))

    def test_cross_split_duplicate_quarantine(self):
        builder, _, _ = fixture(self.root / "data")
        self.addCleanup(builder.close)
        source = builder.recipe["sources"][0]
        ids = {}
        for i in range(100):
            row = {"id": str(i), "text": "Identical shared passage."}
            doc, _ = adapt(source, row)
            ids.setdefault(doc["split"], row)
        rows = [ids["train"], ids["validation"]]
        with self.assertRaises(ValueError):  # Inventory cannot satisfy quotas.
            builder.collect(
                "fineweb_edu", reader=lambda s, f, start: enumerate(rows[start:], start)
            )
        self.assertEqual(builder.counts("fineweb_edu")["train"], 0)
        with self.assertRaises(ValueError):
            builder.freeze()

    def test_corruption_and_unrequested_candidate_rejected(self):
        builder, reader, _ = fixture(self.root / "data")
        self.addCleanup(builder.close)
        builder.collect("fineweb_edu", reader=reader)
        builder.freeze()
        path = self.root / "data/frozen"
        config = ROOT / "tokenizer/scholarly-code-v1/config.json"
        with self.assertRaises(ValueError):
            train_candidate(path, self.root / "bad", 2, 512, config)
        with (path / "fineweb_edu-validation.jsonl").open("a") as handle:
            handle.write(json.dumps({"text": "tampered"}) + "\n")
        with self.assertRaises(ValueError):
            train_candidate(path, self.root / "bad2", 1 / 128, 512, config)


if __name__ == "__main__":
    unittest.main()
