import sqlite3
import tempfile
import unittest
from pathlib import Path

from training_data.fast_raw_corpus import FastRawCorpus
from training_data.raw_corpus import RawCorpus


def fixture():
    sources = [
        {
            "name": name,
            "weight_percent": 50,
            "files": ["a", "b"],
            "repo": "fixture",
            "revision": "0" * 40,
        }
        for name in ("fineweb_edu", "finemath_4plus")
    ]
    rows = {
        s["name"]: [
            {
                "id": str(i),
                "text": ("shared duplicate" if i < 3 else f"{s['name']} document {i}. ")
                + ("Text for tokenizer 12345. " * 10),
            }
            for i in range(40)
        ]
        for s in sources
    }

    def reader(source, shard, start):
        records = rows[source["name"]][:20] if shard == "a" else rows[source["name"]][20:]
        yield from enumerate(records[start:], start)

    return {"sources": sources}, reader


class FastRawCorpusTests(unittest.TestCase):
    def compare(self, base, fast):
        for table in ("documents", "cursors", "decisions"):
            self.assertEqual(
                base.db.execute(f"SELECT * FROM {table} ORDER BY 1,2").fetchall(),
                fast.db.execute(f"SELECT * FROM {table} ORDER BY 1,2").fetchall(),
            )
        for source in base.recipe["sources"]:
            self.assertEqual(base.counts(source["name"]), fast.counts(source["name"]))

    def test_parallel_matches_same_deterministic_schedule(self):
        recipe, reader = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = RawCorpus(root / "base", recipe)
            original = RawCorpus(root / "fast", recipe)
            for corpus in (base, original):
                corpus.prepare(recipe["sources"][0]["name"], max_records=3, reader=reader)
            original.close()
            fast = FastRawCorpus(root / "fast", recipe, workers=2, batch_size=3, reader=reader)
            self.compare(base, fast)
            for _ in range(30):
                needed = [
                    s
                    for s in recipe["sources"]
                    if not fast.enough(base.report(1000)["sources"][s["name"]])
                ]
                for source in needed:
                    base.prepare(source["name"], max_records=3, reader=reader)
                result = fast.prepare_parallel(1000, max_seconds=0)
                self.compare(base, fast)
                if result["ready"]:
                    break
            else:
                self.fail("Fixture did not reach target")
            fast.close()
            fast = FastRawCorpus(root / "fast", recipe, reader=reader)
            self.compare(base, fast)
            fast.close()
            base.close()

    def test_network_failure_replays_only_uncommitted_batch(self):
        recipe, original = fixture()
        failed = []

        def reader(source, shard, start):
            for index, row in original(source, shard, start):
                if index == 1 and not failed:
                    failed.append(True)
                    raise ConnectionError("injected disconnect")
                yield index, row

        with tempfile.TemporaryDirectory() as temporary:
            corpus = FastRawCorpus(temporary, recipe, batch_size=3, reader=reader)
            source = recipe["sources"][0]
            corpus._write(source, corpus._fetch(source, (0, 0)))
            self.assertEqual(corpus.cursor(source["name"]), (0, 3))
            self.assertEqual(corpus.db.execute("SELECT SUM(count) FROM decisions").fetchone()[0], 3)
            corpus.close()

    def test_transaction_failure_rolls_back_documents_and_cursor(self):
        recipe, reader = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            corpus = FastRawCorpus(temporary, recipe, batch_size=3, reader=reader)
            source = recipe["sources"][0]
            batch = corpus._fetch(source, corpus.cursor(source["name"]))
            corpus.db.execute(
                "CREATE TRIGGER fail_write BEFORE INSERT ON decisions BEGIN SELECT RAISE(ABORT,'injected'); END"
            )
            with self.assertRaises(sqlite3.IntegrityError):
                corpus._write(source, batch)
            self.assertEqual(corpus.cursor(source["name"]), (0, 0))
            self.assertEqual(corpus.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 0)
            self.assertEqual(corpus.counts(source["name"]), {})
            corpus.db.execute("DROP TRIGGER fail_write")
            corpus._write(source, corpus._fetch(source, corpus.cursor(source["name"])))
            self.assertEqual(corpus.cursor(source["name"]), (0, 3))
            corpus.close()


if __name__ == "__main__":
    unittest.main()
