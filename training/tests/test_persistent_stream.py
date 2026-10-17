import tempfile
import unittest
from pathlib import Path

from training_data.persistent_stream import PersistentStream
from training_data.raw_corpus import RawCorpus


class PersistentStreamTests(unittest.TestCase):
    def test_chunks_keep_stream_and_preserve_database(self):
        source = {
            "name": "fineweb_edu",
            "files": ["fixture"],
            "repo": "test",
            "revision": "0" * 40,
            "weight_percent": 100,
        }
        rows = [{"id": str(i), "text": f"Distinct raw document {i}"} for i in range(30)]
        opens = []
        closed = []

        def reader(source, shard, start):
            opens.append(start)
            try:
                yield from enumerate(rows[start:], start)
            finally:
                closed.append(start)

        with tempfile.TemporaryDirectory() as temporary:
            base = RawCorpus(Path(temporary) / "base", {"sources": [source]})
            base.prepare("fineweb_edu", max_records=100, reader=reader)
            opens.clear()
            stream = PersistentStream(reader)
            fast = RawCorpus(Path(temporary) / "fast", {"sources": [source]})
            for _ in range(15):
                fast.prepare("fineweb_edu", max_records=2, reader=stream)
            self.assertEqual(opens, [0])
            self.assertEqual(fast.prepare("fineweb_edu", reader=stream), 0)
            for table in ("documents", "cursors", "decisions"):
                self.assertEqual(
                    base.db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall(),
                    fast.db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall(),
                )
            stream.close()
            base.close()
            fast.close()

    def test_rewind_reopens_at_saved_cursor(self):
        opens = []

        def reader(source, shard, start):
            opens.append(start)
            yield from ((i, {"id": i}) for i in range(start, 5))

        stream = PersistentStream(reader)
        chunk = stream({}, "a", 0)
        self.assertEqual(next(chunk)[0], 0)
        self.assertEqual(next(chunk)[0], 1)
        chunk.close()
        chunk = stream({}, "a", 1)  # Simulate a failed row transaction.
        self.assertEqual(next(chunk)[0], 1)
        chunk.close()
        self.assertEqual(opens, [0, 1])
        stream.close()


if __name__ == "__main__":
    unittest.main()
