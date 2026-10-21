import tempfile
import unittest
import zlib

from training_data.raw_corpus import RawCorpus
from training_data.simple_llm import adapt


class Encoding:
    ids = (8, 9, 10)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return Encoding()


class RawCorpusTests(unittest.TestCase):
    def test_raw_storage_resume_dedup_and_conflicting_splits(self):
        source = {
            "name": "fineweb_edu",
            "weight_percent": 100,
            "files": ["a"],
            "repo": "test",
            "revision": "0" * 40,
        }
        rows = []
        for split in ("train", "validation"):
            for i in range(100):
                row = {"id": str(i), "text": "same text"}
                if adapt(source, row)[0]["split"] == split:
                    rows.append(row)
                    break

        def reader(source, shard, start):
            yield from enumerate(rows[start:], start)

        with tempfile.TemporaryDirectory() as folder:
            corpus = RawCorpus(folder, {"sources": [source]}, Tokenizer())
            self.assertEqual(corpus.prepare("fineweb_edu", max_records=1, reader=reader), 1)
            self.assertEqual(corpus.counts("fineweb_edu")["train"], 4)
            raw = corpus.db.execute("SELECT text FROM documents").fetchone()[0]
            self.assertEqual(zlib.decompress(raw), b"same text")
            corpus.close()
            corpus = RawCorpus(folder, {"sources": [source]}, Tokenizer())
            self.assertEqual(corpus.prepare("fineweb_edu", reader=reader), 1)
            self.assertEqual(corpus.counts("fineweb_edu"), {})
            self.assertEqual(corpus.prepare("fineweb_edu", reader=reader), 0)
            report = corpus.report(1000)
            self.assertTrue(report["sources"]["fineweb_edu"]["upstream_exhausted"])
            self.assertFalse(report["training_ready"])
            self.assertEqual(report["supported_mixture_exposures"], 0)
            corpus.close()


if __name__ == "__main__":
    unittest.main()
