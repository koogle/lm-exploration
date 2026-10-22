import tempfile
import unittest
import zlib
from pathlib import Path

import torch
from training_data.export_raw import export_raw
from training_data.mixture import open_mixture
from training_data.raw_corpus import RawCorpus
from training_data.raw_token_stream import RawTokenMixture
from training_data.simple_llm import adapt
from training_data.subword import LOSS_IGNORE_INDEX


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


class RawTokenStreamTests(unittest.TestCase):
    def test_resume_inside_document_and_batch_rollback(self):
        source = {
            "name": "fineweb_edu",
            "weight_percent": 100,
            "files": ["a"],
            "repo": "test",
            "revision": "0" * 40,
        }
        rows = []
        for i in range(20):
            row = {"id": str(i), "text": ("hello world this is a test sentence " * (i + 1))}
            if adapt(source, row)[0]["split"] == "train":
                rows.append(row)

        def reader(source, shard, start):
            yield from enumerate(rows[start:], start)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus = RawCorpus(root, {"sources": [source]})
            corpus.prepare("fineweb_edu", reader=reader)
            counts = corpus.counts("fineweb_edu")
            corpus.close()
            snapshot = root / "snapshot"
            export_raw(root, snapshot)
            path = snapshot / "manifest.json"
            with self.assertRaises(ValueError):
                RawTokenMixture(path)
            with self.assertRaisesRegex(ValueError, "Seekable token snapshot required"):
                open_mixture(path, loader="token_shards", context=8, allow_unreviewed=True)
            a = open_mixture(path, loader="raw", context=8, allow_unreviewed=True)
            a.next_batch(3)
            state = a.state_dict()
            expected = a.next_batch(5)
            b = open_mixture(path, loader="raw", context=8, allow_unreviewed=True)
            b.load_state_dict(state)
            actual = b.next_batch(5)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected, actual)))
            self.assertEqual(a.state_dict(), b.state_dict())
            # The final partial batch is padded; afterwards the cursor no longer moves.
            _, last = b.next_batch(100000)
            self.assertTrue((last[-1] == LOSS_IGNORE_INDEX).all())
            saved = b.state_dict()
            with self.assertRaises(StopIteration):
                b.next_batch(4)
            self.assertEqual(saved, b.state_dict())
            a.close()
            b.close()
            c = RawTokenMixture(path, context=8, allow_unreviewed=True)
            total = 0
            while True:
                try:
                    total += c.next_block()["loss_tokens"]
                except StopIteration:
                    break
            self.assertEqual(total, counts["train"] * 2)
            c.close()

            # Exported pass caps and exact targets bound one full read of the source.
            once = export_raw(
                root,
                root / "once",
                max_passes={"fineweb_edu": 1},
                target_tokens=counts["train"],
            )
            entry = once["sources"]["fineweb_edu"]
            self.assertEqual((entry["max_passes"], entry["target_tokens"]), (1, counts["train"]))
            d = RawTokenMixture(root / "once/manifest.json", context=8, allow_unreviewed=True)
            _, y = d.next_batch(100000)
            self.assertEqual(int((y != LOSS_IGNORE_INDEX).sum()), counts["train"])
            d.close()
            with self.assertRaisesRegex(ValueError, "unknown source"):
                export_raw(root, root / "bad", max_passes={"missing": 1})
            self.assertFalse((root / "bad").exists())
