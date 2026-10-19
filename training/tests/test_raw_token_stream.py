import json
import tempfile
import unittest
from pathlib import Path

import torch
from training_data.export_raw import export_raw
from training_data.raw_corpus import RawCorpus
from training_data.raw_token_stream import RawTokenMixture
from training_data.scholarly import adapt


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
            a = RawTokenMixture(path, context=8, allow_unreviewed=True)
            a.next_batch(3)
            state = a.state_dict()
            expected = a.next_batch(5)
            b = RawTokenMixture(path, context=8, allow_unreviewed=True)
            b.load_state_dict(state)
            actual = b.next_batch(5)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected, actual)))
            self.assertEqual(a.state_dict(), b.state_dict())
            saved = b.state_dict()
            with self.assertRaises(StopIteration):
                b.next_batch(100000)
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

            # One-pass source caps survive restore and never silently wrap.
            manifest = json.loads(path.read_text())
            manifest["sources"]["fineweb_edu"]["max_passes"] = 1
            path.write_text(json.dumps(manifest))
            d = RawTokenMixture(path, context=8, allow_unreviewed=True)
            first = d.next_block()
            state = d.state_dict()
            e = RawTokenMixture(path, context=8, allow_unreviewed=True)
            e.load_state_dict(state)
            total = first["loss_tokens"] + sum(
                block["loss_tokens"] for block in e.blocks("fineweb_edu")
            )
            self.assertEqual(total, counts["train"])
            bad = d.state_dict()
            bad["raw_cursors"]["fineweb_edu"]["pass"] = 1
            with self.assertRaisesRegex(ValueError, "Invalid raw cursor"):
                d.load_state_dict(bad)
            d.close()
            e.close()


if __name__ == "__main__":
    unittest.main()
