import json
import tempfile
import unittest
from pathlib import Path

import torch
from training_data.live_stream import LiveTokenMixture, SplitConflict
from training_data.scholarly import SPLIT_VERSION, adapt
from training_data.subword import TOKENIZER_SHA256


def fixture(folder):
    sources = [
        {
            "name": n,
            "weight_percent": 50,
            "repo": "fixture",
            "revision": "0" * 40,
            "format": "json.gz",
            "files": ["a"],
        }
        for n in ("fineweb_edu", "finemath_4plus")
    ]
    rows = {s["name"]: [] for s in sources}
    for s in sources:
        for i in range(100):
            row = {
                "id": str(i),
                "text": f"{s['name']} example {i}. " + "Lots of content tokens. " * 8,
            }
            if adapt(s, row)[0]["split"] == "train":
                rows[s["name"]].append(row)
    manifest = {
        "format": "scholarly-live-v1",
        "recipe": {"sources": sources},
        "max_passes": 2,
        "split_policy": SPLIT_VERSION,
        "tokenizer_sha256": TOKENIZER_SHA256,
    }
    path = Path(folder) / "manifest.json"
    path.write_text(json.dumps(manifest))

    def reader(source, shard, start):
        yield from enumerate(rows[source["name"]][start:], start)

    return path, reader, rows


class LiveStreamTests(unittest.TestCase):
    def test_resume_inside_document_and_two_pass_ceiling(self):
        with tempfile.TemporaryDirectory() as folder:
            path, reader, rows = fixture(folder)
            a = LiveTokenMixture(path, context=16, allow_unreviewed=True, reader=reader)
            a.next_batch(7)
            state = a.state_dict()
            expected = a.next_batch(5)
            b = LiveTokenMixture(path, context=16, allow_unreviewed=True, reader=reader)
            b.load_state_dict(state)
            actual = b.next_batch(5)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected, actual)))
            self.assertEqual(a.state_dict(), b.state_dict())
            before = b.state_dict()
            with self.assertRaises(StopIteration):
                b.next_batch(100000)
            self.assertEqual(before, b.state_dict())
            a.close()
            b.close()
            c = LiveTokenMixture(path, context=16, allow_unreviewed=True, reader=reader)
            count = 0
            for block in c.blocks("fineweb_edu"):
                count += block["loss_tokens"]
            expected_count = 2 * sum(
                len(c.tokenizer.encode(r["text"], add_special_tokens=False).ids) + 1
                for r in rows["fineweb_edu"]
            )
            self.assertEqual(count, expected_count)
            c.close()

    def test_split_conflict_stops_instead_of_changing_mix(self):
        with tempfile.TemporaryDirectory() as folder:
            path, reader, rows = fixture(folder)
            source = json.loads(path.read_text())["recipe"]["sources"][0]
            text = rows["fineweb_edu"][0]["text"]
            validation = next(
                {"id": str(i), "text": text}
                for i in range(100)
                if adapt(source, {"id": str(i), "text": text})[0]["split"] == "validation"
            )
            rows["fineweb_edu"] = [rows["fineweb_edu"][0], validation]
            c = LiveTokenMixture(path, context=2048, allow_unreviewed=True, reader=reader)
            iterator = c.blocks("fineweb_edu")
            next(iterator)
            with self.assertRaises(SplitConflict):
                next(iterator)
            c.close()


if __name__ == "__main__":
    unittest.main()
