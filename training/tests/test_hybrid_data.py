"""Hybrid data preparation, contamination screen and the prepared-shard loader."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from model_hybrid import data

EOS = 99


class CharTokenizer:
    """One id per character, enough to exercise preparation without GLM files."""

    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode_batch(self, texts, add_special_tokens=False):
        return [self.Encoding([ord(c) % 90 for c in text]) for text in texts]


def recipe(shards=3):
    return {
        "name": "test-recipe",
        "context": 8,
        "seed": 7,
        "sources": {
            "web": {
                "weight": 0.75,
                "repo": "w",
                "revision": "r",
                "pattern": "web-{index}.parquet",
                "shards": shards,
                "column": "text",
                "max_passes": 1,
            },
            "code": {
                "weight": 0.25,
                "repo": "c",
                "revision": "r",
                "pattern": "code-{index}.parquet",
                "shards": shards,
                "column": "code",
                "licenses": ["mit"],
                "max_passes": 1,
            },
        },
        "contamination": {"ngram": 13, "benchmarks": []},
    }


class HybridDataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.recipe = recipe()
        for i in range(3):
            web = [f"web document {i}-{j} " * (j % 4 + 1) for j in range(30)] + ["   "]
            pq.write_table(pa.table({"text": web}), self.root / f"web-{i}.parquet")
            code = [f"def f{i}_{j}(): return {j}" for j in range(20)]
            licenses = ["mit" if j % 2 else "gpl-3.0" for j in range(20)]
            pq.write_table(
                pa.table({"code": code, "license": licenses}), self.root / f"code-{i}.parquet"
            )
        self.download = lambda repo, file, revision: self.root / file
        self.tokenizer = CharTokenizer()

    def tearDown(self):
        self.folder.cleanup()

    def prepare(self, screen=None):
        screen = np.empty(0, np.uint64) if screen is None else screen
        return [
            data.prepare_shard(
                self.recipe, name, i, self.root / "out", self.tokenizer, screen, self.download, EOS
            )
            for name in ("web", "code")
            for i in range(3)
        ]

    def test_ngram_hashes_and_screen(self):
        ids = list(range(20))
        hashes = data.ngram_hashes(ids, 13)
        self.assertEqual(len(hashes), 8)
        self.assertEqual(len(data.ngram_hashes(ids[:12], 13)), 0)
        screen = np.unique(data.ngram_hashes(list(range(5, 18)), 13))
        self.assertTrue(data.contaminated(ids, screen, 13))
        self.assertFalse(data.contaminated(list(range(100, 130)), screen, 13))

    def test_prepared_shards(self):
        metas = self.prepare()
        web = metas[0]
        self.assertEqual(web["documents"], 30)
        self.assertEqual(web["dropped"]["empty"], 1)
        self.assertEqual(metas[3]["dropped"]["license"], 10)
        bin_path, docs_path, meta_path = data.shard_paths(self.root / "out", "web", 0)
        tokens = np.fromfile(bin_path, dtype=np.uint32)
        lengths = np.load(docs_path)
        self.assertEqual(len(tokens), lengths[:, 0].sum())
        ends = np.cumsum(lengths[:, 0]) - 1
        self.assertTrue((tokens[ends] == EOS).all())
        self.assertEqual((tokens == EOS).sum(), 30)
        self.assertEqual(json.loads(meta_path.read_text())["tokens"], len(tokens))
        self.assertTrue(metas[2]["heldout"])

    def test_contaminated_documents_are_dropped(self):
        # Only document 0-17 contains "ment 0-17 web" (13 characters).
        ids = self.tokenizer.encode_batch(["ment 0-17 web"])[0].ids
        metas = self.prepare(np.unique(data.ngram_hashes(ids, 13)))
        self.assertEqual(metas[0]["dropped"]["contaminated"], 1)
        self.assertEqual(metas[0]["documents"], 29)

    def test_mixture_weights_order_and_resume(self):
        self.prepare()
        mixture = data.PreparedMixture(self.recipe, self.root / "out")
        self.assertEqual(sorted(mixture.orders["web"]), [0, 1])
        inputs, labels, names = mixture.next_batch(8)
        self.assertEqual(inputs.shape, (8, 8))
        np.testing.assert_array_equal(inputs[:, 1:], labels[:, :-1])
        self.assertEqual(names.count("web"), 6)
        state = json.loads(json.dumps(mixture.state_dict()))
        expected = mixture.next_batch(4)
        resumed = data.PreparedMixture(self.recipe, self.root / "out")
        resumed.load_state_dict(state)
        actual = resumed.next_batch(4)
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[2], expected[2])

    def test_stream_is_contiguous_across_shards(self):
        self.prepare()
        mixture = data.PreparedMixture(self.recipe, self.root / "out")
        stream = np.concatenate(
            [
                np.fromfile(data.shard_paths(self.root / "out", "code", i)[0], dtype=np.uint32)
                for i in mixture.orders["code"]
            ]
        )
        windows = []
        while True:
            try:
                name, window = mixture.next_window()
            except StopIteration:
                break
            if name == "code":
                windows.append(window)
        rebuilt = np.concatenate([windows[0]] + [w[1:] for w in windows[1:]])
        np.testing.assert_array_equal(rebuilt, stream[: len(rebuilt)])

    def test_missing_shard_waits_then_fails(self):
        mixture = data.PreparedMixture(self.recipe, self.root / "out", wait_seconds=0)
        with self.assertRaises(TimeoutError):
            mixture.next_window()

    def test_heldout_windows(self):
        self.prepare()
        _, labels, total_bytes = data.heldout_windows(self.recipe, self.root / "out", "web", 8, 5)
        lengths = np.load(data.shard_paths(self.root / "out", "web", 2)[1])[:5]
        self.assertEqual(total_bytes, lengths[:, 1].sum())
        self.assertEqual((labels != -100).sum(), lengths[:, 0].sum() - 1)


if __name__ == "__main__":
    unittest.main()
