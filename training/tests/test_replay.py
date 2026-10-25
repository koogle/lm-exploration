import json
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

from training_data.replay import Replay
from training_data.subword import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    LOSS_IGNORE_INDEX,
    TOKENIZER_SHA256,
    load_frozen,
)

DOCUMENTS = (
    ("code", "train", "def add(a, b):\n    return a + b\n"),
    ("web", "train", "The cat sat on the mat."),
    ("code", "validation", "print('hello world')\n"),
    ("web", "train", "A second web page about rivers."),
    ("web", "validation", "Validation text about mountains and valleys."),
    ("code", "train", "x = [i * i for i in range(10)]\n"),
)


def snapshot(folder):
    folder = Path(folder)
    db = sqlite3.connect(folder / "raw.sqlite")
    db.execute(
        "CREATE TABLE documents (hash TEXT, source TEXT, split TEXT, text BLOB, conflict INTEGER)"
    )
    for index, (source, split, text) in enumerate(DOCUMENTS):
        db.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, 0)",
            (str(index), source, split, zlib.compress(text.encode())),
        )
    db.commit()
    db.close()
    manifest = {
        "tokenizer_sha256": TOKENIZER_SHA256,
        "database": {"path": "raw.sqlite"},
        "sources": {"code": {"weight_percent": 50}, "web": {"weight_percent": 50}},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.tokenizer = load_frozen()
        self.snapshot = snapshot(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def replay(self, weights=None, context=16):
        return Replay(self.snapshot, weights or {"code": 1, "web": 1}, context, self.tokenizer, 0)

    def test_tokens_pack_documents_of_one_source_and_wrap(self):
        replay = self.replay()
        tokens = replay.tokens("code", "train", 0, 200)
        encoded = [
            [BOS_TOKEN_ID, *self.tokenizer.encode(text, add_special_tokens=False).ids, EOS_TOKEN_ID]
            for source, split, text in DOCUMENTS
            if source == "code" and split == "train"
        ]
        cycle = encoded[0] + encoded[1]
        self.assertEqual(tokens, (cycle * (200 // len(cycle) + 1))[:200])

    def test_windows_mask_only_bos_targets(self):
        window = self.replay().sample()
        self.assertEqual(len(window["x"]), 16)
        self.assertEqual(len(window["y"]), 16)
        for token, label in zip(window["x"][1:], window["y"], strict=False):
            self.assertEqual(label, LOSS_IGNORE_INDEX if token == BOS_TOKEN_ID else token)
        self.assertIn(window["bucket"], ("code", "web"))

    def test_sampling_follows_weights_and_seed(self):
        replay = self.replay({"web": 1})
        self.assertEqual({replay.sample()["bucket"] for _ in range(10)}, {"web"})
        self.assertEqual(self.replay().sample(), self.replay().sample())

    def test_validation_windows_are_fixed_per_source(self):
        replay = self.replay(context=3)
        windows = replay.validation(2)
        self.assertEqual([w["bucket"] for w in windows], ["code", "code", "web", "web"])
        self.assertEqual(windows, self.replay(context=3).validation(2))
        tokens = replay.tokens("code", "validation", 0, 7)
        self.assertEqual(windows[1]["x"], tokens[3:6])

    def test_rejects_unknown_sources_and_short_validation(self):
        with self.assertRaises(ValueError):
            self.replay({"books": 1})
        with self.assertRaises(ValueError):
            self.replay(context=64).validation(10)


if __name__ == "__main__":
    unittest.main()
