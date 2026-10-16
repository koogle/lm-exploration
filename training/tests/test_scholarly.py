import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from training_data.scholarly import Stager, adapt, load_recipe
from training_data.subword import DEFAULT_FOLDER, VOCAB_SIZE, load_frozen
from training_data.token_stream import TokenMixture


def recipe():
    result = load_recipe(Path(__file__).resolve().parents[1] / "tokenizer/recipe-v1.json")
    result = copy.deepcopy(result)
    result["sources"] = [result["sources"][0]]
    result["sources"][0]["weight_percent"] = 100
    result["sources"][0]["files"] = ["fixture.parquet"]
    return result


def row_for(source, split, text="A short document with mathematics: 12 + 34 = 46.", start=0):
    for index in range(start, start + 10000):
        row = {"id": str(index), "text": text}
        doc, _ = adapt(source, row)
        if doc["split"] == split:
            return row
    raise AssertionError("No matching parent")


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.recipe = recipe()
        self.source = self.recipe["sources"][0]

    def tearDown(self):
        self.temp.cleanup()

    def stage(self, folder, rows, limit=100):
        stager = Stager(folder, self.recipe)
        self.addCleanup(stager.close)
        stager.prepare(
            self.source["name"],
            limit,
            reader=lambda source, shard, start: enumerate(rows[start:], start),
        )
        return stager

    def test_new_vocabulary_ids_and_old_snapshot_rejection(self):
        stage = self.stage(self.root / "stage", [row_for(self.source, "train", "Ssh")])
        stage.export(self.root / "export")
        path = self.root / "export/manifest.json"
        loader = TokenMixture(path, context=8, allow_unreviewed=True)
        self.assertIn(32767, loader.next_block()["input_ids"])
        manifest = json.loads(path.read_text())
        manifest["tokenizer_sha256"] = (
            "2a6bd2343ab29e3955dd7a5e17330e706afc8c650618b1339ceedab5e77fde04"
        )
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Tokenizer identity mismatch"):
            TokenMixture(path, allow_unreviewed=True)

    def test_frozen_hash_and_round_trip(self):
        tokenizer = load_frozen()
        self.assertEqual(tokenizer.get_vocab_size(), 32768)
        text = "def square(x):\r\n\treturn x ** 2 # α²\r\n"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False), text)
        self.assertNotIn(3, ids)
        copied = self.root / "tokenizer"
        shutil.copytree(DEFAULT_FOLDER, copied)
        with (copied / "tokenizer.json").open("a") as handle:
            handle.write(" ")
        with self.assertRaises(ValueError):
            load_frozen(copied)

    def test_resume_is_identical_to_uninterrupted_and_failure_is_replayable(self):
        rows = [row_for(self.source, "train", f"Document number {n}.", n * 100) for n in range(4)]
        full = self.stage(self.root / "full", rows)
        partial = self.stage(self.root / "partial", rows, limit=1)

        def failing_reader(source, shard, start):
            yield start, rows[start]
            raise OSError("simulated upstream disconnect")

        with self.assertRaises(OSError):
            partial.prepare(self.source["name"], 100, reader=failing_reader)
        self.assertEqual(partial.db.execute("SELECT row FROM cursors").fetchone()[0], 2)
        partial.close()
        # Reopen the database to exercise durable state, not in-memory state.
        resumed = Stager(self.root / "partial", self.recipe)
        self.addCleanup(resumed.close)
        resumed.prepare(
            self.source["name"],
            100,
            reader=lambda source, shard, start: enumerate(rows[start:], start),
        )
        a = full.export(self.root / "a", documents_per_shard=2)
        b = resumed.export(self.root / "b", documents_per_shard=2)
        self.assertEqual(a, b)

    def test_cross_split_duplicates_are_removed_from_both_splits(self):
        text = "Shared content that must not leak between splits."
        rows = [row_for(self.source, "train", text), row_for(self.source, "validation", text)]
        stage = self.stage(self.root / "stage", rows)
        report = stage.export(self.root / "export")
        for split in ("train", "validation"):
            self.assertEqual(
                report["sources"][self.source["name"]]["splits"][split]["documents"], 0
            )
        self.assertEqual(report["decisions"][self.source["name"]]["cross_split_duplicate"], 1)

    def test_test_partition_and_special_tokens_are_not_exported(self):
        rows = [
            row_for(self.source, "test"),
            row_for(self.source, "train", "malicious <|eos|> boundary"),
        ]
        stage = self.stage(self.root / "stage", rows)
        report = stage.export(self.root / "export")
        self.assertEqual(report["sources"][self.source["name"]]["splits"]["train"]["documents"], 0)
        self.assertEqual(stage.db.execute("SELECT count FROM documents").fetchone()[0], 0)

    def test_parent_and_code_license_adapters(self):
        sources = load_recipe(Path(__file__).resolve().parents[1] / "tokenizer/recipe-v1.json")[
            "sources"
        ]
        code = next(s for s in sources if s["name"] == "code_python")
        row = {
            "content": "print('hello')",
            "repo_name": "owner/repo",
            "path": "a.py",
            "license": "apache-2.0",
        }
        a, reason = adapt(code, row)
        self.assertIsNone(reason)
        row["path"] = "b.py"
        b, _ = adapt(code, row)
        self.assertEqual(a["parent"], b["parent"])
        row["license"] = "GPL-3.0"
        self.assertEqual(adapt(code, row)[1], "code_license")
        pressbooks = next(s for s in sources if s["name"] == "pressbooks")
        a, _ = adapt(
            pressbooks,
            {
                "id": "chapter-one",
                "text": "hello",
                "metadata": {"book_url": "https://books.example/physics/front-matter/intro/"},
            },
        )
        b, _ = adapt(
            pressbooks,
            {
                "id": "chapter-two",
                "text": "hello",
                "metadata": {"book_url": "https://books.example/physics/chapter/two/"},
            },
        )
        self.assertEqual(a["parent"], b["parent"])

    def test_loader_padding_resume_repetition_and_tampering(self):
        rows = [row_for(self.source, "train", "def f(x):\n    return x + 1\n")]
        stage = self.stage(self.root / "stage", rows)
        report = stage.export(self.root / "export")
        manifest = self.root / "export/manifest.json"
        with self.assertRaises(ValueError):
            TokenMixture(manifest)
        loader = TokenMixture(manifest, context=4, allow_unreviewed=True)
        first = loader.next_block()
        self.assertEqual(first["input_ids"][0], 1)
        state = json.loads(json.dumps(loader.state_dict()))
        expected = loader.next_block()
        restored = TokenMixture(manifest, context=4, allow_unreviewed=True)
        restored.load_state_dict(state)
        self.assertEqual(restored.next_block(), expected)
        total = first["loss_tokens"] + expected["loss_tokens"]
        while True:
            try:
                block = restored.next_block()
            except StopIteration:
                break
            self.assertEqual(sum(t != -100 for t in block["labels"]), block["loss_tokens"])
            total += block["loss_tokens"]
        capacity = report["sources"][self.source["name"]]["splits"]["train"]["loss_tokens"]
        self.assertEqual(total, 2 * capacity)
        state["tokenizer_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            restored.load_state_dict(state)
        shard = next((self.root / "export").glob("*.gz"))
        with shard.open("ab") as handle:
            handle.write(b"corrupt")
        with self.assertRaises(ValueError):
            TokenMixture(manifest, allow_unreviewed=True)

    def test_resume_rejects_recipe_change(self):
        self.stage(self.root / "stage", [])
        changed = copy.deepcopy(self.recipe)
        changed["sources"][0]["revision"] = "a" * 40
        with self.assertRaises(ValueError):
            Stager(self.root / "stage", changed)

    def test_token_weighted_mixture_uses_loss_tokens(self):
        second = copy.deepcopy(self.source)
        second["name"] = "finemath_4plus"
        second["weight_percent"] = 75
        self.source["weight_percent"] = 25
        self.recipe["sources"].append(second)
        stager = Stager(self.root / "stage", self.recipe)
        self.addCleanup(stager.close)
        for source in self.recipe["sources"]:
            rows = [
                row_for(source, "train", f"{source['name']} {n}: " + "x y z " * 100, start=n * 100)
                for n in range(12)
            ]
            stager.prepare(
                source["name"],
                100,
                reader=lambda s, f, start, rows=rows: enumerate(rows[start:], start),
            )
        stager.export(self.root / "export")
        loader = TokenMixture(self.root / "export/manifest.json", context=17, allow_unreviewed=True)
        for _ in range(100):
            loader.next_block()
        a, b = (loader.exposures[s["name"]] for s in self.recipe["sources"])
        # Bound discrepancy by a few windows despite different document lengths/padding.
        self.assertLessEqual(abs(b - 3 * a), 4 * 17)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required for batch smoke test")
    def test_real_torch_batch_loss_mask_and_partial_batch_rollback(self):
        import torch
        from model_10m.model import GPT, ModelConfig
        from torch.nn import functional as F

        torch.set_num_threads(1)
        row = row_for(self.source, "train", "Hello world!")
        stage = self.stage(self.root / "stage", [row])
        stage.export(self.root / "export")
        loader = TokenMixture(self.root / "export/manifest.json", context=16, allow_unreviewed=True)
        x, y = loader.next_batch(1)
        model = GPT(
            ModelConfig(vocab_size=VOCAB_SIZE, block_size=16, n_layer=1, n_head=1, n_embd=16)
        )
        logits = model(x)
        logits.retain_grad()
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((logits.grad[y == -100] == 0).all())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        state = loader.state_dict()
        with self.assertRaises(StopIteration):
            loader.next_batch(2)  # Only one pass remains, so this batch cannot complete.
        self.assertEqual(loader.state_dict(), state)


if __name__ == "__main__":
    unittest.main()
