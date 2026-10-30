import json
import math
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import helpers  # noqa: F401  (sets sys.path for the project modules below)
from helpers import tiny_shard

# isort: split
import jobs
import numpy as np
import prepare
import student
import torch
from common import QWEN_VOCAB


class LossTest(unittest.TestCase):
    def test_soft_cross_entropy_matches_explicit_partition(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 11)
        ids = torch.tensor([[1, 4, -1], [2, 2, 7], [0, 5, 9]])
        lp = torch.log(
            torch.tensor([[0.5, 0.3, 0.0], [0.2, 0.3, 0.1], [0.6, 0.2, 0.15]])
        )
        soft, entropy = student.soft_cross_entropy(logits, ids, lp)
        q = logits.softmax(-1)
        for row in range(3):
            p = {}
            for i, value in zip(ids[row].tolist(), lp[row].exp().tolist()):
                if i >= 0:
                    p[i] = p.get(i, 0) + value
            rest = 1 - sum(p.values())
            q_rest = 1 - sum(q[row, i] for i in p)
            expected = -sum(
                v * math.log(q[row, i]) for i, v in p.items()
            ) - rest * math.log(q_rest)
            h = -sum(v * math.log(v) for v in p.values()) - rest * math.log(rest)
            self.assertAlmostEqual(float(soft[row]), expected, places=5)
            self.assertAlmostEqual(float(entropy[row]), h, places=5)

    def test_chunk_cross_entropy_matches_torch_with_ignored_labels(self):
        torch.manual_seed(1)

        class Head:
            lm_head = torch.nn.Linear(8, 13, bias=False)

        features = torch.randn(6, 8)
        labels = torch.tensor([3, -100, 0, 12, -100, 7])
        ids = torch.randint(0, 13, (6, 4))
        lps = torch.log(torch.full((6, 4), 0.2))
        ce, soft, _ = student.chunk_losses(Head, features, labels, ids, lps)
        expected = torch.nn.functional.cross_entropy(
            Head.lm_head(features), labels, ignore_index=-100, reduction="sum"
        )
        self.assertAlmostEqual(float(ce), float(expected), places=4)
        rows, _ = student.soft_cross_entropy(Head.lm_head(features), ids, lps)
        self.assertAlmostEqual(float(soft), float(rows[labels != -100].sum()), places=4)

    def test_kl_is_zero_when_student_matches_teacher(self):
        logits = torch.log(torch.tensor([[0.5, 0.25, 0.125, 0.125]]))
        soft, entropy = student.soft_cross_entropy(
            logits, torch.tensor([[0, 1]]), torch.log(torch.tensor([[0.5, 0.25]]))
        )
        self.assertAlmostEqual(float(soft - entropy), 0.0, places=6)


class StreamTest(unittest.TestCase):
    def test_packing_inserts_bos_and_ignores_next_document_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard = tiny_shard(tmp, [np.array([5, 6, 7]), np.array([8, 9])])
            vocab = student.Vocabulary.full(64, bos=1)
            inputs, labels, ids, lps, native = student.pack_shard(shard, vocab)
            np.testing.assert_array_equal(inputs, [1, 5, 6, 7, 1, 8, 9])
            np.testing.assert_array_equal(labels, [5, 6, 7, -100, 8, 9, -100])
            np.testing.assert_array_equal(ids[:, 0], [5, 6, 7, -1, 8, 9, -1])
            self.assertTrue(np.isneginf(lps[3]).all())
            np.testing.assert_array_equal(native, [5, 6, 7, -1, 8, 9, -1])

    def test_stream_windows_cross_shards_without_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = tiny_shard(Path(tmp) / "a", [np.arange(10, 15)])
            b = tiny_shard(Path(tmp) / "b", [np.arange(20, 24)])
            windows = list(
                student.Stream([a, b], student.Vocabulary.full(64, bos=1), 4)
            )
            flat = np.concatenate([w[0] for w in windows])
            np.testing.assert_array_equal(
                flat, [1, 10, 11, 12, 13, 14, 1, 20]
            )  # partial tail dropped


class ModelTest(unittest.TestCase):
    def model(self, vocab=40):
        torch.manual_seed(0)
        config = student.ModelConfig(
            vocab_size=vocab, block_size=64, n_layer=2, n_head=2, n_embd=32, eos_id=7
        )
        return student.GPT(config).eval()

    def test_training_reduces_loss_on_repeated_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard = tiny_shard(
                Path(tmp) / "s", [np.array([5, 6, 7, 8, 9, 10, 11])] * 600
            )
            vocab = student.Vocabulary.full(40, bos=1)
            settings = {
                "seed": 0,
                "window": 32,
                "size": "small",
                "lr": 3e-3,
                "batch_windows": 4,
                "micro_windows": 2,
                "positions": 4 * 32 * 30,
                "kd_weight": 0.5,
                "log_every": 10,
            }
            student.SIZES["tiny"] = {"n_layer": 1, "n_head": 2, "n_embd": 32}
            settings["size"] = "tiny"
            logs = []
            model, summary = student.train(
                settings, [shard], vocab, 7, tmp, logs.append, device="cpu"
            )
            steps = [entry for entry in logs if entry["event"] == "step"]
            self.assertLess(steps[-1]["ce"], steps[0]["ce"] / 2)
            self.assertEqual(summary["steps"], 30)
            metrics = student.validate(
                model,
                [shard],
                vocab,
                np.ones(64, dtype=np.int64),
                256,
                window=32,
                device="cpu",
            )
            self.assertGreater(metrics["labels"], 0)
            self.assertAlmostEqual(
                metrics["ce_per_byte"] * metrics["bytes"],
                metrics["ce_per_token"] * metrics["labels"],
                places=3,
            )


class CoveringVocabularyTest(unittest.TestCase):
    def fake_tokenizer(self):
        decoder = prepare.byte_decoder()
        vocab = {char: byte for char, byte in decoder.items()}
        for index, token in enumerate(["ab", "abc", "Ġsolve", "xyz"], start=256):
            vocab[token] = index
        return {
            "model": {"vocab": vocab},
            "added_tokens": [{"id": 300, "content": "<|end|>"}],
        }

    def test_dropped_tokens_are_spelled_from_kept_pieces(self):
        tokenizer = self.fake_tokenizer()
        counts = np.zeros(QWEN_VOCAB, dtype=np.int64)
        counts[[256, 258, 300]] = [9, 5, 7]  # "ab", "Ġsolve", "<|end|>" are frequent
        kept, pieces, offsets, lengths = prepare.build_covering_vocab(
            counts, 3 + 256 + 3, tokenizer
        )
        self.assertEqual(kept.size, 259)
        vocab = student.Vocabulary.covering(kept, pieces, offsets, lengths)
        strings = {i: s for s, i in tokenizer["model"]["vocab"].items()}
        strings[300] = "<|end|>"
        for native in (257, 259, 300):  # "abc" and "xyz" were dropped
            expanded, _ = vocab.expand([native])
            spelled = "".join(strings[i] for i in vocab.decode_native(expanded))
            self.assertEqual(spelled, strings[native])
        self.assertEqual(lengths[257], 2)  # "ab" + "c"
        self.assertEqual(lengths[256], 1)

    def test_packing_expands_targets_and_document_boundaries(self):
        tokenizer = self.fake_tokenizer()
        counts = np.zeros(QWEN_VOCAB, dtype=np.int64)
        counts[[256, 300]] = [9, 7]
        kept, pieces, offsets, lengths = prepare.build_covering_vocab(
            counts, 3 + 256 + 2, tokenizer
        )
        vocab = student.Vocabulary.covering(kept, pieces, offsets, lengths)
        with tempfile.TemporaryDirectory() as tmp:
            # document 1: "abc" (2 pieces), "ab"; document 2: "xyz" (3 pieces)
            shard = tiny_shard(tmp, [np.array([257, 256]), np.array([259])])
            inputs, labels, ids, lps, native = student.pack_shard(Path(shard), vocab)
        abc, _ = vocab.expand([257])
        xyz, _ = vocab.expand([259])
        ab = vocab.encode([256])[0]
        self.assertEqual(inputs.tolist(), [student.BOS, *abc, ab, student.BOS, *xyz])
        self.assertEqual(labels.tolist(), [*abc, ab, -100, *xyz, -100])
        self.assertEqual(native.tolist(), [257, -1, 256, -1, 259, -1, -1, -1])
        self.assertEqual(ids[1, 0], abc[1])  # continuation piece: one-hot on itself
        self.assertEqual(float(lps[1, 0]), 0.0)
        self.assertTrue((ids[1, 1:] == -1).all())
        self.assertEqual(
            ids[0, 0], abc[0]
        )  # teacher's top-1 (the token) maps to its first piece


class ScoreTest(unittest.TestCase):
    def test_score_counts_passes_per_set_with_interval(self):
        items = [
            {"id": "a", "set": "mbpp"},
            {"id": "b", "set": "mbpp"},
            {"id": "c", "set": "curriculum"},
        ]
        completions = {"a": {"code": "x"}, "b": {"code": None}, "c": {"code": "y"}}
        grades = {"a": {"passed": True}, "c": {"passed": False}}
        scores = jobs.score(items, completions, grades)
        self.assertEqual(scores["mbpp"]["passed"], 1)
        self.assertEqual(scores["mbpp"]["parsed"], 1)
        self.assertEqual(scores["curriculum"]["pass_rate"], 0.0)
        low, high = jobs.wilson(3, 32)
        self.assertAlmostEqual(low, 0.0324, places=3)
        self.assertAlmostEqual(high, 0.2422, places=3)


class VllmModelDirTest(unittest.TestCase):
    def test_model_dir_links_the_checkpoint_and_describes_it(self):
        config = student.ModelConfig(
            vocab_size=40, block_size=32, n_layer=1, n_head=2, n_embd=16, eos_id=7
        )
        model = student.GPT(config)
        vocab = student.Vocabulary.full(40, bos=1)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "run" / "model.pt"
            checkpoint.parent.mkdir()
            torch.save(
                {"model": model.state_dict(), "config": config.__dict__}, checkpoint
            )
            with unittest.mock.patch.object(jobs, "IM_END", 7):
                out = jobs.vllm_model_dir(checkpoint, vocab, Path(tmp) / "vllm")
            written = json.loads((out / "config.json").read_text())
            self.assertEqual(written["architectures"], ["SimpleLLMForCausalLM"])
            self.assertEqual(written["num_hidden_layers"], 1)
            self.assertEqual(written["eos_token_id"], 7)
            self.assertEqual((out / "model.pt").resolve(), checkpoint.resolve())
            self.assertEqual(int(np.load(out / "vocab.npz")["bos"]), 1)
            saved = torch.load(
                out / "model.pt", weights_only=True
            )  # what vLLM's pt loader does
            self.assertEqual(set(saved["model"]), set(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
