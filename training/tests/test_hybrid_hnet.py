"""Byte-level H-Net: dynamic chunking, the dechunk EMA, document isolation and byte data."""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from model_hybrid import byte_data, data, hnet

from trainer.optimizers import RoleOptimizer, parameter_groups

CONFIG = Path(__file__).parents[1] / "hybrid" / "configs" / "hnet-720m-40b.json"
LOCAL = os.environ.get("HYBRID_GLM_TOKENIZER", "artifacts/glm/GLM-4.7-Flash/tokenizer.json")


def tiny_config(**overrides):
    main = {
        "n_layer": 2,
        "n_embd": 48,
        "n_head": 2,
        "n_kv_head": 1,
        "head_dim": 16,
        "mlp_hidden": 64,
        "layer_pattern": ["kda", "full"],
        "n_kda_head": 2,
    }
    settings = {
        "main": main,
        "block_size": 64,
        "outer_embd": 32,
        "outer_layers": 1,
        "outer_kda_head": 2,
        "outer_mlp_hidden": 48,
        "target_ratio": 3.0,
        "chunk_bucket": 8,
        "loss_chunk_size": 50,
    }
    return hnet.HNetConfig(**(settings | overrides))


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return hnet.HNetLM(tiny_config()).double()


def sequential_ema(values, probability, documents):
    out, state = torch.zeros_like(values), None
    for t in range(values.size(1)):
        p = probability[:, t, None].to(values.dtype)
        fresh = values[:, t] * p
        if t == 0:
            state = fresh
        else:
            same = (documents[:, t] == documents[:, t - 1])[:, None]
            state = torch.where(same, fresh + (1 - p) * state, fresh)
        out[:, t] = state
    return out


class DechunkTests(unittest.TestCase):
    def test_ema_matches_recurrence(self):
        torch.manual_seed(0)
        for length in (5, hnet.EMA_BLOCK, 2 * hnet.EMA_BLOCK + 37):
            values = torch.randn(2, length, 3, dtype=torch.float64)
            probability = 0.5 + 0.5 * torch.rand(2, length, dtype=torch.float64)
            documents = (torch.rand(2, length) < 0.05).long().cumsum(1)
            expected = sequential_ema(values, probability, documents)
            torch.testing.assert_close(
                hnet.dechunk_ema(values, probability, documents), expected, rtol=1e-9, atol=1e-9
            )

    def test_chunk_order(self):
        boundaries = torch.tensor([[1, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 1]], dtype=torch.bool)
        order, counts = hnet.chunk_order(boundaries, bucket=4)
        self.assertEqual(counts.tolist(), [3, 2])
        self.assertEqual(order.shape, (2, 4))
        self.assertEqual(order[0, :3].tolist(), [0, 2, 3])
        self.assertEqual(order[1, :2].tolist(), [0, 5])


class HNetModelTests(unittest.TestCase):
    def test_config_matches_baseline_compute(self):
        settings = json.loads(CONFIG.read_text())
        self.assertEqual(settings["architecture"], "hnet")
        config = hnet.HNetConfig(**settings["model"])
        baseline = json.loads(CONFIG.with_name("hybrid-720m-40b.json").read_text())["model"]
        for key, value in config.main.items():
            self.assertEqual(baseline[key], value, key)
        with torch.device("meta"):
            lm = hnet.HNetLM(config)
        # Training FLOPs per byte at the GLM tokenizer's 4.54 bytes per token on this recipe;
        # the 720M hybrid (with MTP) spends 1.31 GFLOP per byte.
        flops = hnet.flops_per_byte(lm, 4.54, 800 / 4.54)
        self.assertAlmostEqual(flops / 1e9, 1.31, delta=0.05)

    def test_rejects_tokens_and_mtp(self):
        with self.assertRaises(ValueError):
            tiny_config(eos_id=0)
        with self.assertRaises(ValueError):
            tiny_config(mtp=True)
        with self.assertRaises(ValueError):
            tiny_config(main={"n_layer": 2, "vocab": 3})

    def test_loss_and_gradients(self):
        lm = tiny_model()
        tokens = torch.randint(0, 256, (2, 64))
        tokens[0, 20] = hnet.BYTE_EOS
        labels = torch.roll(tokens, -1, 1)
        labels[:, -1] = -100
        losses = lm(tokens, labels)
        self.assertEqual(int(losses["count"]), 2 * 63)
        expected = (
            losses["ce"]
            + lm.config.z_loss * losses["z"]
            + lm.config.ratio_loss_weight * losses["ratio"] * losses["count"]
        )
        torch.testing.assert_close(losses["loss"], expected)
        logits = lm(tokens)
        ce = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
        )
        torch.testing.assert_close(losses["ce"], ce)
        losses["loss"].backward()
        missing = [n for n, p in lm.named_parameters() if p.grad is None or not p.grad.any()]
        self.assertEqual(missing, [])

    def test_ratio_loss(self):
        lm = tiny_model()
        n = lm.config.target_ratio
        boundaries = torch.zeros(1, 30, dtype=torch.bool)
        boundaries[0, ::3] = True
        probability = torch.full((1, 30), 1 / n, dtype=torch.float64)
        self.assertAlmostEqual(lm.ratio_loss(probability, boundaries).item(), 1.0)
        boundaries[:] = True
        self.assertGreater(lm.ratio_loss(torch.ones(1, 30), boundaries).item(), 1.0)

    def test_document_starts_are_boundaries(self):
        lm = tiny_model()
        tokens = torch.randint(0, 256, (1, 40))
        tokens[0, [9, 25]] = hnet.BYTE_EOS
        _, probability, boundaries = lm.forward_hidden(tokens)
        self.assertTrue(boundaries[0, [0, 10, 26]].all())
        self.assertTrue((probability[0, [0, 10, 26]] == 1).all())

    def test_causal(self):
        lm = tiny_model()
        tokens = torch.randint(0, 256, (1, 48))
        changed = tokens.clone()
        changed[0, 30:] = torch.randint(0, 256, (18,))
        with torch.no_grad():
            first, second = lm(tokens), lm(changed)
        torch.testing.assert_close(first[:, :30], second[:, :30], rtol=0, atol=1e-10)

    def test_documents_are_isolated(self):
        lm = tiny_model()
        first = torch.randint(0, 256, (1, 23))
        first[0, -1] = hnet.BYTE_EOS
        second = torch.randint(0, 256, (1, 30))
        with torch.no_grad():
            alone = lm(second)
            after = lm(torch.cat((first, second), 1))[:, first.size(1) :]
        torch.testing.assert_close(alone, after, rtol=0, atol=1e-10)

    def test_optimizer_roles(self):
        lm = hnet.HNetLM(tiny_config())
        groups = parameter_groups(lm)
        names = {role: {n for n, _ in group} for role, group in groups.items()}
        self.assertIn("routing.q_proj.weight", names["adamw_no_decay"])
        self.assertIn("residual_proj.weight", names["muon"])
        self.assertIn("lm_head.weight", names["adamw_decay"])
        self.assertIn("pad_dimension", names["adamw_no_decay"])
        RoleOptimizer(lm)

    def test_initialization(self):
        lm = hnet.HNetLM(tiny_config())
        width = lm.config.outer_embd
        torch.testing.assert_close(lm.routing.q_proj.weight, torch.eye(width))
        self.assertFalse(lm.residual_proj.weight.any())
        self.assertFalse(lm.pad_dimension.any())

    def test_byte_tokenizer(self):
        tokenizer = hnet.ByteTokenizer()
        ids = tokenizer.encode("héllo").ids
        self.assertEqual(len(ids), 6)
        self.assertEqual(tokenizer.decode([*ids, hnet.BYTE_EOS]), "héllo")


class HNetFamilyTests(unittest.TestCase):
    def family(self):
        from model_hybrid.train import HNetFamily, RunConfig, family_of

        settings = tiny_config()
        config = RunConfig(
            recipe="",
            model={
                k: getattr(settings, k)
                for k in (
                    "main",
                    "block_size",
                    "outer_embd",
                    "outer_layers",
                    "outer_kda_head",
                    "outer_mlp_hidden",
                    "target_ratio",
                    "chunk_bucket",
                    "loss_chunk_size",
                )
            },
            target_tokens=10_000,
            architecture="hnet",
            global_rows=4,
            microbatch=2,
        )
        self.assertIs(family_of(config), HNetFamily)
        with tempfile.TemporaryDirectory() as output:
            return family_of(config)(config, output, data=({}, None))

    def test_loss_and_log(self):
        family = self.family()
        torch.manual_seed(0)
        lm = family.build("cpu")
        inputs = torch.randint(0, 256, (2, 64))
        labels = torch.roll(inputs, -1, 1)
        extras, counts = family.prepare(inputs, labels)
        self.assertEqual(extras, {})
        objective, sums = family.loss(lm, inputs, labels, extras, counts, {"tokens": 0})
        torch.testing.assert_close(
            objective, lm(inputs, labels)["loss"] / counts["ntp"], rtol=1e-5, atol=1e-6
        )
        self.assertEqual(set(sums), {"ce", "z", "ratio", "chunks"})
        # Two microbatches per step: the logged ratio loss is their mean.
        doubled = {k: 2 * v for k, v in sums.items()}
        row = family.log(doubled, {"ntp": 2 * counts["ntp"]})
        self.assertAlmostEqual(row["ratio_loss"], float(sums["ratio"]), places=5)
        self.assertAlmostEqual(row["bytes_per_chunk"], 4 * 64 / float(doubled["chunks"]))
        self.assertIn("mfu", family.step_metrics(inputs, 1.0, inputs.numel(), 1))
        self.assertEqual(family.metadata()["architecture"]["module"], "model_hybrid.hnet")

    def test_heldout_evaluation(self):
        family = self.family()
        torch.manual_seed(0)
        lm = family.build("cpu")
        stream = torch.randint(0, 256, (130,))
        stream[[40, 129]] = hnet.BYTE_EOS
        inputs, labels = data.pack_windows(stream.numpy().astype(np.int64), 64)
        windows = {"web": (inputs, labels, 128)}
        heldout = {"windows": windows, "tokenizer": hnet.ByteTokenizer()}
        result = family.evaluate(lm, heldout, {"step": 1}, final=False)["heldout"]["web"]
        self.assertEqual(result["tokens"], 129)
        self.assertAlmostEqual(result["bits_per_byte"], result["loss"] * 129 / 128 / math.log(2))
        self.assertGreater(result["bytes_per_chunk"], 1)

    def test_workflows_count_bytes(self):
        from workflows.config import resolve_payload, step_tokens
        from workflows.plan import load_workflow

        root = Path(__file__).parents[1] / "workflows"
        plan = load_workflow(root / "hnet-h100x8.json", "check")
        policy, _, settings = resolve_payload(plan["stages"][-1]["spec"])
        self.assertEqual(policy.family, "hybrid")
        self.assertEqual(settings["architecture"], "hnet")
        self.assertEqual(step_tokens("hybrid", settings), 288 * 16_384)


def write_shard(root, name, index, ids, documents, heldout):
    bin_path, docs_path, meta_path = data.shard_paths(root, name, index)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(ids, dtype=np.uint32).tofile(bin_path)
    np.save(docs_path, np.asarray(documents, dtype=np.int64))
    meta = {
        "documents": len(documents),
        "tokens": len(ids),
        "bytes": int(sum(d[1] for d in documents)),
        "heldout": heldout,
    }
    meta_path.write_text(json.dumps(meta))


class ByteDataTests(unittest.TestCase):
    # Token 0 -> "a", 1 -> "bc", 2 -> "def", 3 -> EOS.
    TABLE = (
        np.array([97, 98, 99, 100, 101, 102, hnet.BYTE_EOS], dtype=np.uint16),
        np.array([0, 1, 3, 6, 7]),
    )

    def test_alphabet_is_a_bijection(self):
        alphabet = byte_data.byte_level_alphabet()
        self.assertEqual(sorted(alphabet.values()), list(range(256)))
        self.assertEqual(alphabet["Ġ"], ord(" "))

    def test_token_bytes(self):
        stream = byte_data.token_bytes([2, 0, 1, 3, 1], self.TABLE, block=2)
        self.assertEqual(stream.tolist(), [100, 101, 102, 97, 98, 99, hnet.BYTE_EOS, 98, 99])
        self.assertEqual(byte_data.token_bytes([], self.TABLE).size, 0)

    def test_mixture_and_heldout(self):
        recipe = {
            "name": "bytes",
            "context": 4,
            "seed": 1,
            "sources": {
                "short": {"weight": 0.5, "shards": 2, "max_passes": 1},
                "long": {"weight": 0.5, "shards": 2, "max_passes": 1},
            },
        }
        with tempfile.TemporaryDirectory() as root:
            for index in range(2):
                # "short" decodes 1 byte per token, "long" 3, each with EOS documents.
                write_shard(root, "short", index, [0] * 9 + [3], [[10, 9]], index == 1)
                write_shard(root, "long", index, [2] * 9 + [3], [[10, 27]], index == 1)
            mixture = byte_data.ByteMixture(recipe, root, 5, self.TABLE)
            self.assertAlmostEqual(mixture.weights["long"] / mixture.weights["short"], 28 / 10)
            inputs, labels, names = mixture.next_batch(2)
            self.assertEqual(inputs.shape, (2, 5))
            np.testing.assert_array_equal(inputs[:, 1:], labels[:, :-1])
            self.assertEqual(sorted(names), ["long", "short"])
            inputs, labels, total = byte_data.heldout_byte_windows(
                recipe, root, "long", 8, 1, self.TABLE
            )
            self.assertEqual(total, 27)
            self.assertEqual(int((labels != -100).sum()), 27)
            self.assertEqual(labels[labels != -100][-1], hnet.BYTE_EOS)

    def test_heldout_rejects_mismatched_lengths(self):
        recipe = {"sources": {"web": {"shards": 1}}}
        with tempfile.TemporaryDirectory() as root:
            write_shard(root, "web", 0, [1, 3], [[2, 5]], True)
            with self.assertRaises(ValueError):
                byte_data.heldout_byte_windows(recipe, root, "web", 4, 1, self.TABLE)

    @unittest.skipUnless(Path(LOCAL).exists(), "GLM tokenizer.json not downloaded")
    def test_glm_tokens_decode_to_utf8(self):
        from model_hybrid import tokenizer as glm

        tokenizer = glm.load(LOCAL)
        table = byte_data.token_byte_table(tokenizer, glm.EOS_ID)
        for text in ("héllo wörld 中文\t\n", "def f():\r\n    return '🎉'", "<think> x"):
            ids = [*tokenizer.encode(text, add_special_tokens=False).ids, glm.EOS_ID]
            stream = byte_data.token_bytes(ids, table)
            self.assertEqual(stream[-1], hnet.BYTE_EOS)
            self.assertEqual(bytes(stream[:-1].astype(np.uint8)), text.encode())


if __name__ == "__main__":
    unittest.main()
