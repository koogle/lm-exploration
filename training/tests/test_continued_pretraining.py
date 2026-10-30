import hashlib
import json
import random
import sqlite3
import tempfile
import unittest
import zlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from model_100m.model import GPT, ModelConfig
from torch.nn import functional as F
from training_data.mixture import open_mixture
from training_data.subword import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    LOSS_IGNORE_INDEX,
    TOKENIZER_SHA256,
    VOCAB_SIZE,
    digest,
    load_frozen,
)

from distributed.config import DistributedConfig
from trainer.continued_pretraining import migrate_phase
from trainer.loop import CHECKPOINT_FORMAT, check_resume
from trainer.optimizers import make_optimizer, set_learning_rate
from trainer.subword import SubwordFamily, source_hashes


def snapshot(root, name):
    """A tiny reviewed raw snapshot with one train and one validation document."""
    root.mkdir()
    db = sqlite3.connect(root / "raw.sqlite")
    db.execute(
        """CREATE TABLE documents (hash TEXT PRIMARY KEY, source TEXT, split TEXT, parent TEXT,
        provenance TEXT, text BLOB, loss_tokens INTEGER, conflict INTEGER DEFAULT 0)"""
    )
    counts = {}
    for split in ("train", "validation"):
        text = f"{name} {split}: def add(a, b): return a + b. " * 30
        count = len(load_frozen().encode(text, add_special_tokens=False).ids) + 1
        db.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,0)",
            (
                hashlib.sha256(text.encode()).hexdigest(),
                name,
                split,
                name,
                "{}",
                zlib.compress(text.encode()),
                count,
            ),
        )
        counts[split] = {"loss_tokens": count}
    db.commit()
    db.close()
    manifest = {
        "format": "raw-text-v1",
        "training_ready": True,
        "max_passes": 2,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "database": {"path": "raw.sqlite", "sha256": digest(root / "raw.sqlite")},
        "sources": {name: {"weight_percent": 100, "splits": counts}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root / "manifest.json"


def tiny_config(optimizer, **overrides):
    return DistributedConfig(
        world_size=1,
        global_windows=2,
        microbatch=1,
        context=8,
        target_tokens=1000,
        warmup_tokens=10,
        optimizer=optimizer,
        data_loader="raw",
        n_layer=1,
        n_head=2,
        n_embd=16,
        **overrides,
    )


def build(config):
    model_config = ModelConfig(
        vocab_size=VOCAB_SIZE,
        block_size=config.context,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
    )
    model = GPT(model_config)
    return model, make_optimizer(model, config), model_config


def step(model, optimizer, config, data, tokens):
    x, y = data.next_batch(config.global_windows)
    count = int((y != LOSS_IGNORE_INDEX).sum())
    set_learning_rate(optimizer, config.learning_rate(tokens + count) / config.peak_lr)
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=LOSS_IGNORE_INDEX)
    loss.backward()
    optimizer.step()
    return count


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def tensors(state):
    """Flatten nested optimizer state into {path: tensor}."""
    if torch.is_tensor(state):
        return {"": state}
    if isinstance(state, dict):
        return {
            f"{key}/{path}": value
            for key, child in state.items()
            for path, value in tensors(child).items()
        }
    return {}


class ContinuedPretrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.parent_manifest = snapshot(self.root / "parent-data", "web")
        self.child_manifest = snapshot(self.root / "child-data", "code")

    def tearDown(self):
        self.folder.cleanup()

    def parent_checkpoint(self, config, lineage=None):
        """Two optimizer updates written exactly as trainer.loop writes checkpoints."""
        model, optimizer, model_config = build(config)
        data = open_mixture(self.parent_manifest, loader="raw", context=config.context)
        tokens = 0
        for index in range(2):
            tokens += step(model, optimizer, config, data, tokens)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "config": config.as_dict(),
            "architecture": {"module": ModelConfig.__module__, "config": asdict(model_config)},
            "special_tokens": {"bos": BOS_TOKEN_ID, "eos": EOS_TOKEN_ID},
            "data_identity": data.identity,
            "source_sha256": source_hashes(),
            "torch_version": str(torch.__version__),
            "step": 2,
            "tokens": tokens,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "data": data.state_dict(),
            "rng": [rng_state()],
            **({"lineage": lineage} if lineage else {}),
        }
        data.close()
        path = self.root / f"parent-{config.optimizer}.pt"
        torch.save(payload, path)
        return path, payload

    def migrate_and_step(self, optimizer_name):
        parent_config = tiny_config(optimizer_name)
        path, parent = self.parent_checkpoint(parent_config, lineage={"parent_tokens": 1000})
        original = path.read_bytes()
        # The new phase changes schedule, regularisation and data, not architecture.
        config = replace(parent_config, target_tokens=500, warmup_tokens=5, weight_decay=0.05)
        seed_path = self.root / "seed" / "step-000000000.pt"
        lineage = migrate_phase(path, seed_path, self.child_manifest, config, note="test")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(lineage["parent_tokens"], 1000 + parent["tokens"])
        self.assertEqual(lineage["parent_phase_tokens"], parent["tokens"])
        self.assertEqual(lineage["parent_lineage"], {"parent_tokens": 1000})
        self.assertEqual(lineage["parent_sha256"], digest(path))

        seed = torch.load(seed_path, weights_only=False)
        self.assertEqual((seed["step"], seed["tokens"]), (0, 0))
        self.assertEqual(seed["rng"][0]["python"], parent["rng"][0]["python"])
        self.assertTrue(torch.equal(seed["rng"][0]["torch"], parent["rng"][0]["torch"]))
        for name, value in parent["model"].items():
            self.assertTrue(torch.equal(seed["model"][name], value), name)
        parent_state, seed_state = tensors(parent["optimizer"]), tensors(seed["optimizer"])
        self.assertEqual(parent_state.keys(), seed_state.keys())
        for name, value in parent_state.items():
            self.assertTrue(torch.equal(seed_state[name], value), name)

        # The trainer accepts the seed for the new config and data exactly as on resume.
        data = open_mixture(self.child_manifest, loader="raw", context=config.context)
        family = SubwordFamily(config, self.child_manifest)
        check_resume(seed, family, data.identity, source_hashes(), str(torch.__version__))
        family.check_resume(seed)
        self.assertEqual(sum(seed["data"]["loss_token_exposures"].values()), 0)
        data.load_state_dict(seed["data"])
        model, optimizer, _ = build(config)
        model.load_state_dict(seed["model"])
        optimizer.load_state_dict(seed["optimizer"])
        groups = optimizer.param_groups
        self.assertTrue(all(group["weight_decay"] == 0.05 for group in groups))
        step(model, optimizer, config, data, 0)
        data.close()
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        return parent, optimizer.state_dict()

    def test_adamw_parent_keeps_moments_and_step_counts(self):
        parent, state = self.migrate_and_step("adamw")
        for index, entry in state["adamw"]["state"].items():
            previous = parent["optimizer"]["adamw"]["state"][index]["step"]
            self.assertEqual(float(entry["step"]), float(previous) + 1)

    def test_muon_parent_keeps_momentum_and_auxiliary_adamw_state(self):
        parent, state = self.migrate_and_step("muon")
        self.assertTrue(state["muon"]["state"])
        for index, entry in state["adamw"]["state"].items():
            previous = parent["optimizer"]["adamw"]["state"][index]["step"]
            self.assertEqual(float(entry["step"]), float(previous) + 1)
        # Base learning rates follow the new config (same values here, but from the new build).
        self.assertAlmostEqual(state["muon"]["param_groups"][0]["base_lr"], 0.02)

    def test_rejects_mismatches_and_overwrites(self):
        config = tiny_config("muon")
        path, _ = self.parent_checkpoint(config)
        seed = self.root / "seed.pt"
        cases = {
            "architecture": replace(config, n_layer=2),
            "different optimizer": replace(config, optimizer="adamw"),
            "world size": replace(config, world_size=2, global_windows=2),
        }
        for message, changed in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                migrate_phase(path, seed, self.child_manifest, changed)
        broken = torch.load(path, weights_only=False)
        next(iter(broken["model"].values())).fill_(float("nan"))
        torch.save(broken, self.root / "broken.pt")
        with self.assertRaisesRegex(ValueError, "not finite"):
            migrate_phase(self.root / "broken.pt", seed, self.child_manifest, config)
        seed.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            migrate_phase(path, seed, self.child_manifest, config)


if __name__ == "__main__":
    unittest.main()
