"""Fill-in-the-middle settings of hybrid run configs."""

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from model_hybrid import train
from training_data import packed_shards

from tests.test_packed_shards import CharTokenizer
from tokenizer import glm
from trainer.continued_pretraining import seed_phase
from trainer.loop import CHECKPOINT_FORMAT, check_resume, source_hashes, trainer_sources

FIM = {"rate": 0.5, "spm_rate": 0.5, "sources": ["code"], "seed": 1}


def recipe():
    source = {"weight": 0.5, "max_passes": 1, "shards": 2}
    return {
        "name": "fim-test",
        "context": 16,
        "seed": 3,
        "sources": {"web": source, "code": source},
    }


class HybridFimTests(unittest.TestCase):
    def test_run_config_checks_fim(self):
        train.RunConfig(recipe="", model={}, target_tokens=1, fim=FIM)
        for bad in (
            {**FIM, "rate": 0},
            {**FIM, "spm_rate": 2},
            {**FIM, "sources": []},
            {"rate": 1},
        ):
            with self.assertRaises(ValueError):
                train.RunConfig(recipe="", model={}, target_tokens=1, fim=bad)

    def test_recipe_data_passes_fim_with_glm_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fim-test.json"
            path.write_text(json.dumps(recipe()))
            config = train.RunConfig(recipe=str(path), model={}, target_tokens=1, fim=FIM)
            identity, mixture, _, chat = train.recipe_data(config, folder)
            self.assertFalse(chat)
            self.assertEqual(identity["fim"], FIM)
            built = mixture()
            self.assertEqual(
                built.fim["tokens"],
                {
                    "prefix": glm.SPECIAL_TOKENS["<|code_prefix|>"],
                    "suffix": glm.SPECIAL_TOKENS["<|code_suffix|>"],
                    "middle": glm.SPECIAL_TOKENS["<|code_middle|>"],
                },
            )
            plain = train.recipe_data(
                train.RunConfig(recipe=str(path), model={}, target_tokens=1), folder
            )
            self.assertNotIn("fim", plain[0])
            self.assertIsNone(plain[1]().fim)


MODEL = {
    "vocab_size": glm.VOCAB_SIZE,
    "eos_id": glm.EOS_ID,
    "block_size": 16,
    "n_layer": 2,
    "n_embd": 32,
    "n_head": 4,
    "n_kv_head": 2,
    "head_dim": 8,
    "mlp_hidden": 48,
    "loss_chunk_size": 7,
    "layer_pattern": ["full"],
    "mtp": False,
}


class HybridPhaseSeedTests(unittest.TestCase):
    """A FIM branch seeded from a hybrid checkpoint written as trainer.loop writes them."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        import pyarrow as pa
        import pyarrow.parquet as pq

        source = {"repo": "r", "revision": "r", "shards": 2, "max_passes": 1}
        recipe = {
            "name": "seed-recipe",
            "context": 16,
            "seed": 5,
            "sources": {
                "web": {
                    **source,
                    "weight": 0.5,
                    "pattern": "web-{index}.parquet",
                    "column": "text",
                },
                "code": {
                    **source,
                    "weight": 0.5,
                    "pattern": "code-{index}.parquet",
                    "column": "code",
                },
            },
            "contamination": {"ngram": 13, "benchmarks": []},
        }
        for i in range(2):
            pq.write_table(
                pa.table({"text": [f"web {i} {j} " * 5 for j in range(40)]}),
                self.root / f"web-{i}.parquet",
            )
            pq.write_table(
                pa.table({"code": [f"def f{i}_{j}(x): return x + {j}" for j in range(40)]}),
                self.root / f"code-{i}.parquet",
            )
            for name in ("web", "code"):
                packed_shards.prepare_shard(
                    recipe,
                    name,
                    i,
                    self.root / "data" / recipe["name"],
                    CharTokenizer(),
                    np.empty(0, np.uint64),
                    glm.EOS_ID,
                    lambda repo, rev, file: self.root / file,
                )
        self.recipe_path = self.root / "seed-recipe.json"
        self.recipe_path.write_text(json.dumps(recipe))

    def tearDown(self):
        self.folder.cleanup()

    def family(self, **overrides):
        settings = {
            "recipe": str(self.recipe_path),
            "model": MODEL,
            "target_tokens": 10_000,
            "warmup_tokens": 0,
            "optimizer": {},
            "compile": False,
        }
        config = train.RunConfig(**(settings | overrides))
        return train.HybridFamily(config, self.root / "run", data_root=self.root / "data")

    def parent(self):
        family = self.family()
        model = family.build("cpu")
        optimizer = family.make_optimizer(model)
        loss = sum(p.float().square().sum() for p in model.parameters())
        loss.backward()
        optimizer.step()
        data = family.open_data(None, None)
        for _ in range(7):
            data.next_batch(1)
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.get_rng_state(),
        }
        sources = source_hashes({**trainer_sources(), **family.source_files()})
        payload = {
            "format": CHECKPOINT_FORMAT,
            "config": family.settings,
            **family.metadata(),
            "data_identity": family.identity,
            "source_sha256": sources,
            "torch_version": str(torch.__version__),
            "step": 3,
            "tokens": 48,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "data": data.state_dict(),
            "rng": [rng],
        }
        path = self.root / "parent.pt"
        torch.save(payload, path)
        return path, payload

    def test_fim_branch_keeps_optimizer_and_continues_the_cursor(self):
        path, parent = self.parent()
        child = self.family(target_tokens=5_000, decay_fraction=1.0, fim=FIM)
        lineage = seed_phase(child, path, self.root / "seed" / "step-000000000.pt", note="fim")
        seed = torch.load(self.root / "seed" / "step-000000000.pt", weights_only=False)
        self.assertEqual((seed["step"], seed["tokens"]), (0, 0))
        self.assertEqual((lineage["parent_step"], lineage["parent_tokens"]), (3, 48))
        self.assertEqual(seed["data_identity"]["fim"], FIM)
        self.assertEqual(seed["data"]["sources"]["web"], parent["data"]["sources"]["web"])
        expected = child.factory().continue_state(parent["data"])
        self.assertEqual(seed["data"], expected)
        for key, state in parent["optimizer"]["adamw"]["state"].items():
            torch.testing.assert_close(
                seed["optimizer"]["adamw"]["state"][key]["exp_avg"], state["exp_avg"]
            )
        sources = source_hashes({**trainer_sources(), **child.source_files()})
        check_resume(seed, child, child.identity, sources, str(torch.__version__))

    def test_seed_rejects_another_architecture(self):
        path, _ = self.parent()
        other = self.family(model={**MODEL, "n_layer": 3}, fim=FIM)
        with self.assertRaisesRegex(ValueError, "architecture"):
            seed_phase(other, path, self.root / "other.pt")


if __name__ == "__main__":
    unittest.main()
