"""Offline layout checks; never import Modal apps or resolve cloud resources."""

import ast
import importlib.util
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScriptLayoutTests(unittest.TestCase):
    def test_installed_module_locations(self):
        expected = {
            "training_data.prepare_simple_llm": "data/prepare_simple_llm.py",
            "training_data.simple_llm": "data/simple_llm.py",
            "modal_scripts.modal_prepare_simple_llm": "modal_scripts/modal_prepare_simple_llm.py",
            "tokenizer.tokenizer_research": "tokenizer/tokenizer_research.py",
            "tokenizer.tokenizer_evaluation": "tokenizer/tokenizer_evaluation.py",
            "tokenizer.tokenizer_report": "tokenizer/tokenizer_report.py",
            "tokenizer.structured_tokenizer": "tokenizer/structured_tokenizer.py",
            "tokenizer.structured_tokenizer_report": "tokenizer/structured_tokenizer_report.py",
            "tokenizer.package_tokenizer_research": "tokenizer/package_tokenizer_research.py",
            "tokenizer.verify_tokenizer_publication": "tokenizer/verify_tokenizer_publication.py",
            "trainer.loop": "trainer/loop.py",
            "distributed.config": "distributed/config.py",
            "distributed.modal_train": "distributed/modal_train.py",
            "distributed.modal_checks": "distributed/modal_checks.py",
            "workflows.modal_train": "workflows/modal_train.py",
            "workflows.config": "workflows/config.py",
            "workflows.checks": "workflows/checks.py",
            "distributed.modal_prep_benchmark": "distributed/modal_prep_benchmark.py",
            "sampling.modal_sample": "analysis/sampling/modal_sample.py",
            "sampling.report": "analysis/sampling/report.py",
            "workflows.modal_post_train": "workflows/modal_post_train.py",
            "workflows.modal_app": "workflows/modal_app.py",
            "modal_scripts.modal_sft_eval": "modal_scripts/modal_sft_eval.py",
            "modal_scripts.modal_sft_data": "modal_scripts/modal_sft_data.py",
            "trainer.sft": "trainer/sft.py",
            "trainer.sft_recipe": "trainer/sft_recipe.py",
        }
        for name, relative in expected.items():
            with self.subTest(module=name):
                spec = importlib.util.find_spec(name)
                self.assertIsNotNone(spec)
                self.assertEqual(Path(spec.origin).resolve(), ROOT / relative)
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(
            config["project"]["scripts"]["lm-prepare-simple-llm"],
            "training_data.prepare_simple_llm:main",
        )

    def test_modal_image_local_sources_exist(self):
        paths = [
            *(ROOT / "modal_scripts").glob("modal_*.py"),
            *(ROOT / "distributed").glob("*.py"),
            *(ROOT / "workflows").glob("*.py"),
            ROOT / "analysis/sampling/modal_sample.py",
        ]
        for path in paths:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr not in ("add_local_file", "add_local_dir"):
                    continue
                source = node.args[0]
                self.assertIsInstance(source, ast.Constant)
                local = ROOT / source.value
                with self.subTest(script=path.name, source=source.value):
                    if node.func.attr == "add_local_file":
                        self.assertTrue(local.is_file(), str(local))
                    else:
                        self.assertTrue(local.is_dir(), str(local))

    def test_training_worker_callers_pass_dataset_root(self):
        """distributed.modal_train.worker takes dataset_root as a required keyword."""
        for path in sorted(ROOT.rglob("*.py")):
            if ".venv" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            imported = any(
                isinstance(node, ast.ImportFrom)
                and node.module == "distributed.modal_train"
                and any(alias.name == "worker" and not alias.asname for alias in node.names)
                for node in ast.walk(tree)
            )
            if not imported:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "worker"
                    and node.func.attr in ("local", "remote", "spawn")
                ):
                    with self.subTest(script=str(path.relative_to(ROOT)), line=node.lineno):
                        self.assertIn("dataset_root", [k.arg for k in node.keywords])


if __name__ == "__main__":
    unittest.main()
