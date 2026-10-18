"""Offline layout checks; never import Modal apps or resolve cloud resources."""

import ast
import importlib.util
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScriptLayoutTests(unittest.TestCase):
    def test_training_root_keeps_only_generation_entrypoint(self):
        self.assertEqual({p.name for p in ROOT.glob("*.py")}, {"generate.py"})
        self.assertEqual(len(list((ROOT / "modal_scripts").glob("modal_*.py"))), 8)
        self.assertFalse((ROOT / "modal_scripts/modal_scholarly_pilot.py").exists())
        self.assertFalse((ROOT / "trainer/subword_pilot.py").exists())
        self.assertTrue((ROOT / "data/prepare_scholarly.py").is_file())

    def test_installed_module_locations(self):
        expected = {
            "training_data.prepare_scholarly": "data/prepare_scholarly.py",
            "tokenizer.tokenizer_research": "tokenizer/tokenizer_research.py",
            "tokenizer.tokenizer_evaluation": "tokenizer/tokenizer_evaluation.py",
            "tokenizer.tokenizer_report": "tokenizer/tokenizer_report.py",
            "tokenizer.structured_tokenizer": "tokenizer/structured_tokenizer.py",
            "tokenizer.structured_tokenizer_report": "tokenizer/structured_tokenizer_report.py",
            "tokenizer.package_tokenizer_research": "tokenizer/package_tokenizer_research.py",
            "tokenizer.verify_tokenizer_publication": "tokenizer/verify_tokenizer_publication.py",
            "modal_scripts.modal_train": "modal_scripts/modal_train.py",
            "distributed_training.train": "distributed_training/train.py",
            "distributed_training.config": "distributed_training/config.py",
            "distributed_training.modal_train": "distributed_training/modal_train.py",
            "distributed_training.modal_checks": "distributed_training/modal_checks.py",
        }
        for name, relative in expected.items():
            with self.subTest(module=name):
                spec = importlib.util.find_spec(name)
                self.assertIsNotNone(spec)
                self.assertEqual(Path(spec.origin).resolve(), ROOT / relative)
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(
            config["project"]["scripts"]["lm-prepare-scholarly"],
            "training_data.prepare_scholarly:main",
        )

    def test_modal_image_local_sources_exist(self):
        paths = [
            *(ROOT / "modal_scripts").glob("modal_*.py"),
            *(ROOT / "distributed_training").glob("*.py"),
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

    def test_no_old_top_level_tool_imports(self):
        old = {
            "tokenizer_research",
            "tokenizer_evaluation",
            "tokenizer_report",
            "structured_tokenizer",
            "structured_tokenizer_report",
            "package_tokenizer_research",
            "prepare_scholarly",
            "modal_scholarly_pilot",
        }
        for folder in ("modal_scripts", "distributed_training", "tokenizer", "tests"):
            for path in (ROOT / folder).glob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        self.assertNotIn(node.module, old, str(path))
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertNotIn(alias.name, old, str(path))


if __name__ == "__main__":
    unittest.main()
