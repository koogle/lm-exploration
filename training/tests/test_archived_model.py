"""Archived evaluation sources are verified and do not replace live packages."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from model_100m import model as current_model

from distributed_training.archived_model import load_archived_model


class ArchivedModelTests(unittest.TestCase):
    def test_verified_sources_load_and_restore_current_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            definitions = {
                "model_10m": "class GPT: pass\nclass ModelConfig: pass\n",
                "model_100m": (
                    "from model_10m.model import GPT as Base, ModelConfig\nclass GPT(Base): pass\n"
                ),
            }
            hashes = {}
            for package, text in definitions.items():
                folder = source / package
                folder.mkdir()
                path = folder / "model.py"
                path.write_text(text)
                hashes[package] = hashlib.sha256(path.read_bytes()).hexdigest()
            before = {k: v for k, v in sys.modules.copy().items() if k.startswith("model_")}
            GPT, config = load_archived_model(source, hashes)
            self.assertEqual(GPT.__bases__[0].__module__, "model_10m.model")
            self.assertEqual(config.__module__, "model_10m.model")
            self.assertIsNot(GPT, current_model.GPT)
            self.assertEqual(
                {k: v for k, v in sys.modules.copy().items() if k.startswith("model_")}, before
            )
            for package in definitions:
                with (
                    self.subTest(package=package),
                    self.assertRaisesRegex(ValueError, "checksum mismatch"),
                ):
                    load_archived_model(source, {**hashes, package: "0" * 64})
            # Restore imports even when verified code itself fails during import.
            path = source / "model_100m/model.py"
            path.write_text("raise RuntimeError('broken archive')\n")
            hashes["model_100m"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "broken archive"):
                load_archived_model(source, hashes)
            self.assertIs(sys.modules["model_100m.model"], current_model)
