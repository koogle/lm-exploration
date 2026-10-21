"""Pending data must be rejected before ingestion; no Modal app imports required."""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class PreparationPreflightTests(unittest.TestCase):
    def test_pending_data_fails_before_ingestion(self):
        tree = ast.parse((ROOT / "distributed_training/modal_small_5b.py").read_text())
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "prepare_and_launch"
        )
        function.decorator_list = []
        settings = json.loads(
            (ROOT / "distributed_training/recipes/simple-llm-10m-5b.json").read_text()
        )
        with tempfile.TemporaryDirectory() as temporary:
            namespace = {
                "Path": lambda path: Path(temporary) / str(path).lstrip("/"),
                "CORPUS_ROOT": Path(temporary) / "data",
                "json": json,
                "record": Mock(),
                "data_volume": Mock(),
            }
            # Execute only the trusted repository function, without cloud decorators.
            exec(  # noqa: S102
                compile(ast.Module(body=[function], type_ignores=[]), str(ROOT), "exec"), namespace
            )
            with (
                patch("training_data.simple_llm.load_recipe", return_value={}),
                patch("training_data.prepared_snapshot.find_prepared_snapshot", return_value=None),
                patch("training_data.fast_raw_corpus.FastRawCorpus") as corpus,
                self.assertRaisesRegex(ValueError, "allow_unreviewed_data required"),
            ):
                namespace[function.name](
                    "run", "data-v1", {**settings, "allow_unreviewed_data": False}, {}
                )
            corpus.assert_not_called()
