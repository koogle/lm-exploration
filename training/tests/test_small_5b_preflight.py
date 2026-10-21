"""Exercise controller preflight offline, without importing Modal entrypoints."""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def controller(name):
    tree = ast.parse((ROOT / "distributed_training/modal_small_5b.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    function.decorator_list = []
    namespace = {"Path": Mock(side_effect=AssertionError("filesystem accessed before preflight"))}
    # Only trusted repository code is compiled, without its cloud decorators.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT), "exec"), namespace)  # noqa: S102
    return namespace[name]


class Small5BPreflightTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads(
            (ROOT / "distributed_training/recipes/scholarly-modern-10m-5b.json").read_text()
        )

    def test_bad_paths_fail_before_filesystem_access(self):
        prepare = controller("prepare_and_launch")
        for run_id, version in (
            ("../escape", "data-v1"),
            ("/absolute", "data-v1"),
            ("r" * 77, "data-v1"),  # Leave room for the segment suffix.
            ("run", "../escape"),
            ("run", "/absolute"),
        ):
            with self.subTest(run_id=run_id, version=version), self.assertRaises(ValueError):
                prepare(run_id, version, self.settings, {})

    def test_pending_data_fails_before_ingestion(self):
        prepare = controller("prepare_and_launch")
        with tempfile.TemporaryDirectory() as temporary:
            prepare.__globals__.update(
                Path=lambda path: Path(temporary) / str(path).lstrip("/"),
                json=json,
                record=Mock(),
                data_volume=Mock(),
            )
            with (
                patch("training_data.simple_llm.load_recipe", return_value={}),
                patch("training_data.prepared_snapshot.find_prepared_snapshot", return_value=None),
                patch("training_data.fast_raw_corpus.FastRawCorpus") as corpus,
            ):
                with self.assertRaisesRegex(ValueError, "allow_unreviewed_data required"):
                    prepare("run", "data-v1", {**self.settings, "allow_unreviewed_data": False}, {})
                corpus.assert_not_called()

    def test_continuation_preflight_is_also_early(self):
        resume = controller("continue_training")
        for run_id, snapshot, segment in (
            ("../escape", "data-v1/snapshots/run", 0),
            ("run", "../escape", 0),
            ("run", "data-v1/snapshots/run", -1),
            ("run", "data-v1/snapshots/run", 20),
        ):
            with (
                self.subTest(run_id=run_id, snapshot=snapshot, segment=segment),
                self.assertRaises(ValueError),
            ):
                resume(run_id, snapshot, self.settings, {}, first_segment=segment)
