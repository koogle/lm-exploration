"""Exercise generic controller routing and preflight without importing Modal apps."""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from workflows.config import load_workflow

ROOT = Path(__file__).resolve().parents[1]


def controller(name):
    tree = ast.parse((ROOT / "workflows/modal_train.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    function.decorator_list = []
    namespace = {"Path": Mock(side_effect=AssertionError("filesystem accessed before preflight"))}
    # Only trusted repository code is compiled, without its cloud decorators.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT), "exec"), namespace)  # noqa: S102
    return namespace[name]


class WorkflowControllerTests(unittest.TestCase):
    def setUp(self):
        self.bundle = load_workflow(ROOT / "workflows/small-5b.json")

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
                patch("training_data.prepared_snapshot.find_prepared_snapshot", return_value=None),
                patch("training_data.fast_raw_corpus.FastRawCorpus") as corpus,
            ):
                with self.assertRaisesRegex(ValueError, "allow_unreviewed_data required"):
                    prepare(
                        "run",
                        "data-v1",
                        {
                            **self.bundle,
                            "workflow": {**self.bundle["workflow"], "allow_unreviewed_data": False},
                        },
                        {},
                    )
                corpus.assert_not_called()

    def test_snapshot_selection_preflight_before_gpu_scheduling(self):
        from check_distributed_cuda import fixture

        for invalid in ("none", "review", "capacity"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                parent = root / "data/other-corpus/v1/snapshots"
                parent.mkdir(parents=True)
                fixture(parent / "prepared")
                bundle = load_workflow(ROOT / "workflows/h100x1-pilot.json")
                bundle["dataset"].update(root="other-corpus", snapshot="v1/snapshots/prepared")
                if invalid == "review":
                    bundle["workflow"]["allow_unreviewed_data"] = False
                if invalid == "capacity":
                    bundle["workflow"]["pilot"] = False
                launch = controller("launch_snapshot")
                continuation = Mock()
                launch.__globals__.update(
                    Path=lambda path, root=root: root / str(path).lstrip("/"),
                    json=json,
                    record=Mock(),
                    continue_training=continuation,
                )
                if invalid == "none":
                    launch("run", bundle, {})
                    continuation.spawn.assert_called_once()
                    self.assertEqual(continuation.spawn.call_args.args[1], "v1/snapshots/prepared")
                    self.assertEqual(continuation.spawn.call_args.args[2], bundle)
                else:
                    with self.assertRaises(ValueError):
                        launch("run", bundle, {})
                    continuation.spawn.assert_not_called()
