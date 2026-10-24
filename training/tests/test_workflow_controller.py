"""Exercise generic controller routing and preflight without importing Modal apps."""

import ast
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from workflows.config import load_workflow

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = {"deadline_unix": time.time() + 86400}


# Module helpers the controllers call; compiled alongside each one.
HELPERS = ("spawn_training", "started_chain")


def controller(name):
    tree = ast.parse((ROOT / "workflows/modal_train.py").read_text())
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in (name, *HELPERS)
    ]
    for function in functions:
        function.decorator_list = []
    namespace = {"Path": Mock(side_effect=AssertionError("filesystem accessed before preflight"))}
    # Only trusted repository code is compiled, without its cloud decorators.
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT), "exec"), namespace)  # noqa: S102
    return namespace[name]


class WorkflowControllerTests(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as temporary:
            workflow = json.loads((ROOT / "workflows/h100x1-pilot.json").read_text())
            model = (ROOT / "workflows" / workflow["model_config"]).read_text()
            workflow.update(model_config="model.json", dataset_config="data.json")
            (Path(temporary) / "run.json").write_text(json.dumps(workflow))
            (Path(temporary) / "model.json").write_text(model)
            (Path(temporary) / "data.json").write_text(
                json.dumps(
                    {
                        "mode": "prepare_raw",
                        "root": "fixture",
                        "loader": "raw",
                        "recipe": "tests/fixtures/corpus-recipe.json",
                    }
                )
            )
            self.bundle = load_workflow(Path(temporary) / "run.json", project_root=ROOT)

    def test_pending_data_fails_before_ingestion(self):
        prepare = controller("prepare_and_launch")
        with tempfile.TemporaryDirectory() as temporary:
            prepare.__globals__.update(
                Path=lambda path: Path(temporary) / str(path).lstrip("/"),
                json=json,
                record=Mock(),
                data_volume=Mock(),
                track_call=Mock(),
                modal=Mock(),
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
                        LAUNCH,
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
                    track_call=Mock(),
                    modal=Mock(),
                )
                spawn = continuation.with_options.return_value.spawn
                if invalid == "none":
                    launch("run", bundle, LAUNCH)
                    spawn.assert_called_once()
                    self.assertEqual(spawn.call_args.args[1], "v1/snapshots/prepared")
                    self.assertEqual(spawn.call_args.args[2], bundle)
                    # The training chain may not outlive the launch deadline.
                    timeout = continuation.with_options.call_args.kwargs["timeout"]
                    self.assertLessEqual(timeout, LAUNCH["deadline_unix"] - time.time() + 1)
                else:
                    with self.assertRaises(ValueError):
                        launch("run", bundle, LAUNCH)
                    spawn.assert_not_called()

    def test_growing_snapshot_launches_once_its_preparation_completes(self):
        import os
        import shutil
        import sqlite3

        from test_growing import build
        from training_data.export_raw import complete_growing, export_growing

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version = root / "data/fixture/v1"
            version.mkdir(parents=True)
            build(version, documents=2)
            recipe = {
                "sources": [{"name": "web", "weight_percent": 100, "repo": "r", "revision": "x"}]
            }
            db = sqlite3.connect(version / "raw.sqlite")
            db.execute("CREATE TABLE settings(id INTEGER PRIMARY KEY, value TEXT)")
            db.execute("INSERT INTO settings VALUES(1, ?)", (json.dumps({"recipe": recipe}),))
            db.commit()
            db.close()
            snapshot = version / "snapshots/overlap"
            export_growing(version, snapshot, ["audit"])
            bundle = load_workflow(ROOT / "workflows/h100x1-pilot.json")
            bundle["workflow"]["pilot"] = False
            bundle["dataset"].update(root="fixture", snapshot="v1/snapshots/overlap", loader="raw")
            target = bundle["workflow"]["target_tokens"]
            launch = controller("launch_snapshot")
            continuation = Mock()
            launch.__globals__.update(
                Path=lambda path: root / str(path).lstrip("/"),
                json=json,
                record=Mock(),
                continue_training=continuation,
                track_call=Mock(),
                modal=Mock(),
            )
            spawn = continuation.with_options.return_value.spawn
            with patch.dict(os.environ), patch("tempfile.mkdtemp", return_value=temporary):
                with self.assertRaisesRegex(ValueError, "resume-preparation"):
                    launch("run", bundle, LAUNCH)
                shutil.copyfile(version / "raw.sqlite", snapshot / "raw.sqlite")
                complete_growing(snapshot, capacity_verified_for_tokens=target - 1)
                with self.assertRaisesRegex(ValueError, "fewer tokens"):
                    launch("run", bundle, LAUNCH)
                spawn.assert_not_called()
                complete_growing(snapshot, capacity_verified_for_tokens=target)
                launch("run", bundle, LAUNCH)
            spawn.assert_called_once()
            self.assertEqual(spawn.call_args.args[1], "v1/snapshots/overlap")

    def test_expired_launch_deadline_refuses_work(self):
        launch = controller("launch_snapshot")
        continuation = Mock()
        launch.__globals__.update(record=Mock(), continue_training=continuation, track_call=Mock())
        bundle = load_workflow(ROOT / "workflows/h100x1-pilot.json")
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            launch("run", bundle, {"deadline_unix": time.time() + 60})
        continuation.with_options.assert_not_called()
