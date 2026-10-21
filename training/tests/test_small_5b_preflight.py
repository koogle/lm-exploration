"""Exercise generic controller routing and preflight without importing Modal apps."""

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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


class Small5BPreflightTests(unittest.TestCase):
    def setUp(self):
        self.bundle = load_workflow(ROOT / "workflows/small-5b.json")

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
                prepare(run_id, version, self.bundle, {})

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
                resume(run_id, snapshot, self.bundle, {}, first_segment=segment)

    def test_controller_uses_configured_qualification_and_continuation(self):
        resume = controller("continue_training")
        self.bundle["workflow"].update(
            max_segments=5,
            segments_per_call=2,
            qualification_segments=1,
            qualification_steps=7,
            gpu="L40S",
        )
        validation = {"weighted_loss": 2.0}
        reports = [
            {
                "total_loss_tokens": tokens,
                "initial_validation": validation,
                "final_validation": validation,
                "final_checkpoint": f"/runs/run-s{segment:02d}/step-000000007.pt",
                "stop_reason": "step_limit" if segment == 0 else "time_limit",
            }
            for segment, tokens in enumerate((100, 200))
        ]
        train_segment = Mock()
        train_segment.with_options.return_value.remote.side_effect = reports
        continuation = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            # The real controller writes one report per segment in this directory.
            (Path(temporary) / "runs/run").mkdir(parents=True)
            resume.__globals__.update(
                json=json,
                record=Mock(),
                run_volume=Mock(),
                train_segment=train_segment,
                continue_training=continuation,
            )

            # Resume paths are interpreted relative to /runs by production code;
            # use real Paths there and redirect only the report directory.
            def local_path(path):
                return Path(temporary) / "runs" if path == "/runs" else Path(path)

            resume.__globals__["Path"] = local_path
            resume("run", "data-v1/snapshots/run", self.bundle, {})
        self.assertEqual(train_segment.with_options.return_value.remote.call_count, 2)
        self.assertEqual(
            train_segment.with_options.return_value.remote.call_args_list[0].args[-1], 7
        )
        self.assertIsNone(train_segment.with_options.return_value.remote.call_args_list[1].args[-1])
        self.assertEqual(continuation.spawn.call_args.args[-1], 2)
        train_segment.with_options.assert_called_with(gpu="L40S:3", cpu=12, timeout=25200)
        self.assertEqual(continuation.spawn.call_args.args[2], self.bundle)

    def test_manual_resume_forwards_checkpoint_and_requires_new_progress(self):
        resume = controller("continue_training")
        segment = Mock()
        segment.with_options.return_value.remote.return_value = {
            "total_loss_tokens": 100,
            "session_loss_tokens": 0,
            "final_validation": {"weighted_loss": 2.0},
            "stop_reason": "time_limit",
        }
        resume.__globals__.update(record=Mock(), train_segment=segment)
        with self.assertRaisesRegex(ValueError, "no progress"):
            resume(
                "run",
                "v1/snapshots/prepared",
                self.bundle,
                {},
                initial_resume="previous/step-000000007.pt",
            )
        self.assertEqual(
            segment.with_options.return_value.remote.call_args.args[4], "previous/step-000000007.pt"
        )

    def test_selected_dataset_and_model_reach_gpu_worker(self):
        segment = controller("train_segment")
        bundle = load_workflow(
            ROOT / "workflows/small-5b.json", model_config="model_configs/modern-124m.json"
        )
        bundle["dataset"]["root"] = "other-corpus"
        worker = Mock()
        with patch.dict(sys.modules, {"distributed.modal_train": SimpleNamespace(worker=worker)}):
            segment("run-s00", "v1/snapshots/prepared", bundle, {})
        settings = worker.local.call_args.args[2]
        self.assertEqual(settings["n_layer"], 14)
        self.assertEqual(settings["target_tokens"], 5_000_000_000)
        self.assertEqual(worker.local.call_args.kwargs["dataset_root"], "other-corpus")

    def test_snapshot_selection_preflight_before_gpu_scheduling(self):
        from check_distributed_cuda import fixture

        for invalid in ("none", "review", "capacity", "loader", "validation"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                parent = root / "data/other-corpus/v1/snapshots"
                parent.mkdir(parents=True)
                manifest = fixture(parent / "prepared")
                bundle = load_workflow(ROOT / "workflows/h100x1-pilot.json")
                bundle["dataset"].update(root="other-corpus", snapshot="v1/snapshots/prepared")
                if invalid == "review":
                    bundle["workflow"]["allow_unreviewed_data"] = False
                if invalid == "capacity":
                    bundle["workflow"]["pilot"] = False
                if invalid == "loader":
                    bundle["dataset"]["loader"] = "raw"
                if invalid == "validation":
                    value = json.loads(manifest.read_text())
                    value["sources"]["short"]["splits"]["validation"]["loss_tokens"] = 0
                    manifest.write_text(json.dumps(value))
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

    def test_preparation_resume_rejects_a_changed_selection(self):
        prepare = controller("prepare_and_launch")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "runs/run"
            folder.mkdir(parents=True)
            (folder / "workflow.json").write_text('{"stage": "preparing"}')
            (folder / "resolved-workflow.json").write_text("{}")
            prepare.__globals__.update(Path=lambda path: root / str(path).lstrip("/"), json=json)
            with self.assertRaisesRegex(ValueError, "identical resolved workflow"):
                prepare("run", "v1", self.bundle, {}, resume_preparation=True)

    def test_preparation_uses_workflow_settings_and_records_them(self):
        prepare = controller("prepare_and_launch")
        self.bundle["workflow"].update(
            source_workers=2,
            tokenizer_threads=4,
            database_batch_size=16,
            preparation_chunk_seconds=9,
        )
        self.bundle["dataset"]["root"] = "different-corpus"
        continuation = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data/different-corpus/data-v1").mkdir(parents=True)
            prepare.__globals__.update(
                Path=lambda path: root / str(path).lstrip("/"),
                json=json,
                record=Mock(),
                data_volume=Mock(),
                continue_training=continuation,
            )

            def export(folder, destination):
                destination.mkdir(parents=True)
                return {}

            with (
                patch.dict(os.environ),
                patch("training_data.prepared_snapshot.find_prepared_snapshot", return_value=None),
                patch("training_data.export_raw.export_raw", side_effect=export),
                patch("training_data.fast_raw_corpus.FastRawCorpus") as corpus,
            ):
                corpus.return_value.prepare_parallel.return_value = {"ready": True}
                corpus.return_value.report.return_value = {
                    "supported_mixture_exposures": self.bundle["workflow"]["target_tokens"],
                    "sources": {"fixture": {"validation_loss_tokens": 10}},
                }
                prepare("run", "data-v1", self.bundle, {})
                self.assertEqual(os.environ["RAYON_NUM_THREADS"], "4")
                self.assertEqual(corpus.call_args.args[1], self.bundle["preparation_recipe"])
                self.assertEqual(corpus.call_args.kwargs["workers"], 2)
                self.assertEqual(corpus.call_args.kwargs["batch_size"], 16)
                corpus.return_value.prepare_parallel.assert_called_once_with(
                    self.bundle["workflow"]["target_tokens"], max_seconds=9
                )
                recorded = continuation.spawn.call_args.args[3]["resolved_workflow"]
                self.assertEqual(recorded, self.bundle)
