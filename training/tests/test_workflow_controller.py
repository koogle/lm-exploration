"""Exercise generic controller routing and preflight without importing Modal apps."""

import ast
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from workflows.config import pretrain_bundle
from workflows.plan import load_workflow as load_plan
from workflows.plan import select

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = {"deadline_unix": time.time() + 86400, "workflow_run_id": "run"}


def load_workflow(name="h100x1-pilot.json", dataset_config=None):
    """The resolved pretraining stage of a shipped workflow."""
    entry = dict(json.loads((ROOT / "workflows" / name).read_text())["stages"][-1])
    dataset = entry.pop("dataset_config")
    return pretrain_bundle(entry, "subword", dataset_config or dataset, ROOT)


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
        self.bundle = load_workflow(dataset_config="workflows/datasets/climbmix-code-v4-raw.json")

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
                            "stage": {**self.bundle["stage"], "allow_unreviewed_data": False},
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
                bundle = load_workflow()
                bundle["dataset"].update(root="other-corpus", snapshot="v1/snapshots/prepared")
                if invalid == "review":
                    bundle["stage"]["allow_unreviewed_data"] = False
                if invalid == "capacity":
                    bundle["stage"]["pilot"] = False
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
            bundle = load_workflow()
            bundle["stage"]["pilot"] = False
            bundle["dataset"].update(root="fixture", snapshot="v1/snapshots/overlap", loader="raw")
            target = bundle["stage"]["target_tokens"]
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
                with self.assertRaisesRegex(ValueError, "workflow run again"):
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
        bundle = load_workflow()
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            launch("run", bundle, {"deadline_unix": time.time() + 60})
        continuation.with_options.assert_not_called()


class RunWorkflowTests(unittest.TestCase):
    """`run_workflow` with its stage runners replaced: order, parents, branches, relaunch."""

    def run_plan(self, root, stages, start="", end="", checkpoint="", fail=(), reasons=None):
        path = Path(root, "workflow.json")
        path.write_text(
            json.dumps({"family": "subword", "from": "base/step-000000100.pt", "stages": stages})
        )
        plan = load_plan(path, "flow", ROOT)
        run = controller("run_workflow")
        calls = []

        def run_stage(kind, family, recipe, launch, deadline, examples):
            calls.append(recipe)
            call = Mock()
            if recipe["run_id"].removeprefix("flow-").split("-a")[0] in fail:
                call.get.side_effect = RuntimeError("training failed")
            else:
                call.get.return_value = {
                    "checkpoint": f"/runs/{recipe['run_id']}/step-1.pt",
                    **({"stop_reason": reasons.pop(0)} if reasons else {}),
                }
            return call

        run.__globals__.update(
            Path=lambda path: Path(root) / str(path).lstrip("/"),
            json=json,
            time=time,
            record=Mock(),
            run_volume=Mock(),
            track_call=Mock(),
            modal=Mock(),
            STAGE_RESERVE_SECONDS=1800,
            modal_post_train=Mock(run_stage=run_stage),
            relative=lambda path: str(Path(path).relative_to("/runs")),
        )
        result = run(plan, select(plan, start, end), LAUNCH, checkpoint)
        return result, {recipe["run_id"]: recipe["parent"] for recipe in calls}

    def test_chains_post_training_and_skips_finished_stages(self):
        sft = {"stage": "sft", "data": {"volume": "math-data"}, "settings": {"steps": 1}}
        stages = [
            {**sft, "name": "chat"},
            {"stage": "sft", "branches": [{**sft, "name": "a"}, {**sft, "name": "b"}]},
        ]
        with tempfile.TemporaryDirectory() as root:
            result, parents = self.run_plan(root, stages, end="chat")
            self.assertEqual(parents, {"flow-chat": "base/step-000000100.pt"})
            self.assertEqual(result["chat"]["output"]["checkpoint"], "flow-chat/step-1.pt")
            # A later launch continues the same workflow run with the unfinished stages.
            _, parents = self.run_plan(root, stages)
            self.assertEqual(
                parents, {"flow-a": "flow-chat/step-1.pt", "flow-b": "flow-chat/step-1.pt"}
            )

    def test_sequential_post_training_waits_for_its_parent(self):
        sft = {"stage": "sft", "data": {"volume": "d"}, "settings": {"steps": 1}}
        stages = [{**sft, "name": "chat"}, {**sft, "name": "more"}]
        with tempfile.TemporaryDirectory() as root:
            result, parents = self.run_plan(root, stages)
            self.assertEqual(parents["flow-more"], "flow-chat/step-1.pt")
            self.assertEqual(result["more"]["status"], "completed")

    def test_rl_stopped_short_pauses_and_resumes_its_own_run(self):
        stages = [{"stage": "rl", "name": "rl", "pool": "pool", "settings": {"steps": 100}}]
        with tempfile.TemporaryDirectory() as root:
            result, parents = self.run_plan(root, stages, reasons=["deadline"])
            self.assertEqual(result["rl"]["status"], "paused")
            result, parents = self.run_plan(root, stages, reasons=["steps"])
            self.assertEqual(result["rl"]["status"], "completed")
            # The second launch continued run `flow-rl`, not a new attempt's run.
            self.assertEqual(list(parents), ["flow-rl"])
            self.assertEqual(result["rl"]["attempt"], 2)

    def test_failed_stage_runs_again_as_a_new_attempt(self):
        stages = [
            {"stage": "sft", "name": "chat", "data": {"volume": "d"}, "settings": {"steps": 1}}
        ]
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                self.run_plan(root, stages, fail=("chat",))
            result, parents = self.run_plan(root, stages, checkpoint="other/step-000000002.pt")
            self.assertEqual(parents, {"flow-chat-a2": "other/step-000000002.pt"})
            self.assertEqual(result["chat"]["status"], "completed")

    def test_unfinished_parent_needs_a_checkpoint(self):
        sft = {"stage": "sft", "data": {"volume": "d"}, "settings": {"steps": 1}}
        stages = [{**sft, "name": "chat"}, {**sft, "name": "more"}]
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "--checkpoint"):
                self.run_plan(root, stages, start="more")
            _, parents = self.run_plan(root, stages, start="more", checkpoint="x/step-000000003.pt")
            self.assertEqual(parents, {"flow-more": "x/step-000000003.pt"})
