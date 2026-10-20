import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import modal_parallel_code as workflow


class CoordinatorTests(unittest.TestCase):
    def test_existing_remote_workers_are_reconnected_after_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            ids = [f"worker-{i}" for i in range(8)]
            (folder / "calls-r0.json").write_text(
                json.dumps({"workers": ids, "per_worker_target": 1100000000})
            )
            with (
                patch.object(workflow, "FOLDER", folder),
                patch.object(workflow.exclusions, "remote", return_value={}),
                patch.object(workflow.data_volume, "reload") as reload,
                patch.object(workflow.data_volume, "commit"),
                patch.object(
                    workflow.modal.FunctionCall,
                    "from_id",
                    return_value=SimpleNamespace(get=lambda **kw: {}),
                ) as reconnect,
                patch.object(workflow.partition, "spawn") as spawn,
                patch.object(
                    workflow.merge, "remote", return_value={"fresh_code_tokens": 9000000000}
                ),
            ):
                result = workflow.collect.local(0)
            reload.assert_called_once()
            self.assertEqual(reconnect.call_count, 8)
            spawn.assert_not_called()
            self.assertEqual(result["collection"]["fresh_code_tokens"], 9000000000)
            self.assertEqual(
                json.loads((folder / "coordinator.json").read_text())["stage"], "merged"
            )

    def test_new_ledger_directory_is_created_before_worker_bookkeeping(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "child-created-folder"
            calls = [SimpleNamespace(object_id=f"w{i}", get=lambda **kw: {}) for i in range(8)]
            with (
                patch.object(workflow, "FOLDER", folder),
                patch.object(workflow.exclusions, "remote", return_value={}),
                patch.object(workflow.data_volume, "reload"),
                patch.object(workflow.data_volume, "commit"),
                patch.object(workflow.partition, "spawn", side_effect=calls) as spawn,
                patch.object(
                    workflow.merge, "remote", return_value={"fresh_code_tokens": 9000000000}
                ),
            ):
                workflow.collect.local(0)
            self.assertEqual(spawn.call_count, 8)
            self.assertEqual(
                json.loads((folder / "calls-r0.json").read_text())["workers"],
                [f"w{i}" for i in range(8)],
            )
