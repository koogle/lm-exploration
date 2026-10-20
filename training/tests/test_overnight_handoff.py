"""Exercise the actual Modal controller locally with fake calls; allocate no GPUs."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import modal_overnight as workflow


class HandoffTests(unittest.TestCase):
    def test_ready_audit_schedules_freeze_once_not_direct_gpu(self):
        calls = {
            "code": SimpleNamespace(
                get=lambda **kw: {"collection": {"fresh_code_tokens": 9_000_000_000}}
            ),
            "original": SimpleNamespace(get=lambda **kw: {"ready": True}),
            "audit": SimpleNamespace(get=lambda **kw: {"stage": "audited_not_training_ready"}),
        }
        with (
            patch.object(workflow, "mark"),
            patch.object(
                workflow.modal.FunctionCall, "from_id", side_effect=lambda key: calls[key]
            ),
            patch.object(
                workflow.audit_augmented, "spawn", return_value=SimpleNamespace(object_id="audit")
            ),
            patch.object(
                workflow.freeze_and_schedule,
                "spawn",
                return_value=SimpleNamespace(object_id="freeze"),
            ) as freeze,
            patch.object(workflow.train, "spawn") as train,
        ):
            result = workflow.wait_for_data.local(
                "test-overnight", "code", "original", {"commit": "fixture"}
            )
        freeze.assert_called_once_with("test-overnight", "original", {"commit": "fixture"}, 0)
        train.assert_not_called()
        self.assertEqual(result, {"preparation_call_id": "freeze"})

    def test_pending_data_only_chains_cpu_waiter(self):
        with (
            patch.object(workflow, "mark"),
            patch.object(workflow.time, "monotonic", side_effect=[0, 1, 3000]),
            patch.object(
                workflow.modal.FunctionCall,
                "from_id",
                return_value=SimpleNamespace(
                    get=lambda **kw: (_ for _ in ()).throw(TimeoutError())
                ),
            ),
            patch.object(
                workflow.wait_for_data, "spawn", return_value=SimpleNamespace(object_id="next")
            ) as again,
            patch.object(workflow.freeze_and_schedule, "spawn") as freeze,
            patch.object(workflow.train, "spawn") as train,
        ):
            result = workflow.wait_for_data.local("test-overnight", "code", "original", {})
        again.assert_called_once()
        freeze.assert_not_called()
        train.assert_not_called()
        self.assertEqual(result, {"continuation_call_id": "next"})

    def test_short_code_collects_more_without_gpu(self):
        with (
            patch.object(workflow, "mark"),
            patch.object(
                workflow.freeze_snapshot,
                "local",
                side_effect=ValueError("Need >=6.1B new eligible code targets"),
            ),
            patch.object(
                workflow.collect_more_code, "spawn", return_value=SimpleNamespace(object_id="more")
            ) as more,
            patch.object(
                workflow.wait_for_data, "spawn", return_value=SimpleNamespace(object_id="next")
            ),
            patch.object(workflow.train, "spawn") as train,
        ):
            result = workflow.freeze_and_schedule.local("test-overnight", "original", {}, 0)
        more.assert_called_once_with(1)
        train.assert_not_called()
        self.assertTrue(result["capacity_retry"])
