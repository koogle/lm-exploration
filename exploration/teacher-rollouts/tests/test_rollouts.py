import asyncio
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from core import (
    Store,
    digest,
    ignore_source,
    reserve_budget,
    select_best,
    summarize,
    supervise_call,
    validate,
    validate_config,
)
from runner import trial

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = json.loads((ROOT / "prompts.json").read_text())
CONFIG = json.loads((ROOT / "config.json").read_text())


class ValidationTests(unittest.TestCase):
    def test_code_semantics_and_no_execution(self):
        p = PROMPTS[0]
        self.assertTrue(
            validate(p, "def solve(x, lo, hi):\n return max(lo, min(x, hi))", "stop")[
                "accepted"
            ]
        )
        self.assertFalse(
            validate(p, "def solve(x, lo, hi):\n return x", "stop")["accepted"]
        )
        for source in [
            "import os\ndef solve(x, lo, hi): return x",
            "def solve(x, lo, hi): return __import__('os').system('false')",
            "def solve(x, lo, hi):\n while True: pass",
        ]:
            self.assertFalse(validate(p, source, "stop")["accepted"])

    def test_structured_strict_types_and_duplicate_keys(self):
        p = PROMPTS[-1]
        valid = json.dumps(p["expected"])
        self.assertTrue(validate(p, valid, "stop")["accepted"])
        for s in [
            valid.replace("true", "1"),
            valid.replace("3", '"3"'),
            '{"name":"Ada","count":3,"active":false,"active":true}',
            "```json\n" + valid + "\n```",
        ]:
            self.assertFalse(validate(p, s, "stop")["accepted"])
        self.assertIn("truncated", validate(p, valid, "length")["reasons"])

    def test_technical_rubric_quality_limit(self):
        p = PROMPTS[2]
        result = validate(
            p,
            json.dumps(
                {
                    "answer": 80,
                    "explanation": "Divide cache hits by all requests and multiply by 100.",
                }
            ),
            "stop",
        )
        self.assertTrue(result["accepted"])
        self.assertTrue(result["human_review_required"])
        self.assertFalse(
            validate(p, '{"answer":99,"explanation":"hits requests"}', "stop")[
                "accepted"
            ]
        )


class PersistenceTests(unittest.TestCase):
    def test_batched_checkpoints_flush_tail_before_validation_and_resume(self):
        c = {
            **CONFIG,
            "repeats": 1,
            "checkpoint_every": 3,
            "checkpoint_interval_seconds": 30,
        }
        with tempfile.TemporaryDirectory() as d:
            store = Store(d, {"config": c})
            validated = Path(d) / "validated"
            validated.mkdir()
            snapshots = []

            def commit():
                snapshots.append(
                    (
                        len(list(store.records.glob("*.json"))),
                        len(list(validated.glob("*.json"))),
                    )
                )

            with patch("runner.request", side_effect=TimeoutError("fixture")):
                result = asyncio.run(trial(c, PROMPTS[:5], store, 1, 0, "a", commit))
            self.assertEqual(snapshots, [(3, 0), (5, 0), (5, 5)])
            self.assertEqual(result["raw_checkpoint_count"], 2)
            (Path(d) / "c1-r0.json").unlink()
            with patch("runner.request") as request:
                resumed = asyncio.run(trial(c, PROMPTS[:5], store, 1, 0, "b", commit))
                request.assert_not_called()
            self.assertEqual(resumed["raw_checkpoint_count"], 0)
            self.assertIsNone(resumed["accepted_examples_per_hour"])
            self.assertEqual(len(list(store.records.glob("*.json"))), 5)

    def test_checkpoint_interval_is_checked_on_completion(self):
        c = {
            **CONFIG,
            "repeats": 1,
            "checkpoint_every": 48,
            "checkpoint_interval_seconds": 1,
        }
        clock = [0.0]

        def respond(*args):
            clock[0] += 2
            raise TimeoutError("fixture")

        with tempfile.TemporaryDirectory() as d:
            store = Store(d, {"config": c})
            (Path(d) / "validated").mkdir()
            snapshots = []
            with (
                patch("runner.request", side_effect=respond),
                patch("runner.time.perf_counter", side_effect=lambda: clock[0]),
            ):
                result = asyncio.run(
                    trial(
                        c,
                        PROMPTS[:2],
                        store,
                        1,
                        0,
                        "a",
                        lambda: snapshots.append(
                            len(list(store.records.glob("*.json")))
                        ),
                    )
                )
            self.assertEqual(snapshots, [1, 2, 2])
            self.assertEqual(result["raw_checkpoint_count"], 2)

    def test_conflict_and_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(d, {"revision": "abc"})
            r = {"example_id": digest("one"), "output": "answer"}
            self.assertTrue(store.put(r))
            self.assertFalse(store.put(r))
            with self.assertRaises(ValueError):
                store.put({**r, "output": "different"})
            with self.assertRaises(ValueError):
                Store(d, {"revision": "changed"})
            (store.records / (r["example_id"] + ".json")).write_text("broken")
            with self.assertRaises(ValueError):
                store.read(r["example_id"])

    def test_actual_dispatch_concurrency_and_persisted_errors(self):
        c = {**CONFIG, "repeats": 8}
        active = peak = 0
        lock = threading.Lock()
        seeds = []

        def respond(path, payload):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(active, peak)
                seeds.append(payload["seed"])
            time.sleep(0.02)
            with lock:
                active -= 1
            raise TimeoutError("synthetic fixture failure")

        with tempfile.TemporaryDirectory() as d:
            store = Store(d, {"config": c})
            (Path(d) / "validated").mkdir()
            with patch("runner.request", side_effect=respond):
                result = asyncio.run(
                    trial(c, PROMPTS[:1], store, 4, 0, "a", lambda: None)
                )
            self.assertEqual(peak, 4)
            self.assertEqual(sorted(seeds), list(range(c["seed"], c["seed"] + 8)))
            self.assertEqual(result["completed"], 0)
            self.assertEqual(len(list(store.records.glob("*.json"))), 8)
            r = json.loads(next((Path(d) / "validated").glob("*.json")).read_text())
            self.assertEqual(r["validation"]["reasons"], ["request_error"])
            self.assertEqual(r["teacher_revision"], CONFIG["revision"])

    def test_resume_skips_durable_records_and_suppresses_rates(self):
        c = {**CONFIG, "repeats": 1}
        manifest = {"config": c, "prompts": PROMPTS[-2:]}
        response = {
            "choices": [
                {
                    "message": {"content": '{"values":[-1,2,4,7]}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 20},
        }
        with tempfile.TemporaryDirectory() as d:
            store = Store(d, manifest)
            (Path(d) / "validated").mkdir()
            with patch("runner.request", return_value=response) as req:
                first = asyncio.run(
                    trial(c, PROMPTS[-2:], store, 1, 0, "a", lambda: None)
                )
                self.assertEqual(req.call_count, 2)
                self.assertEqual(first["accepted"], 1)
            records = sorted(store.records.glob("*.json"))
            raw = json.loads(records[0].read_text())
            self.assertEqual(raw["prompt_tokens"], 42)
            self.assertEqual(raw["request"]["seed"], c["seed"])
            self.assertIsNone(raw["validation"])
            records[1].unlink()
            (Path(d) / "c1-r0.json").unlink()
            with patch("runner.request", return_value=response) as req:
                resumed = asyncio.run(
                    trial(c, PROMPTS[-2:], store, 1, 0, "b", lambda: None)
                )
                self.assertEqual(req.call_count, 1)
                self.assertFalse(resumed["comparable"])
                self.assertIsNone(resumed["accepted_examples_per_hour"])
            with patch("runner.request") as req:
                asyncio.run(trial(c, PROMPTS[-2:], store, 1, 0, "c", lambda: None))
                req.assert_not_called()
            self.assertEqual(len(list(store.records.glob("*.json"))), 2)


class BudgetAndMetricsTests(unittest.TestCase):
    def test_no_authorization_and_cumulative_reservations(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                reserve_budget(CONFIG, d, "a")
            c = {
                **CONFIG,
                "budget_authorized": True,
                "authorization_reference": "test fixture only",
                "campaign_usd_cap": 5.0,
            }
            self.assertEqual(reserve_budget(c, d, "a"), 2.1)
            with self.assertRaises(ValueError):
                reserve_budget(c, d, "a")
            reserve_budget(c, d, "b")
            with self.assertRaises(ValueError):
                reserve_budget(c, d, "c")

    def test_ranking_counts_zero_acceptance_and_requires_complete_rounds(self):
        def row(c, n, accepted, seconds=10, comparable=True):
            return {
                "concurrency": c,
                "round": n,
                "accepted": accepted,
                "steady_generation_seconds": seconds,
                "comparable": comparable,
            }

        rows = [row(1, 0, 10), row(1, 1, 0), row(4, 0, 6), row(4, 1, 6)]
        self.assertEqual(select_best(rows, 2), 4)
        self.assertEqual(select_best(rows[:3], 2), 1)
        self.assertIsNone(select_best([row(1, 0, 0), row(1, 1, 0)], 2))

    def test_launch_rejects_before_modal_import(self):
        import launch

        with (
            patch.object(
                sys,
                "argv",
                ["launch.py", "--run-id", "test", "--attempt", "test", "--launch"],
            ),
            patch.dict(sys.modules, {"modal_app": None}),
            self.assertRaisesRegex(ValueError, "No explicit budget authorization"),
        ):
            launch.main()

    def test_remote_deadline_explicitly_cancels_child(self):
        from unittest.mock import Mock

        call = Mock()
        call.get.side_effect = TimeoutError("still queued")
        with self.assertRaises(TimeoutError):
            supervise_call(call, 100, now=lambda: 90)
        call.get.assert_called_once_with(timeout=10)
        call.cancel.assert_called_once_with(terminate_containers=True)
        expired = Mock()
        with self.assertRaises(TimeoutError):
            supervise_call(expired, 100, now=lambda: 101)
        expired.get.assert_not_called()
        expired.cancel.assert_called_once_with(terminate_containers=True)

    def test_image_sources_exclude_nested_caches_and_run_outputs(self):
        for path in [
            "megakernel/__pycache__/probe.pyc",
            "artifacts/run/results.json",
            ".budget/ledger.json",
            "megakernel/kernel.so",
        ]:
            self.assertTrue(ignore_source(Path(path)))
        for path in ["megakernel/kernel.cu", "megakernel/model.py", "prompts.json"]:
            self.assertFalse(ignore_source(Path(path)))

    def test_invalid_caps(self):
        for key, value in [
            ("runtime_seconds", 1201),
            ("concurrency", [1, 1]),
            ("attempt_usd_cap", 0.1),
            ("gpu_usd_per_hour", float("nan")),
        ]:
            with self.assertRaises(ValueError):
                validate_config({**CONFIG, key: value})

    def test_raw_accepted_and_zero_acceptance(self):
        record = {
            "error": None,
            "output_tokens": 50,
            "prompt_tokens": 10,
            "finish_reason": "length",
            "validation": {"accepted": False},
            "verification_seconds": 0.1,
        }
        s = summarize([record], 10, 3.6)
        self.assertEqual(s["generated_tokens_per_second"], 5)
        self.assertEqual(s["completed_examples_per_hour"], 360)
        self.assertEqual(s["accepted_examples_per_hour"], 0)
        self.assertEqual(s["truncation_rate"], 1)
        self.assertIsNone(s["estimated_steady_gpu_usd_per_accepted"])
        self.assertIsNone(s["actual_billed_usd"])


if __name__ == "__main__":
    unittest.main()
