"""Check the distinctions that drive the investigation's conclusions."""

import json
import unittest

from .audit_episodes import duplicate_segments, episode, paired
from .prepare_probes import views


def action(command, identifier):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": identifier,
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
    }


class InvestigationTests(unittest.TestCase):
    def test_censoring_does_not_reweight_remaining_seed(self):
        original = [
            {"id": task, "seed": seed, "resolved": False}
            for task, seeds in [("a", [1]), ("b", [1, 2, 3])]
            for seed in seeds
        ]
        edited = [{**row, "resolved": row["id"] == "a"} for row in original]
        estimate = paired(original, edited, metrics=("resolved",))
        self.assertEqual(estimate["resolved"]["difference"], 0.25)

    def test_pipeline_duplicates_and_write_boundaries(self):
        self.assertEqual(duplicate_segments("grep -v foo file | grep -v foo file"), 1)
        self.assertEqual(duplicate_segments("grep -v foo file | grep -v bar file"), 0)
        messages = []
        for i, command in enumerate(
            ["cat x", "cat x", "python -c \"p.write_text('new')\"", "cat x", "cat x"]
        ):
            messages += [
                action(command, str(i)),
                {"role": "tool", "content": "exit code: 0\nunchanged"},
            ]
        row = episode(
            {
                "messages": messages,
                "repository": "x/y",
                "sha": "1",
                "end": "max_turns",
                "resolved": False,
                "turns": 5,
            }
        )
        self.assertEqual(row["counts"]["exact_repeats"], 3)
        self.assertEqual(row["counts"]["same_result_repeats_without_detected_write"], 2)

    def test_context_dedup_ignores_call_ids_but_keeps_new_results(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Repository: x/y\n\n# Issue\n\nFix it"},
        ]
        for i, output in enumerate(["same", "same", "changed"]):
            messages += [
                action("cat x", str(i)),
                {"role": "tool", "tool_call_id": str(i), "content": "exit code: 0\n" + output},
            ]
        dedup = views(messages)["dedup"]
        self.assertEqual(len(dedup), 6)
        self.assertEqual(dedup[:2], messages[:2])
        self.assertTrue(dedup[-1]["content"].endswith("changed"))


if __name__ == "__main__":
    unittest.main()
