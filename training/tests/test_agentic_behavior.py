"""Behavior statistics of PR-task agent episodes."""

import unittest

from evaluation.agentic_behavior import behavior


def call(index, name, **arguments):
    import json

    return {"id": f"c{index}", "function": {"name": name, "arguments": json.dumps(arguments)}}


ROWS = [
    {
        "end": "submitted",
        "resolved": True,
        "patch": "diff --git a/x b/x\n",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [call(1, "bash", command="ls")]},
            {"role": "tool", "tool_call_id": "c1", "content": "exit code: 0\nx"},
            {"role": "assistant", "content": "", "tool_calls": [call(2, "bash", command="ls")]},
            {"role": "tool", "tool_call_id": "c2", "content": "exit code: 0\nx"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call(3, "edit_file", path="x", old="a", new="b")],
            },
            {"role": "tool", "tool_call_id": "c3", "content": "edited x"},
            {"role": "assistant", "content": "", "tool_calls": [call(4, "submit")]},
            {"role": "tool", "tool_call_id": "c4", "content": "submitted"},
        ],
    },
    {
        "end": "answered",
        "resolved": False,
        "patch": "",
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [call(1, "edit_file", path="y")]},
            {"role": "tool", "tool_call_id": "c1", "content": "error: `old` occurs 0 times"},
            # A repeated call id still pairs with its own result, by position.
            {"role": "assistant", "content": "", "tool_calls": [call(1, "write_file", path="z")]},
            {"role": "tool", "tool_call_id": "c1", "content": "wrote z"},
            {"role": "assistant", "content": "I am done."},
        ],
    },
]


class AgenticBehaviorTests(unittest.TestCase):
    def test_counts(self):
        stats = behavior(ROWS)
        self.assertEqual(stats["episodes"], 2)
        self.assertEqual(stats["resolved"], 1)
        self.assertEqual(stats["ends"], {"answered": 1, "submitted": 1})
        self.assertEqual(stats["assistant_turns"], 7)
        self.assertEqual(stats["turns_with_tool_call"], round(6 / 7, 4))
        self.assertEqual(stats["tool_calls"], 6)
        self.assertEqual(
            stats["calls_by_tool"],
            {"bash": 0.3333, "edit_file": 0.3333, "submit": 0.1667, "write_file": 0.1667},
        )
        # The second `ls` repeats the first; calls in another episode do not count.
        self.assertEqual(stats["repeated_calls"], 0.1667)
        self.assertEqual(stats["edits"], {"attempted": 3, "applied": 2})
        self.assertEqual(stats["episodes_with_patch"], 1)
        self.assertEqual(stats["episodes_submitted"], 1)

    def test_empty(self):
        self.assertIsNone(behavior([])["turns_with_tool_call"])


if __name__ == "__main__":
    unittest.main()
