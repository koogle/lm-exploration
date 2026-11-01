"""Open-SWE-Traces (mini-swe-agent v2) and Terminus 2 formats for the agent mixes."""

import json
import unittest

from training_data import open_swe_format as swe
from training_data import qwen_agent_chat, terminus_format

EPISODE = [
    {"role": "system", "content": "harness system prompt"},
    {"role": "user", "content": "Repository: owner/repo\n\n# Broken parser\n\nIt fails on x."},
    {
        "role": "assistant",
        "content": "Look around.",
        "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
    },
    {"role": "tool", "content": "exit code: 0\na.py <b> & 'c'"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "submit", "arguments": "{}"}}],
    },
]


class OpenSweFormatTests(unittest.TestCase):
    def test_conversation(self):
        messages = swe.conversation(EPISODE)
        self.assertEqual(messages[0], {"role": "system", "content": swe.SYSTEM})
        self.assertTrue(
            messages[1]["content"].startswith(
                "<pr_description>\nConsider the following PR description:\n# Title\n"
                "Broken parser\n\n# Description\nIt fails on x.\n</pr_description>"
            )
        )
        self.assertIn("/testbed", messages[1]["content"])
        self.assertEqual(
            messages[3]["content"],
            '{\n  "returncode": 0,\n  "output": "a.py \\u003cb\\u003e \\u0026 \\u0027c\\u0027"\n}',
        )
        submit = json.loads(messages[4]["tool_calls"][0]["function"]["arguments"])["command"]
        self.assertIn(swe.SUBMIT, submit)

    def test_clipped_output_uses_the_elided_layout(self):
        clipped = "exit code: 1\nhead\n[... 1234 characters omitted ...]\ntail"
        self.assertEqual(
            json.loads(swe.observation(clipped)),
            {
                "returncode": 1,
                "output_head": "head",
                "output_tail": "tail",
                "elided_chars": 1234,
                "warning": "Output too long.",
            },
        )

    def test_parse(self):
        text = (
            'Done.\n<tool_call>\n{"name": "bash", "arguments": {"command": "git diff > patch.txt"}}'
            "\n</tool_call>"
        )
        self.assertEqual(swe.parse_assistant(text)["tool_calls"][0]["function"]["name"], "bash")
        submit = (
            '<tool_call>\n{"name": "bash", "arguments": {"command": "echo '
            'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"}}\n</tool_call>'
        )
        self.assertEqual(swe.parse_assistant(submit)["tool_calls"][0]["function"]["name"], "submit")

    def test_training_messages_drop_reasoning_and_render(self):
        row = {
            "messages": [
                {
                    "role": "system",
                    "content": swe.SYSTEM,
                    "reasoning_content": None,
                    "tool_calls": None,
                },
                {"role": "user", "content": "task", "reasoning_content": None, "tool_calls": None},
                {
                    "role": "assistant",
                    "content": "Look.",
                    "reasoning_content": "secret plan",
                    "tool_calls": [
                        {
                            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                            "id": "x",
                            "type": "function",
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{\n  "returncode": 0,\n  "output": ""\n}',
                    "reasoning_content": None,
                    "tool_calls": None,
                },
            ]
        }
        messages = swe.training_messages(row)
        self.assertNotIn("reasoning_content", messages[2])
        text = qwen_agent_chat.render(messages, swe.TOOLS)
        self.assertNotIn("secret plan", text)
        self.assertIn(
            '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>', text
        )
        self.assertIn("<tool_response>", text)


class TerminusFormatTests(unittest.TestCase):
    def test_both_layouts_and_think_removal(self):
        lego = [{"from": "human", "value": "task"}, {"from": "gpt", "value": '{"analysis": "a"}'}]
        nemotron = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '<think>\nhm\n</think>\n\n{"analysis": "a"}'},
        ]
        expected = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"analysis": "a"}'},
        ]
        self.assertEqual(terminus_format.training_messages(lego), expected)
        self.assertEqual(terminus_format.training_messages(nemotron), expected)
        self.assertIsNone(terminus_format.training_messages([{"from": "human", "value": "x"}]))


if __name__ == "__main__":
    unittest.main()
