"""PR-task episodes in mini-swe-agent's bash-only format."""

import json
import unittest

from training_data import mini_swe_format as mini

EPISODE = [
    {"role": "system", "content": "harness system prompt"},
    {"role": "user", "content": "Repository: owner/repo\n\n# Broken parser\n\nIt fails on x."},
    {
        "role": "assistant",
        "content": "THOUGHT: Look around.\n\n```bash\nls -la\n```",
        "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls -la"}'}}],
    },
    {"role": "tool", "content": "exit code: 0\na.py\nb.py"},
    {"role": "assistant", "content": "THOUGHT: Submit.\n\n```bash\necho done\n```"},
    {"role": "tool", "content": "exit code: 2\n"},
]


class MiniSweFormatTests(unittest.TestCase):
    def test_conversation(self):
        messages = mini.conversation(EPISODE)
        self.assertEqual(messages[0], {"role": "system", "content": mini.SYSTEM})
        self.assertTrue(
            messages[1]["content"].startswith(
                "<pr_description>\nConsider the following PR description:\n"
                "Broken parser\nIt fails on x.\n</pr_description>\n\n<instructions>"
            )
        )
        self.assertTrue(messages[1]["content"].endswith("</instructions>"))
        self.assertIn(mini.SUBMIT, messages[1]["content"])
        self.assertEqual(messages[2]["content"], EPISODE[2]["content"])
        self.assertEqual(
            messages[3]["content"], "<returncode>0</returncode>\n<output>\na.py\nb.py\n</output>"
        )
        self.assertEqual(messages[5]["content"], "<returncode>2</returncode>\n<output>\n</output>")
        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "user"] * 1 + ["assistant", "user"],
        )

    def test_parse(self):
        bash = mini.parse_assistant("THOUGHT: x\n\n```bash\ngrep -n foo a.py\n```<|im_end|>")
        call = bash["tool_calls"][0]["function"]
        self.assertEqual(
            (call["name"], json.loads(call["arguments"])), ("bash", {"command": "grep -n foo a.py"})
        )
        self.assertEqual(bash["content"], "THOUGHT: x\n\n```bash\ngrep -n foo a.py\n```")
        submit = mini.parse_assistant(
            "THOUGHT: done\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A"
            " && git diff --cached\n```"
        )
        self.assertEqual(submit["tool_calls"][0]["function"]["name"], "submit")
        self.assertNotIn("tool_calls", mini.parse_assistant("THOUGHT: no command"))
        two = "```bash\nls\n```\n```bash\npwd\n```"
        self.assertNotIn("tool_calls", mini.parse_assistant(two))
        self.assertIn(
            "tool_calls", mini.parse_assistant("<think>\nhm\n</think>\n\n```bash\nls\n```")
        )

    def test_encode_masks_assistant_turns(self):
        messages = mini.conversation(EPISODE)
        ids, mask = mini.encode(lambda s: [ord(c) for c in s], messages)
        self.assertEqual("".join(map(chr, ids)), mini.render(messages))
        target = "".join(chr(i) for i, m in zip(ids, mask, strict=True) if m)
        self.assertEqual(
            target, EPISODE[2]["content"] + "<|im_end|>" + EPISODE[4]["content"] + "<|im_end|>"
        )
        prompt = mini.render(messages[:4], add_generation_prompt=True)
        self.assertTrue(prompt.endswith("</output><|im_end|>\n<|im_start|>assistant\n"))


if __name__ == "__main__":
    unittest.main()
