"""Qwen3 chat rendering, loss mask and tool-call parsing for the reference agents."""

import json
import unittest

from training_data import pr_task_agent, qwen_agent_chat

MESSAGES = [
    {"role": "system", "content": "You are an engineer."},
    {"role": "user", "content": "Fix the bug"},
    {
        "role": "assistant",
        "content": "Look first.",
        "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
    },
    {"role": "tool", "content": "a.py"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "a.py"}}},
            {"function": {"name": "bash", "arguments": {"command": "cat b"}}},
        ],
    },
    {"role": "tool", "content": "x = 1"},
    {"role": "tool", "content": "no b"},
]


def row():
    return pr_task_agent.training_row(MESSAGES)["messages"]


class QwenAgentChatTests(unittest.TestCase):
    def test_render_layout(self):
        text = qwen_agent_chat.render(row(), pr_task_agent.TOOLS)
        assert text.startswith("<|im_start|>system\nYou are an engineer.\n\n# Tools")
        assert '"name": "submit"' in text
        assert (
            '<|im_start|>assistant\nLook first.\n<tool_call>\n{"name": "bash", "arguments": '
            '{"command": "ls"}}\n</tool_call><|im_end|>\n'
        ) in text
        # Consecutive tool results share one user turn.
        assert (
            "<|im_start|>user\n<tool_response>\nx = 1\n</tool_response>\n"
            "<tool_response>\nno b\n</tool_response><|im_end|>\n"
        ) in text
        assert "<think>" not in text

    def test_generation_prompt_chat_mode_only_adds_empty_think_at_the_end(self):
        plain = qwen_agent_chat.render(row(), pr_task_agent.TOOLS, add_generation_prompt=True)
        chat = qwen_agent_chat.render(
            row(), pr_task_agent.TOOLS, add_generation_prompt=True, chat=True
        )
        assert plain.endswith("<|im_start|>assistant\n")
        assert chat == plain + qwen_agent_chat.EMPTY_THINK
        assert chat.count("<think>") == 1

    def test_encode_masks_assistant_turns_and_their_end(self):
        ids, mask = qwen_agent_chat.encode(
            lambda s: [ord(c) for c in s], row(), pr_task_agent.TOOLS
        )
        text = "".join(map(chr, ids))
        assert text == qwen_agent_chat.render(row(), pr_task_agent.TOOLS)
        target = "".join(chr(i) for i, m in zip(ids, mask, strict=True) if m)
        first = qwen_agent_chat._assistant_body(row()[2])
        second = qwen_agent_chat._assistant_body(row()[4])
        assert target == first + second
        assert target.count("<|im_end|>") == 2
        assert "<|im_start|>" not in target

    def test_parse_round_trip(self):
        body = qwen_agent_chat._assistant_body(row()[4])
        message = qwen_agent_chat.parse_assistant("<think>\n\n</think>\n\n" + body)
        assert message["content"] == ""
        calls = [
            (c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in message["tool_calls"]
        ]
        assert calls == [("read_file", {"path": "a.py"}), ("bash", {"command": "cat b"})]

    def test_parse_text_and_malformed_calls(self):
        message = qwen_agent_chat.parse_assistant(
            "Done.\n<tool_call>\n{not json}\n</tool_call>\n<tool_call>\n"
            '{"name": "submit", "arguments": {}}\n</tool_call><|im_end|>trailing'
        )
        assert message["content"] == "Done."
        assert [c["function"]["name"] for c in message["tool_calls"]] == ["submit"]
        assert qwen_agent_chat.parse_assistant("Just text") == {
            "role": "assistant",
            "content": "Just text",
        }
        # An unfinished call (cut at the token limit) is no call and leaves no markup.
        cut = qwen_agent_chat.parse_assistant('Plan\n<tool_call>\n{"name": "bash"')
        assert cut == {"role": "assistant", "content": "Plan"}


if __name__ == "__main__":
    unittest.main()
