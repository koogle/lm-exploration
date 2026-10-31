"""Chat SFT: smoltalk2 conversion, GLM rendering with loss masks, packing and resume."""

import json
import unittest

import numpy as np
from training_data import chat, chat_sft, packed_shards

from tokenizer import glm

XLAM_ROW = {
    "messages": [
        {"role": "user", "content": "Giveaways for beta?"},
        {
            "role": "assistant",
            "content": '<tool_call>\n{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}\n</tool_call>',
        },
    ],
    "chat_template_kwargs": {
        "custom_instructions": "",
        "enable_thinking": False,
        "python_tools": [],
        "xml_tools": [
            '<tools>\n[{"type": "function", "function": {"name": "live_giveaways_by_type", "parameters": {}}}]\n</tools>'
        ],
    },
}
HERMES_ROW = {
    "messages": [
        {"role": "user", "content": "Feed and record?"},
        {
            "role": "assistant",
            "content": "<tool_call>\n{'arguments': {'id': 'a'}, 'name': 'feed'}\n</tool_call>\n"
            "<tool_call>\n{'arguments': {'id': 'a', 'minutes': 30}, 'name': 'record'}\n</tool_call>",
        },
        {
            "role": "tool",
            "content": '<tool_response>\n{"name": "feed", "content": {"url": "u"}}\n</tool_response>\n'
            '<tool_response>\n{"name": "record", "content": {"ok": true}}\n</tool_response>',
        },
        {"role": "assistant", "content": "Here is the feed u; recording started."},
    ],
    "chat_template_kwargs": {"custom_instructions": "Be brief.", "xml_tools": []},
}


def template_available():
    try:
        chat.template()
        glm.load()
        return True
    except Exception:  # noqa: BLE001 - offline machines skip the rendering tests
        return False


class ConversionTests(unittest.TestCase):
    def test_xlam_tool_calls_become_openai_tool_calls(self):
        out = chat_sft.convert(XLAM_ROW)
        call = out["messages"][1]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "live_giveaways_by_type")
        self.assertEqual(json.loads(call["arguments"]), {"type": "beta"})
        self.assertEqual(out["messages"][1]["content"], "")
        self.assertEqual(out["tools"][0]["function"]["name"], "live_giveaways_by_type")

    def test_hermes_python_dicts_and_split_tool_responses(self):
        out = chat_sft.convert(HERMES_ROW)
        roles = [m["role"] for m in out["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "tool", "assistant"])
        calls = out["messages"][2]["tool_calls"]
        self.assertEqual([c["function"]["name"] for c in calls], ["feed", "record"])
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {"id": "a", "minutes": 30})
        self.assertEqual(json.loads(out["messages"][3]["content"])["name"], "feed")

    def test_thinking_and_unfinished_rows_are_dropped(self):
        thinking = {**XLAM_ROW, "chat_template_kwargs": {"enable_thinking": True}}
        self.assertIsNone(chat_sft.convert(thinking))
        unfinished = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertIsNone(chat_sft.convert(unfinished))

    def test_tool_instructions_go_to_system_and_trailing_turns_are_cut(self):
        block = (
            "Signatures are within <tools></tools> XML tags:\n"
            + XLAM_ROW["chat_template_kwargs"]["xml_tools"][0]
        )
        row = {
            "messages": [*XLAM_ROW["messages"], {"role": "user", "content": "thanks"}],
            "chat_template_kwargs": {"xml_tools": [block]},
        }
        out = chat_sft.convert(row)
        self.assertEqual(
            out["messages"][0],
            {
                "role": "system",
                "content": "Signatures are within <tools></tools> XML tags:",
            },
        )
        self.assertEqual(out["messages"][-1]["role"], "assistant")
        self.assertEqual(out["tools"][0]["function"]["name"], "live_giveaways_by_type")


class PackingTests(unittest.TestCase):
    def conversations(self):
        rng = np.random.default_rng(0)
        out = []
        for length in (5, 7, 3, 9, 4):
            tokens = rng.integers(10, 100, length).astype(np.uint32)
            tokens[-1] = glm.EOS_ID
            mask = (rng.random(length) < 0.5).astype(np.uint8)
            out.append((tokens, mask))
        return out

    def test_windows_keep_whole_conversations_and_mask_labels(self):
        windows = list(chat_sft.pack(self.conversations(), 11))
        for inputs, labels in windows:
            self.assertEqual(inputs.shape, (11,))
            self.assertEqual(labels.shape, (11,))
        flat = np.concatenate([t for t, _ in self.conversations()])
        first, labels = windows[0]
        self.assertEqual(first[:5].tolist(), flat[:5].tolist())
        conv = self.conversations()[0]
        expected = np.where(conv[1][1:].astype(bool), conv[0][1:].astype(np.int64), -100)
        self.assertEqual(labels[:4].tolist(), expected.tolist())

    def test_mixture_resumes_where_it_stopped(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            chat_sft._write(Path(folder), "a", self.conversations())
            chat_sft._write(Path(folder), "b", self.conversations()[::-1])
            recipe = {"files": ["a.parquet", "b.parquet"], "context": 11}
            mixture = chat_sft.ChatMixture(recipe, folder, seed=3)
            first = mixture.next_batch(2)
            state = mixture.state_dict()
            second = mixture.next_batch(2)
            resumed = chat_sft.ChatMixture(recipe, folder, seed=3)
            resumed.load_state_dict(state)
            again = resumed.next_batch(2)
            np.testing.assert_array_equal(second[0], again[0])
            np.testing.assert_array_equal(second[1], again[1])
            self.assertFalse(np.array_equal(first[0], second[0]))

    def test_subsets_prepared_elsewhere_are_read_with_repeats(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            for name in ("chat", "agentic"):
                Path(folder, name).mkdir()
            chat_sft._write(Path(folder, "chat"), "a", self.conversations())
            chat_sft._write(Path(folder, "agentic"), "b", self.conversations()[:2])
            recipe = {
                "name": "agentic",
                "context": 11,
                "subsets": [
                    {"name": "a", "prepared_in": "chat", "repeat": 0.4},
                    {"name": "b", "repeat": 3},
                ],
            }
            mixture = chat_sft.ChatMixture(recipe, Path(folder, "agentic"))
            self.assertEqual(len(mixture.order), 2 + 3 * 2)
            again = chat_sft.ChatMixture(recipe, Path(folder, "agentic"))
            self.assertEqual(mixture.order, again.order)
            self.assertEqual(
                chat_sft.subset_folder(recipe, Path(folder, "agentic"), "a"), Path(folder, "chat")
            )

    def test_replay_rows_are_pretraining_windows_after_the_chat_rows(self):
        import tempfile
        from pathlib import Path

        class Pretraining:
            wait_seconds = 0

            def __init__(self):
                self.position = 0

            def next_batch(self, rows):
                tokens = np.arange(self.position, self.position + rows * 12).reshape(rows, 12)
                self.position += rows * 12
                return tokens[:, :-1], tokens[:, 1:], ["climbmix"] * rows

            def state_dict(self):
                return {"position": self.position}

            def load_state_dict(self, state):
                self.position = state["position"]

        with tempfile.TemporaryDirectory() as folder:
            chat_sft._write(Path(folder), "a", self.conversations())
            chat_sft._write(Path(folder), "b", self.conversations()[::-1])
            recipe = {"files": ["a.parquet", "b.parquet"], "context": 11}
            mixture = packed_shards.ReplayMixture(
                chat_sft.ChatMixture(recipe, folder), Pretraining(), replay_rows=1
            )
            inputs, labels, names = mixture.next_batch(3)
            self.assertEqual(names, ["chat", "chat", "climbmix"])
            self.assertEqual(inputs.shape, (3, 11))
            self.assertTrue((labels[2] >= 0).all())
            self.assertTrue((labels[:2] == chat_sft.IGNORE_INDEX).any())
            state = mixture.state_dict()
            second = mixture.next_batch(2)
            resumed = packed_shards.ReplayMixture(
                chat_sft.ChatMixture(recipe, folder), Pretraining(), replay_rows=1
            )
            resumed.load_state_dict(state)
            again = resumed.next_batch(2)
            np.testing.assert_array_equal(second[0], again[0])
            np.testing.assert_array_equal(second[1], again[1])


@unittest.skipUnless(template_available(), "GLM tokenizer or chat template not downloadable")
class RenderingTests(unittest.TestCase):
    def test_thinking_render_keeps_every_turn_reasoning(self):
        messages = [
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "<think> step one </think>\nAnswer 42."},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "<think>x</think> final"},
        ]
        text = chat.render(messages, thinking=True)
        self.assertIn("<|assistant|><think>step one</think>Answer 42.", text)
        self.assertIn("<|assistant|><think>x</think>final", text)
        self.assertNotIn("<think>", chat.render(messages).split("again")[0])

    def test_loss_mask_covers_assistant_output_and_turn_ends(self):
        tokenizer = glm.load()
        conversation = chat_sft.convert(HERMES_ROW)
        ids, mask = chat.encode(tokenizer, conversation["messages"], conversation["tools"])
        trained = tokenizer.decode([i for i, m in zip(ids, mask) if m], skip_special_tokens=False)
        self.assertIn(
            "<tool_call>feed<arg_key>id</arg_key><arg_value>a</arg_value></tool_call>", trained
        )
        self.assertIn("<|observation|>", trained)
        self.assertIn("Here is the feed u; recording started.", trained)
        self.assertTrue(trained.endswith("<|endoftext|>"))
        self.assertNotIn("Feed and record?", trained)
        self.assertNotIn("Be brief.", trained)
        self.assertNotIn("<tool_response>", trained)

    def test_generated_tool_calls_parse_back_to_openai(self):
        text = "<tool_call>record<arg_key>id</arg_key><arg_value>a</arg_value><arg_key>minutes</arg_key><arg_value>30</arg_value></tool_call><|observation|>"
        message = chat.parse_assistant(text)
        self.assertIsNone(message["content"])
        arguments = json.loads(message["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments, {"id": "a", "minutes": 30})


if __name__ == "__main__":
    unittest.main()
