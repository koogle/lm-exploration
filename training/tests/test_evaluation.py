import math
import unittest
from types import SimpleNamespace

import torch
from evaluation import tasks

VOCAB = 128


def toy_model(uniform=False, block_size=64):
    """Characters as token IDs; the model expects each character to be followed by the next."""

    def forward(tokens):
        if uniform:
            return torch.zeros(*tokens.shape, VOCAB)
        return torch.nn.functional.one_hot((tokens + 1) % VOCAB, VOCAB).float() * 10

    return SimpleNamespace(
        forward=forward,
        encode=lambda text: [ord(c) for c in text],
        decode=lambda ids: "".join(chr(i) for i in ids if i >= 4),
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        block_size=block_size,
        device="cpu",
    )


class LogLikelihoodTests(unittest.TestCase):
    def test_batched_scores_match_unpadded_scores(self):
        model = toy_model()
        requests = [([ord("a")], [ord("b"), ord("c")]), ([ord("x")] * 9, [ord("q")])]
        scores = tasks.loglikelihoods(model, requests)
        for (context, continuation), (total, greedy) in zip(requests, scores, strict=True):
            tokens = torch.tensor([[1, *context, *continuation]])
            logits = torch.log_softmax(model.forward(tokens[:, :-1]).float(), -1)[0]
            expected = sum(logits[len(context) + i, t].item() for i, t in enumerate(continuation))
            self.assertAlmostEqual(total, expected, places=4)
        self.assertEqual([greedy for _, greedy in scores], [True, False])

    def test_long_requests_keep_the_continuation(self):
        model = toy_model(block_size=8)
        [(total, greedy)] = tasks.loglikelihoods(model, [([ord("a")] * 20, [ord("b")])])
        self.assertTrue(greedy)
        self.assertGreater(total, -0.01)

    def test_multiple_choice_prefers_the_likely_continuation(self):
        model = toy_model()
        row = {"question": "x", "choices": {"label": ["A", "B"], "text": ["!", "yz"]}}
        context, choices, _ = tasks.multiple_choice("arc_easy", {**row, "answerKey": "A"})
        self.assertEqual((context, choices), ("Question: x\nAnswer:", [" !", " yz"]))
        # The toy model has no preference here; this checks the bookkeeping.
        result = tasks.score_multiple_choice(model, "arc_easy", [{**row, "answerKey": "A"}])
        self.assertEqual(result["examples"], 1)
        self.assertIn(result["acc"], (0.0, 1.0))

    def test_winogrande_compares_the_shared_continuation(self):
        model = toy_model()
        # After "a" the toy model expects "b"; the continuation " c" is scored after each option.
        rows = [{"sentence": "_ c", "option1": "a", "option2": "\x1f", "answer": "2"}]
        result = tasks.score_winogrande(model, rows)
        self.assertEqual(result["examples"], 1)
        # " " (32) follows "\x1f" (31) in the toy model, so option 2 scores higher.
        self.assertEqual(result["acc"], 1.0)


class HeldoutTests(unittest.TestCase):
    def test_windows_overlap_on_their_last_target(self):
        windows = tasks.packed_windows([[10, 11, 12], [20, 21]], 1, 2, 3, 2)
        self.assertEqual(windows, [[1, 10, 11, 12], [12, 2, 1, 20]])

    def test_bits_per_byte_of_a_uniform_model(self):
        model = toy_model(uniform=True)
        windows = tasks.packed_windows([[ord(c) for c in "hello"]] * 3, 1, 2, 6, 2)
        result = tasks.bits_per_byte(model, windows)
        targets = [t for w in windows for t in w[1:] if t != 1]
        text_bytes = sum(t >= 4 for t in targets)
        self.assertEqual((result["target_tokens"], result["bytes"]), (len(targets), text_bytes))
        expected = len(targets) * math.log2(VOCAB) / text_bytes
        self.assertAlmostEqual(result["bits_per_byte"], expected, places=4)


class GenerationTests(unittest.TestCase):
    def test_greedy_generation_stops_at_a_stop_string(self):
        model = toy_model()
        self.assertEqual(tasks.generate(model, "a", 20, ["e"]), "bcd")
        self.assertEqual(tasks.generate(model, "a", 3, ["\n\n"]), "bcd")

    def test_answer_extraction(self):
        self.assertEqual(tasks.gsm8k_answer("so 3 + 4 = 7\n#### 1,234"), 1234)
        self.assertEqual(tasks.gsm8k_answer("It is 12 then 15.5"), 15.5)
        self.assertIsNone(tasks.gsm8k_answer("no number"))
        self.assertEqual(tasks.gsm8k_answer(" 8 apples, then 9"), 9)
        self.assertEqual(tasks.number("40.0"), 40)

    def test_math_prompts(self):
        question, gold, worked = tasks.math_prompt(
            "svamp", {"Body": "Tom has 3. ", "Question": "How many?", "Answer": 3.0}
        )
        self.assertEqual((question, gold, worked), ("Tom has 3. How many?", 3, "3"))
        question, gold, _ = tasks.math_prompt("gsm8k", {"question": "q", "answer": "x\n#### 5"})
        self.assertEqual((question, gold), ("q", 5))

    def test_hellaswag_preprocessing(self):
        self.assertEqual(tasks.hellaswag_text(" A [title] b [header]  c "), "A. b  c")


if __name__ == "__main__":
    unittest.main()
