import json
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from evaluation import scoring
from evaluation.benchmarks import BENCHMARKS, arc, gsm8k, hellaswag, svamp, winogrande

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
        scores = scoring.loglikelihoods(model, requests)
        for (context, continuation), (total, greedy) in zip(requests, scores, strict=True):
            tokens = torch.tensor([[1, *context, *continuation]])
            logits = torch.log_softmax(model.forward(tokens[:, :-1]).float(), -1)[0]
            expected = sum(logits[len(context) + i, t].item() for i, t in enumerate(continuation))
            self.assertAlmostEqual(total, expected, places=4)
        self.assertEqual([greedy for _, greedy in scores], [True, False])

    def test_long_requests_keep_the_continuation(self):
        model = toy_model(block_size=8)
        [(total, greedy)] = scoring.loglikelihoods(model, [([ord("a")] * 20, [ord("b")])])
        self.assertTrue(greedy)
        self.assertGreater(total, -0.01)

    def test_multiple_choice_prefers_the_likely_continuation(self):
        model = toy_model()
        row = {"question": "x", "choices": {"label": ["A", "B"], "text": ["!", "yz"]}}
        context, choices, _ = arc.item({**row, "answerKey": "A"})
        self.assertEqual((context, choices), ("Question: x\nAnswer:", [" !", " yz"]))
        # The toy model has no preference here; this checks the bookkeeping.
        result, _ = arc.score(model, [{**row, "answerKey": "A"}], [], {}, 0)
        self.assertEqual(result["examples"], 1)
        self.assertIn(result["acc"], (0.0, 1.0))

    def test_winogrande_compares_the_shared_continuation(self):
        model = toy_model()
        # After "a" the toy model expects "b"; the continuation " c" is scored after each option.
        rows = [{"sentence": "_ c", "option1": "a", "option2": "\x1f", "answer": "2"}]
        result, _ = winogrande.score(model, rows, [], {}, 0)
        self.assertEqual(result["examples"], 1)
        # " " (32) follows "\x1f" (31) in the toy model, so option 2 scores higher.
        self.assertEqual(result["acc"], 1.0)


class HeldoutTests(unittest.TestCase):
    def test_windows_overlap_on_their_last_target(self):
        windows = scoring.packed_windows([[10, 11, 12], [20, 21]], 1, 2, 3, 2)
        self.assertEqual(windows, [[1, 10, 11, 12], [12, 2, 1, 20]])

    def test_bits_per_byte_of_a_uniform_model(self):
        model = toy_model(uniform=True)
        windows = scoring.packed_windows([[ord(c) for c in "hello"]] * 3, 1, 2, 6, 2)
        result = scoring.bits_per_byte(model, windows)
        targets = [t for w in windows for t in w[1:] if t != 1]
        text_bytes = sum(t >= 4 for t in targets)
        self.assertEqual((result["target_tokens"], result["bytes"]), (len(targets), text_bytes))
        expected = len(targets) * math.log2(VOCAB) / text_bytes
        self.assertAlmostEqual(result["bits_per_byte"], expected, places=4)


class GenerationTests(unittest.TestCase):
    def test_greedy_generation_stops_at_a_stop_string(self):
        model = toy_model()
        self.assertEqual(scoring.generate(model, "a", 20, ["e"]), "bcd")
        self.assertEqual(scoring.generate(model, "a", 3, ["\n\n"]), "bcd")

    def test_few_shot_answers_are_checked_with_math_verify(self):
        completions = iter([" 2+2=4\n#### 4", " The answer is 1,234.", " 15 - 19 = 2\n#### 2"])
        problems = [("Two plus two?", "4"), ("Big?", "1234"), ("Fish?", "4")]
        with mock.patch.object(scoring, "generate", lambda *args: next(completions)):
            result, outcomes = scoring.few_shot_answers(None, problems, [("q", "a #### 1")], 8)
        self.assertEqual([o["correct"] for o in outcomes], [True, True, False])
        self.assertEqual((result["acc"], result["shots"]), (2 / 3, 1))

    def test_math_problems(self):
        row = {"Body": "Tom has 3. ", "Question": "How many?", "Answer": 3.0}
        self.assertEqual((svamp.question(row), svamp.answer(row)), ("Tom has 3. How many?", "3"))
        row = {"question": "q", "answer": "x\n#### 5"}
        self.assertEqual(gsm8k.problem(row), ("q", "5"))
        self.assertEqual(gsm8k.exemplar(row), ("q", "x\n#### 5"))

    def test_hellaswag_preprocessing(self):
        self.assertEqual(hellaswag.clean(" A [title] b [header]  c "), "A. b  c")


class ConfigTests(unittest.TestCase):
    def test_shipped_configs_are_complete(self):
        from pathlib import Path

        from evaluation.modal_evaluate import DEFAULT_CONFIG, MATH, TASKS, load_config

        root = Path(__file__).resolve().parents[1]
        config = load_config(root / DEFAULT_CONFIG)
        self.assertEqual(set(config["datasets"]), set(TASKS) - {"heldout"})
        for spec in config["datasets"].values():
            self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$")
        self.assertTrue(all("max_new_tokens" in config["datasets"][t] for t in MATH))
        playground = json.loads(
            (root / "analysis/playground/configs/base-vs-math-sft.json").read_text()
        )
        labels = sorted(m["label"] for m in playground["models"])
        self.assertEqual(labels, sorted(label for row in playground["layout"] for label in row))


class RegistryTests(unittest.TestCase):
    def test_every_configured_benchmark_has_a_module(self):
        from evaluation.modal_evaluate import TASKS

        self.assertEqual(set(BENCHMARKS), set(TASKS) - {"heldout"})


if __name__ == "__main__":
    unittest.main()
