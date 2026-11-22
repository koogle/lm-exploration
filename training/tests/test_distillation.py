import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from model_hybrid.model import chunked_distillation
from training_data import distillation

from tokenizer import glm
from trainer.rl.algorithms import make
from trainer.rl.batch import Tokens
from trainer.rl.off_policy.sources import Scored
from trainer.rl.settings import Settings

USER, ASSISTANT, THINK = (
    glm.SPECIAL_TOKENS[name] for name in ("<|user|>", "<|assistant|>", "<think>")
)
CLOSE = glm.SPECIAL_TOKENS["</think>"]


def write_set(folder, k=3):
    prompts = [[USER, 5, ASSISTANT, THINK], [USER, 6, 7, ASSISTANT, THINK]]
    tokens, answers = distillation.pack(prompts, [[11, 12], [13]], [glm.EOS_ID, None])
    rows = int(answers[:, 2].sum())
    rng = np.random.default_rng(0)
    logprobs = np.log(rng.dirichlet(np.ones(k + 1), rows)[:, :k])
    arrays = {
        "tokens": tokens,
        "answers": answers,
        "teacher_ids": rng.integers(0, 50, (rows, k)).astype(np.int32),
        "teacher_logprobs": logprobs.astype(np.float16),
    }
    for name, array in arrays.items():
        np.save(folder / f"{name}.npy", array)
    return distillation.write_report(folder, {"student": "test"})


class DataTests(unittest.TestCase):
    def test_thinking_prompt_ends_in_the_opened_reasoning_span(self):
        ids = np.array([USER, 5, ASSISTANT, THINK, 9, 9, glm.EOS_ID])
        mask = np.array([0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(distillation.thinking_prompt(ids, mask), [USER, 5, ASSISTANT, THINK])
        # An answer-only conversation gets the reasoning span instead of the closed one.
        ids[3] = CLOSE
        self.assertEqual(distillation.thinking_prompt(ids, mask), [USER, 5, ASSISTANT, THINK])
        self.assertIsNone(distillation.thinking_prompt(ids, np.zeros(7, dtype=np.uint8)))
        self.assertIsNone(distillation.thinking_prompt(ids, np.array([0, 1, 1, 1, 1, 1, 1])))

    def test_a_set_round_trips_answers_and_teacher_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            report = write_set(folder)
            self.assertEqual((report["answers"], report["answer_tokens"]), (2, 4))
            answers = distillation.Answers(folder, verify=True)
            prompt, completion, stop, (ids, _) = answers.answer(0)
            self.assertEqual(
                (prompt, completion, stop), ([USER, 5, ASSISTANT, THINK], [11, 12], glm.EOS_ID)
            )
            self.assertEqual(ids.shape, (3, 3))
            prompt, completion, stop, (ids, _) = answers.answer(1)
            self.assertEqual((completion, stop, len(ids)), ([13], None, 1))
            np.save(folder / "answers.npy", np.load(folder / "answers.npy") * 1 + 0)
            (folder / "tokens.npy").write_bytes((folder / "tokens.npy").read_bytes()[:-4] + b"abcd")
            with self.assertRaises(ValueError):
                distillation.Answers(folder, verify=True)

    def test_merging_shards_drops_empty_answers(self):
        first = distillation.pack([[1, 2]], [[3]], [None])
        second = distillation.pack([[4], [5]], [[], [6, 7]], [None, 9])
        tokens, answers = distillation.merge_answers([first, second])
        self.assertEqual(tokens.tolist(), [1, 2, 3, 5, 6, 7, 9])
        self.assertEqual(answers.tolist(), [[0, 2, 1, 0], [3, 1, 3, 1]])


class LossTests(unittest.TestCase):
    def test_the_listed_kl_is_the_full_kl_when_every_token_is_listed(self):
        torch.manual_seed(0)
        hidden = torch.randn(2, 5, 8, requires_grad=True)
        weight = torch.randn(12, 8)
        targets = torch.randint(0, 12, (2, 5))
        targets[1, 3:] = -100
        teacher = torch.randn(2, 5, 12).log_softmax(-1)
        ids = torch.arange(12).expand(2, 5, 12)
        logp, kl = chunked_distillation(hidden, weight, targets, ids, teacher, chunk_size=3)
        student = (hidden @ weight.T).log_softmax(-1)
        expected = (teacher.exp() * (teacher - student)).sum(-1) * (targets != -100)
        torch.testing.assert_close(kl, expected, atol=1e-5, rtol=1e-4)
        chosen = student.gather(-1, targets.clamp_min(0)[..., None])[..., 0] * (targets != -100)
        torch.testing.assert_close(logp, chosen, atol=1e-5, rtol=1e-4)
        (grad,) = torch.autograd.grad(kl.sum(), hidden)
        self.assertTrue(torch.isfinite(grad).all())

    def test_unlisted_tokens_count_as_one_outcome(self):
        hidden = torch.eye(4)[None]
        weight = torch.eye(4) * 3
        targets = torch.zeros(1, 4, dtype=torch.long)
        student = (hidden @ weight.T).log_softmax(-1)
        teacher = torch.tensor([0.5, 0.3, 0.1, 0.1]).log().expand(1, 4, 4)
        ids = torch.arange(2).expand(1, 4, 2)
        _, kl = chunked_distillation(hidden, weight, targets, ids, teacher[..., :2], chunk_size=8)
        listed = student[..., :2]
        rest = (1 - listed.exp().sum(-1)).log()
        expected = (teacher[..., :2].exp() * (teacher[..., :2] - listed)).sum(-1) + 0.2 * (
            torch.tensor(0.2).log() - rest
        )
        torch.testing.assert_close(kl, expected, atol=1e-5, rtol=1e-4)

    def test_distill_minimizes_the_teacher_kl_on_answer_tokens(self):
        algorithm = make({"name": "off_policy/distill"})
        kl = torch.tensor([[1.0, 2.0, 4.0]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        zeros = torch.zeros(1, 3)
        micro = Tokens(
            None,
            zeros.long(),
            mask,
            zeros,
            zeros,
            torch.zeros(1),
            torch.tensor([0.5]),
            teacher_kl=kl,
        )
        loss, stats = algorithm.loss(micro)
        self.assertAlmostEqual(loss.item(), 1.5)
        self.assertAlmostEqual(stats["teacher_kl"], 3.0)

    def test_a_temperature_below_one_sharpens_the_listed_tokens(self):
        logprobs = torch.tensor([[0.6, 0.3, 0.05]]).log()
        self.assertIs(make({"name": "off_policy/distill"}).targets(logprobs), logprobs)
        sharp = make({"name": "off_policy/distill", "temperature": 0.5}).targets(logprobs).exp()
        expected = torch.tensor([[0.36, 0.09, 0.0025]]) / 0.4525
        torch.testing.assert_close(sharp, expected, atol=1e-6, rtol=1e-5)
        with self.assertRaises(ValueError):
            make({"name": "off_policy/distill", "temperature": 1.5})


class SourceTests(unittest.TestCase):
    def test_scored_draws_answers_with_their_teacher_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_set(folder)
            answers = distillation.Answers(folder)
            source = Scored(answers, [0, 1], 2, random.Random(0))
            batch = source.batch(0)
            sequences = batch.sequences()
            for row, sequence, teacher in zip(batch.rows, sequences, batch.teacher, strict=True):
                targets = [t for t in sequence["y"] if t != -100]
                self.assertEqual(len(targets), len(teacher[0]))
                self.assertEqual(len(sequence["x"]), len(row["prompt_ids"]) + len(targets) - 1)
            self.assertEqual(sorted(batch.finished), [False, True])
            state = json.loads(json.dumps(source.state()))
            source.load_state(state)
            self.assertEqual(source.statistics(1), {"trajectory_passes": 1})

    def test_settings_pair_distill_with_the_scored_source(self):
        scored = {"source": "scored", "data": "answers-01", "answers_per_step": 8}
        settings = Settings.from_dict(
            {"algorithm": {"name": "off_policy/distill"}, "trajectories": scored}
        )
        self.assertTrue(settings.algorithm.needs_teacher)
        for wrong in (
            {"algorithm": {"name": "off_policy/distill"}},
            {"algorithm": {"name": "off_policy/awr"}, "trajectories": scored},
            {"algorithm": {"name": "off_policy/distill"}, "trajectories": {"source": "scored"}},
        ):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                Settings.from_dict(wrong)


if __name__ == "__main__":
    unittest.main()
