import unittest

import torch
from model_10m.model import GPT, ModelConfig

from trainer.rollout import choose, sample

EOS = 2


def naive_greedy(model, prompt, max_new_tokens):
    tokens, generated = list(prompt), []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            token = int(model(torch.tensor([tokens]))[0, -1].argmax())
        if token == EOS:
            return generated, True
        generated.append(token)
        tokens.append(token)
    return generated, False


class RolloutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GPT(
            ModelConfig(vocab_size=64, eos_id=EOS, block_size=48, n_layer=2, n_head=2, n_embd=32)
        )
        self.model.eval()
        self.prompts = [[1, 5, 9, 11], [1, 7], [1, 3, 3, 3, 4, 8, 20]]

    def test_greedy_matches_unbatched_uncached_decoding(self):
        samples = sample(self.model, self.prompts, 20, EOS, temperature=0)
        for prompt, completion, done in zip(
            self.prompts, samples.completions, samples.finished, strict=True
        ):
            self.assertEqual((completion, done), naive_greedy(self.model, prompt, 20))

    def test_sampling_is_seeded_and_stops_at_eos(self):
        draws = [
            sample(self.model, self.prompts, 30, EOS, generator=torch.Generator().manual_seed(3))
            for _ in range(2)
        ]
        self.assertEqual(draws[0].completions, draws[1].completions)
        for completion, done in zip(draws[0].completions, draws[0].finished, strict=True):
            self.assertNotIn(EOS, completion)
            self.assertTrue(done or len(completion) == 30)

    def test_stop_ends_a_row_and_keeps_its_tokens(self):
        samples = sample(
            self.model, self.prompts, 10, -1, temperature=0, stop=lambda g: len(g) == 2
        )
        self.assertEqual([len(c) for c in samples.completions], [2, 2, 2])
        self.assertEqual(samples.finished, [False, False, False])

    def test_rejects_overlong_requests(self):
        with self.assertRaises(ValueError):
            sample(self.model, self.prompts, 42, EOS)


class ChooseTests(unittest.TestCase):
    def draws(self, logits, **settings):
        options = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "penalty": 1.0, **settings}
        seen = torch.zeros(1, logits.numel(), dtype=torch.bool)
        return {
            int(choose(logits[None], seen, generator=torch.Generator().manual_seed(s), **options))
            for s in range(200)
        }

    def test_nucleus_keeps_the_crossing_token(self):
        logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
        self.assertEqual(self.draws(logits, top_p=0.6), {0, 1})

    def test_top_k_keeps_the_largest(self):
        logits = torch.log(torch.tensor([0.1, 0.4, 0.2, 0.3]))
        self.assertEqual(self.draws(logits, top_k=2), {1, 3})

    def test_repetition_penalty_lowers_seen_tokens(self):
        logits = torch.tensor([[2.0, 1.9, -1.0]])
        seen = torch.tensor([[True, False, True]])
        greedy = choose(logits, seen, 0, 0, 1.0, 1.5, None)
        self.assertEqual(int(greedy), 1)


if __name__ == "__main__":
    unittest.main()
