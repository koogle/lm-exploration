import unittest

import torch
from model_10m.model import GPT, ModelConfig

from trainer.rollout import sample

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
        self.model = GPT(ModelConfig(vocab_size=64, block_size=48, n_layer=2, n_head=2, n_embd=32))
        self.model.eval()
        self.prompts = [[1, 5, 9, 11], [1, 7], [1, 3, 3, 3, 4, 8, 20]]

    def test_greedy_matches_unbatched_uncached_decoding(self):
        completions, finished = sample(self.model, self.prompts, 20, EOS, temperature=0)
        for prompt, completion, done in zip(self.prompts, completions, finished, strict=True):
            self.assertEqual((completion, done), naive_greedy(self.model, prompt, 20))

    def test_sampling_is_seeded_and_stops_at_eos(self):
        draws = [
            sample(self.model, self.prompts, 30, EOS, generator=torch.Generator().manual_seed(3))
            for _ in range(2)
        ]
        self.assertEqual(draws[0], draws[1])
        for completion, done in zip(*draws[0], strict=True):
            self.assertNotIn(EOS, completion)
            self.assertTrue(done or len(completion) == 30)

    def test_rejects_overlong_requests(self):
        with self.assertRaises(ValueError):
            sample(self.model, self.prompts, 42, EOS)


if __name__ == "__main__":
    unittest.main()
