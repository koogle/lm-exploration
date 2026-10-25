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
        samples = sample(self.model, self.prompts, 20, EOS, temperature=0)
        self.assertIsNone(samples.logprobs)
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

    def test_recorded_logprobs_match_the_model(self):
        samples = sample(
            self.model, self.prompts, 12, EOS, generator=torch.Generator().manual_seed(1), top_k=4
        )
        for row, prompt in enumerate(self.prompts):
            answer = samples.completions[row] + ([EOS] if samples.finished[row] else [])
            with torch.no_grad():
                logits = self.model(torch.tensor([prompt + answer]))[0]
            expected = torch.log_softmax(logits[len(prompt) - 1 : -1], -1)
            steps = len(answer)
            picked = expected.gather(1, torch.tensor(answer)[:, None])[:, 0]
            torch.testing.assert_close(samples.logprobs[row, :steps], picked, atol=1e-4, rtol=0)
            top, ids = expected.topk(4, -1)
            torch.testing.assert_close(samples.top_logprobs[row, :steps], top, atol=1e-4, rtol=0)
            self.assertTrue(torch.equal(samples.top_ids[row, :steps], ids))

    def test_rejects_overlong_requests(self):
        with self.assertRaises(ValueError):
            sample(self.model, self.prompts, 42, EOS)


if __name__ == "__main__":
    unittest.main()
