"""RL pieces: row layout, the weighted log-likelihood, prompt order and bf16 RoPE tables."""

import unittest

import torch
from model_hybrid import model, rewards, rl, rl_data
from torch.nn import functional as F

from tests.test_hybrid_backbone import tiny_config

EOS = tiny_config().eos_id


class RLTests(unittest.TestCase):
    def test_rows_target_only_completion_tokens_with_their_advantage(self):
        tokens, targets, weights = rl.batch(
            [[5, 6, 7], [8]], [[9, 10], [11, 12, 13]], [0.5, -1.0], EOS, 4
        )
        self.assertEqual(tokens.shape, (2, 4))
        self.assertEqual(tokens[0].tolist(), [5, 6, 7, 9])
        self.assertEqual(targets[0].tolist(), [-100, -100, 9, 10])
        self.assertEqual(weights[0].tolist(), [0.0, 0.0, 0.5, 0.5])
        self.assertEqual(tokens[1].tolist(), [8, 11, 12, EOS])
        self.assertEqual(targets[1].tolist(), [11, 12, 13, -100])
        self.assertEqual(weights[1].tolist(), [-1.0, -1.0, -1.0, 0.0])

    def test_policy_objective_matches_explicit_weighted_logprob(self):
        torch.manual_seed(0)
        lm = model.HybridLM(tiny_config(mtp=False)).double()
        policy = rl.Policy(lm, chunk_size=3)
        tokens, targets, weights = rl.batch(
            [[5, 6], [7, 8, 9]], [[10, 11, 12], [13]], [0.7, -0.2], EOS, 8
        )
        weighted, logprob = policy(tokens, targets, weights.double())
        logp = F.log_softmax(lm(tokens), -1)
        valid = targets != -100
        explicit = logp[valid].gather(-1, targets[valid][:, None])[:, 0]
        expected = (explicit * weights.double()[valid]).sum()
        torch.testing.assert_close(weighted.double(), expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(logprob.double(), explicit.sum(), rtol=1e-5, atol=1e-5)

    def test_prompt_stream_resumes_and_reshuffles_each_pass(self):
        prompts = [{"id": i} for i in range(5)]
        stream = rl_data.PromptStream(prompts, seed=3)
        first = [p["id"] for p in stream.take(7)]
        state = stream.state_dict()
        rest = [p["id"] for p in stream.take(3)]
        resumed = rl_data.PromptStream(prompts, seed=3)
        resumed.load_state_dict(state)
        self.assertEqual([p["id"] for p in resumed.take(3)], rest)
        self.assertEqual(sorted(first[:5]), list(range(5)))

    def test_rope_tables_stay_fp32_when_the_model_is_cast(self):
        lm = model.HybridLM(tiny_config())
        before = lm.rope_cos.clone()
        lm = lm.to(torch.bfloat16)
        self.assertEqual(lm.rope_cos.dtype, torch.float32)
        torch.testing.assert_close(lm.rope_cos, before)

    def test_boxed_answers_with_nested_braces(self):
        self.assertEqual(
            rewards.last_boxed("a \\boxed{1} then \\boxed{\\frac{1}{2}}"), "\\frac{1}{2}"
        )
        self.assertIsNone(rewards.last_boxed("no box"))
        self.assertEqual(rewards.extract_code("x\n```python\nA\n```\n```py\nB\n```"), "B\n")
        self.assertEqual(rl_data.gsm8k_answer("so 1,200\n#### 1,200"), "1200")


if __name__ == "__main__":
    unittest.main()
