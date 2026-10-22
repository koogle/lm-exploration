import unittest

from launch_campaign import candidates


class CampaignGateTests(unittest.TestCase):
    def trial(self, tokens=100, seconds=10, complete=True, accepted=32):
        return {
            "batch_size": 8,
            "records": 32,
            "complete": complete,
            "accepted_records": accepted,
            "accepted_tokens": tokens,
            "durable_seconds": seconds,
        }

    def test_partial_or_single_round_cannot_launch(self):
        self.assertEqual(candidates({"trials": {"0": self.trial()}}), [])
        self.assertEqual(
            candidates(
                {"trials": {"0": self.trial(), "1": self.trial(complete=False)}}
            ),
            [],
        )

    def test_rejection_heavy_candidate_is_excluded(self):
        self.assertEqual(
            candidates(
                {"trials": {"0": self.trial(accepted=20), "1": self.trial(accepted=20)}}
            ),
            [],
        )

    def test_throughput_uses_pooled_tokens_and_durable_time(self):
        result = candidates(
            {
                "trials": {
                    "0": self.trial(tokens=100, seconds=1),
                    "1": self.trial(tokens=100, seconds=9),
                }
            }
        )
        self.assertEqual(result[0]["accepted_tokens_per_second"], 20)
