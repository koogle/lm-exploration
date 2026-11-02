"""`chat_sft.pass_windows` counts the windows `ChatMixture.next_batch` packs."""

import tempfile
import unittest

import numpy as np
from training_data import chat_sft


class PassWindowsTests(unittest.TestCase):
    def test_matches_packing(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as folder:
            for name in ("a", "b"):
                lengths = rng.integers(2, 30, 40)
                conversations = [
                    (list(rng.integers(0, 100, n)), [0] + [1] * (n - 1)) for n in lengths
                ]
                chat_sft._write(chat_sft.Path(folder), name, conversations)
            recipe = {"subsets": [{"name": "a"}, {"name": "b", "repeat": 1.5}], "context": 32}
            expected = chat_sft.pass_windows(chat_sft.ChatMixture(recipe, folder))
            mixture, packed = chat_sft.ChatMixture(recipe, folder), 0
            while True:
                try:
                    packed += len(mixture.next_batch(1)[0])
                except StopIteration:
                    break
            self.assertEqual(expected, packed)


if __name__ == "__main__":
    unittest.main()
