import json
import unittest
from pathlib import Path


class ConfigTests(unittest.TestCase):
    def test_layout_shows_every_model(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "analysis/playground/configs/base-vs-math-sft.json").read_text()
        )
        labels = sorted(m["label"] for m in config["models"])
        self.assertEqual(labels, sorted(label for row in config["layout"] for label in row))


if __name__ == "__main__":
    unittest.main()
