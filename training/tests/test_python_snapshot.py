import json
import tempfile
import unittest
from pathlib import Path

from training_data.python_experiment import Corpus, split_for
from training_data.python_snapshot import freeze, merge_documentation


class SnapshotTests(unittest.TestCase):
    def test_merge_excludes_repository_references_and_refuses_insufficient_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            main = Corpus(root / "main")
            main.db.execute("INSERT INTO excluded_repositories VALUES('org/heldout')")
            main.report("test")
            main.db.close()
            shard = Corpus(root / "docs" / "shard")
            text = "Python tutorial example and explanation with a reference to https://github.com/org/heldout\n"
            shard.add(text, "python_documentation", "stackoverflow:12", {}, split="train")
            parent = next(
                f"stackoverflow:{i}"
                for i in range(100)
                if split_for(f"stackoverflow:{i}") == "train"
            )
            accepted = "Python iterators produce one value at a time and retain generator state between calls.\n"
            shard.add(accepted, "python_documentation", parent, {}, split="train")
            shard.report("test")
            shard.db.close()
            (root / "docs" / "shard" / "complete.json").write_text(json.dumps({"complete": True}))
            first = merge_documentation(root / "main", root / "docs")
            second = merge_documentation(root / "main", root / "docs")
            self.assertEqual(first["categories"], second["categories"])
            self.assertEqual(second["rejections"]["documentation_references_heldout_repository"], 1)
            self.assertGreater(
                second["categories"]["python_documentation"]["unique_train_targets"], 0
            )
            with self.assertRaisesRegex(ValueError, "Insufficient measured corpus"):
                freeze(root / "main", root / "snapshot")
            self.assertFalse((root / "snapshot").exists())


if __name__ == "__main__":
    unittest.main()
