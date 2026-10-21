import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training_data.private_patch_sft import blob_matches, build, commit_split, excerpts, instruction
from training_data.subword import digest


def oid(text):
    raw = text.encode()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


class PrivatePatchTests(unittest.TestCase):
    def test_exact_regions_and_blob_integrity(self):
        old = "def example():\n    return 123\n" + "# unchanged\n" * 20 + "print('old')\n"
        new = old.replace("123", "456").replace("'old'", "'new'")
        groups = list(excerpts(old, new))
        self.assertEqual(len(groups), 2)
        for a, b, c, d, left, right, _ in groups:
            self.assertEqual(left, "".join(old.splitlines(keepends=True)[a:b]))
            self.assertEqual(right, "".join(new.splitlines(keepends=True)[c:d]))
        self.assertTrue(blob_matches(old, oid(old)))
        self.assertFalse(blob_matches(new, oid(old)))

    def test_instructions_and_commit_split(self):
        self.assertIsNone(instruction("Merge branch main"))
        self.assertIsNone(instruction("Revert broken change"))
        self.assertEqual(
            instruction("Fix incorrect return value\n\nSigned-off-by: X <x@example.com>"),
            "Fix incorrect return value",
        )
        self.assertEqual(commit_split("OWNER/REPO", "a" * 40), commit_split("owner/repo", "a" * 40))

    def test_full_preparation_and_heldout_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for i in range(80):
                before = f"def example_{i}():\n    return {i} - 1\n"
                after = f"def example_{i}():\n    return {i} + 1\n"
                rows.append(
                    {
                        "repository": "unit/repo",
                        "sha": f"{i:040x}",
                        "message": f"Fix incorrect return value for example {i}",
                        "license_before": {"text": "MIT license fixture"},
                        "license_after": {"text": "MIT license fixture"},
                        "files": [
                            {
                                "path": "example.py",
                                "before": before,
                                "after": after,
                                "before_blob": oid(before),
                                "after_blob": oid(after),
                            }
                        ],
                    }
                )
            shard = root / "source.jsonl.gz"
            with gzip.open(shard, "wt") as handle:
                handle.write("".join(json.dumps(row) + "\n" for row in rows))
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "local_path": str(shard),
                        "sha256": digest(shard),
                        "dataset": "fixture/mit-claude",
                        "revision": "abc",
                        "file": "source.jsonl.gz",
                    }
                )
                + "\n"
            )
            report = build(manifest, root / "output")
            self.assertGreater(report["total"]["train"], 50)
            self.assertGreater(report["total"]["validation"], 0)
            selected = [
                json.loads(s) for s in (root / "output/examples.jsonl").read_text().splitlines()
            ]
            by_split = {
                split: {r["commit"] for r in selected if r["split"] == split}
                for split in ("train", "validation")
            }
            self.assertFalse(by_split["train"] & by_split["validation"])
            self.assertTrue(all(r["scope"] == "excerpt" for r in selected))

            capped = build(manifest, root / "capped", repository_cap=5)
            self.assertEqual(capped["total"]["train"], 5)
            reserved_row = next(r for r in selected if r["split"] == "train")
            reserved_row["split"] = "validation"
            reserved = root / "previous-validation.jsonl"
            reserved.write_text(json.dumps(reserved_row) + "\n")
            full = build(
                manifest,
                root / "full",
                reserved_paths=[reserved],
                train_cap=None,
                language_cap=None,
                repository_cap=None,
            )
            self.assertEqual(full["total"]["train"], report["total"]["train"] - 1)
            self.assertEqual(full["rejections"]["heldout_overlap"], 1)
            self.assertIsNone(full["limits"]["train_per_repository"])
