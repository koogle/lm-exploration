import unittest

from training_data.commitpack_sft import Families, adapt, fingerprint, repositories, split_for


class CommitPackTests(unittest.TestCase):
    def test_transitive_repository_aliases(self):
        families = Families()
        families.add(["b/repo", "c/repo"])
        families.add(["a/repo", "b/repo"])
        self.assertEqual(families.find("c/repo"), "a/repo")
        self.assertEqual(split_for(families.find("c/repo")), split_for("a/repo"))
        self.assertEqual(repositories("A/Repo,a/repo"), ["a/repo"])
        self.assertEqual(repositories("invalid"), [])

    def test_provenance_and_filters(self):
        raw = {
            "repos": "owner/repo",
            "license": "mit",
            "subject": "Fix handling of negative input",
            "old_contents": "def solve(a, b): return a - b",
            "new_contents": "def solve(a, b): return abs(a - b)",
            "commit": "a" * 40,
            "old_file": "solve.py",
            "new_file": "solve.py",
        }
        row, reason = adapt(raw, "python", Families())
        self.assertIsNone(reason)
        self.assertEqual(row["commit"], raw["commit"])
        self.assertIn("tests not rerun", row["verification"])
        self.assertEqual(adapt(dict(raw, license="unknown"), "python", Families())[1], "license")
        self.assertEqual(
            adapt(dict(raw, new_contents=raw["old_contents"]), "python", Families())[1],
            "size_or_unchanged",
        )
        self.assertEqual(fingerprint("a  + b\n"), fingerprint("a + b"))
