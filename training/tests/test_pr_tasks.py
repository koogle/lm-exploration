"""PR-to-task helpers: commit shapes, sampling, the teacher's input and the leak checks."""

import json
import unittest

from training_data import pr_tasks

PATCH = (
    "diff --git a/pkg/limits.py b/pkg/limits.py\n"
    "--- a/pkg/limits.py\n+++ b/pkg/limits.py\n@@ -1,2 +1,4 @@\n"
    " def clamp(x):\n-    return x\n"
    "+    return _bounded_value(x, low=0, high=10)\n"
    "+def _bounded_value(x, low, high): return max(low, min(high, x))\n"
    "diff --git a/tests/test_limits.py b/tests/test_limits.py\n"
    "--- a/tests/test_limits.py\n+++ b/tests/test_limits.py\n@@ -1 +1,2 @@\n"
    " from pkg.limits import clamp\n+assert clamp(12) == 10\n"
)
TASK = {
    "repository": "owner/repo",
    "message": "Clamp values\n\nCo-Authored-By: Someone <a@b.c>",
    "patch": PATCH,
    "source_paths": ["pkg/limits.py"],
    "test_paths": ["tests/test_limits.py"],
}


def row(files, parents=("p",), message="Fix the clamp bound"):
    return {"parents": list(parents), "message": message, "files": files}


def change(path, before, after):
    return {"path": path, "before": before, "after": after}


class PRTaskTests(unittest.TestCase):
    def test_classifies_tests_and_languages(self):
        self.assertTrue(pr_tasks.is_test("tests/test_x.py"))
        self.assertTrue(pr_tasks.is_test("src/a.spec.ts"))
        self.assertTrue(pr_tasks.is_test("pkg/a_test.go"))
        self.assertFalse(pr_tasks.is_test("src/latest.py"))
        self.assertEqual(pr_tasks.language("a.tsx"), "javascript")
        self.assertIsNone(pr_tasks.language("README.md"))

    def test_describe_keeps_source_plus_test_commits(self):
        source = change("pkg/a.py", "x = 1\n", "x = 2\ny = 3\nz = 4\n")
        test = change("tests/test_a.py", "", "def test():\n    assert x == 2\n    assert y\n")
        shape = pr_tasks.describe(row([source, test]))
        self.assertEqual(
            (shape["language"], shape["source_files"], shape["test_files"]), ("python", 1, 1)
        )
        self.assertEqual(shape["source_lines"], 4)
        self.assertEqual(pr_tasks.describe(row([source]))["reject"], "test_files")
        self.assertEqual(pr_tasks.describe(row([source, test], ("p", "q")))["reject"], "merge")
        go = change("main.go", "", "package main\n")
        self.assertEqual(pr_tasks.describe(row([source, test, go]))["reject"], "mixed_languages")

    def test_sample_caps_each_repository(self):
        items = [{"repository": r, "sha": str(i)} for r in "ab" for i in range(10)]
        chosen = pr_tasks.sample(items, 100, per_repository=3)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(chosen, pr_tasks.sample(items, 100, per_repository=3))

    def test_teacher_sees_tests_but_not_source_changes(self):
        user = pr_tasks.issue_messages(TASK)[1]["content"]
        self.assertIn("assert clamp(12) == 10", user)
        self.assertNotIn("_bounded_value", user)
        self.assertNotIn("Co-Authored-By", user)

    def test_leaks_flag_internal_names_quotes_and_test_mentions(self):
        clean = {"title": "clamp ignores its upper bound", "body": "`clamp(12)` should return 10."}
        self.assertFalse(
            any(pr_tasks.leaks(clean, PATCH, TASK["source_paths"], TASK["test_paths"]).values())
        )
        leaky = {"title": "Add `_bounded_value`", "body": "Tests should check clamp."}
        found = pr_tasks.leaks(leaky, PATCH, TASK["source_paths"], TASK["test_paths"])
        self.assertEqual(found["names"], ["_bounded_value"])
        self.assertEqual(found["meta"], ["tests"])

    def test_best_issue_prefers_the_first_clean_draft(self):
        def draft(title, body, contained=True):
            return json.dumps({"title": title, "body": body, "self_contained": contained})

        texts = [
            "not json",
            draft("Use _bounded_value", "x"),
            draft("clamp bound", "`clamp(12)` should be 10", contained=False),
            draft("clamp upper bound", "`clamp(12)` should return 10"),
        ]
        status, issue, _ = pr_tasks.best_issue(texts, TASK)
        self.assertEqual((status, issue["title"]), ("ok", "clamp upper bound"))


if __name__ == "__main__":
    unittest.main()
