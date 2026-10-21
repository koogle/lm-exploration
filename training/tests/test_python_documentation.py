"""Explanatory documentation must retain attributable, complete Python threads."""

import unittest

from training_data.python_documentation import select_document


class DocumentationTests(unittest.TestCase):
    def record(self):
        return {
            "id": "123",
            "text": "How to write a Python iterator?\n"
            + (
                "An iterator provides values one at a time so the consumer can process each value without storing the entire sequence in memory.\n"
                "The generator function below keeps its local state between calls and returns the next value when the caller requests it.\n"
                "Use a for loop to consume the values, and allow the generator to stop naturally after reaching the requested upper bound.\n"
                "Unlike building a list in advance this approach evaluates each item only when it is needed by the surrounding program.\n"
                "The following example illustrates how the stop condition and increment work together to avoid producing values beyond the desired range.\n"
                "This behavior is particularly useful when reading large inputs that cannot be held entirely in memory at the same time.\n"
                "def numbers(limit):\n    for value in range(limit):\n        yield value\n"
            ),
            "metadata": {
                "url": "https://stackoverflow.com/questions/123",
                "authors": ["example"],
                "license": "https://creativecommons.org/licenses/by-sa/4.0/",
            },
        }

    def test_python2_specific_howto_is_excluded_but_comparison_is_allowed(self):
        row = self.record()
        row["text"] = row["text"].replace(
            "How to write a Python iterator?", "How to write an iterator in Python 2.7?"
        )
        self.assertEqual(select_document(row)[1], "python2_specific_documentation")
        row["text"] = row["text"].replace("Python 2.7?", "Python 2.7 versus Python 3?")
        self.assertIsNone(select_document(row)[1])

    def test_complete_text_and_attribution(self):
        r = self.record()
        selected, reason = select_document(r)
        self.assertIsNone(reason)
        self.assertEqual(selected[0], r["text"])
        self.assertEqual(selected[1], "stackoverflow:123")
        self.assertEqual(selected[2]["attribution"], r["metadata"])

    def test_license_and_mixed_code_are_rejected(self):
        r = self.record()
        r["metadata"]["license"] = "unknown"
        self.assertEqual(select_document(r)[1], "unresolved_license")
        r = self.record()
        r["text"] += "\n#include <stdio.h>\n"
        self.assertEqual(select_document(r)[1], "mixed_language_code")
        self.assertEqual(select_document(None)[1], "malformed_record")


if __name__ == "__main__":
    unittest.main()
