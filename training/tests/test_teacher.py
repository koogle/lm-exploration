import unittest

from training_data.teacher import choose, extract, messages

ROW = {
    "question": "Return the larger of two numbers.",
    "declaration": "def larger(a, b):",
    "tests": "from solution import *\n\ndef test_asserts():\n    assert larger(1, 2) == 2\n",
    "source": "oci x",
}


class TeacherTests(unittest.TestCase):
    def test_messages_name_the_declaration(self):
        prompt = messages(ROW)[-1]["content"]
        self.assertIn("`def larger(a, b):`", prompt)
        self.assertIn("Python function", prompt)

    def test_extract_keeps_definitions_of_the_declared_name(self):
        text = "Sure:\n```python\nimport os\n\ndef larger(a, b):\n    return max(a, b)\n\nprint(larger(1, 2))\n```"
        self.assertEqual(extract(text, ROW), "import os\n\ndef larger(a, b):\n    return max(a, b)")
        self.assertIsNone(extract("```python\ndef smaller(a, b):\n    pass\n```", ROW))
        self.assertIsNone(extract("no code", ROW))
        self.assertIsNone(extract("```python\ndef larger(:\n```", ROW))

    def test_choose_takes_the_shortest_passing_answer(self):
        answers = [
            "def larger(a, b):\n    return a if a > b else b",
            None,
            "def larger(a, b):\n    return max(a, b)",
        ]
        chosen = choose(ROW, answers, [{"passed": True}, {"passed": True}])
        self.assertEqual(chosen["solution"], answers[2])
        self.assertIn("solution by Qwen/", chosen["source"])
        self.assertIsNone(choose(ROW, answers, [{"passed": False}, {"passed": False}]))


if __name__ == "__main__":
    unittest.main()
