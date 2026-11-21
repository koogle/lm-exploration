"""HumanEval-Infilling program assembly in `fim_tasks`."""

import unittest

from model_hybrid import fim_tasks

DOC = {
    "prompt": "def f(x):\n",
    "suffix": "    return y\n",
}


class SingleLinePredictionTests(unittest.TestCase):
    def test_restores_the_newline_the_stop_removed(self):
        [[program]] = fim_tasks._single_line_predictions([["    y = x + 1"]], [DOC])
        self.assertEqual(program, "def f(x):\n    y = x + 1\n    return y\n")
        namespace = {}
        exec(program, namespace)
        self.assertEqual(namespace["f"](1), 2)


if __name__ == "__main__":
    unittest.main()
