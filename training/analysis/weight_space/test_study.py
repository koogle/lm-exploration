"""Sparse edits must affect only the intended direction and restore exact bytes."""

import unittest
from types import SimpleNamespace

import torch
from study import apply, command_metrics, restore


class ProjectionTest(unittest.TestCase):
    def test_projection_scope_and_restore(self):
        torch.manual_seed(7)
        layer = torch.nn.Linear(8, 6, bias=False, dtype=torch.float32)
        original = layer.weight.detach().clone()
        model = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=layer))])
        )
        direction = torch.tensor([1.0, 2.0, 0.0, -1.0, 3.0, 1.0])
        direction /= direction.norm()
        proposal = {"layer": 0, "columns": torch.tensor([1, 4]), "direction": direction}
        undo = apply(model, [proposal])
        torch.testing.assert_close(
            direction @ layer.weight[:, [1, 4]], torch.zeros(2), atol=1e-7, rtol=0
        )
        self.assertTrue(
            torch.equal(layer.weight[:, [0, 2, 3, 5, 6, 7]], original[:, [0, 2, 3, 5, 6, 7]])
        )
        restore(undo)
        self.assertTrue(torch.equal(layer.weight, original))

    def test_novel_spelling_does_not_hide_repetition(self):
        import json

        old = "grep -rn target src/ | head -20"
        command = old + "; " + old
        text = (
            "<tool_call>"
            + json.dumps({"name": "bash", "arguments": {"command": command}})
            + "</tool_call>"
        )
        metrics = command_metrics(text, [old])
        self.assertFalse(metrics["repeated"])
        self.assertTrue(metrics["internal_repeat"])
        self.assertFalse(metrics["valid_distinct"])


if __name__ == "__main__":
    unittest.main()
