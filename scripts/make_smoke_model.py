"""Create a deterministic 1-D Core ML model: y = 3*x + 1.

Run on macOS with coremltools and torch installed:
  python3 scripts/make_smoke_model.py
"""
from pathlib import Path

import coremltools as ct
import torch


class Affine(torch.nn.Module):
    def forward(self, x):
        return 3.0 * x + 1.0


root = Path(__file__).resolve().parents[1]
model = torch.jit.trace(Affine().eval(), torch.tensor([0.0], dtype=torch.float32))
converted = ct.convert(
    model,
    inputs=[ct.TensorType(name="x", shape=(1,), dtype=float)],
    outputs=[ct.TensorType(name="y", dtype=float)],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS15,
)
output = root / "artifacts" / "affine-smoke.mlpackage"
output.parent.mkdir(parents=True, exist_ok=True)
converted.save(str(output))
print(f"wrote {output}")
