"""10M SwiGLU ablation: preserve LayerNorm, learned positions, MHA and biases.

Use hidden width ceil(8 * width / 3), rather than the GELU MLP's 4 * width,
to match matrix parameters (exactly at default width 384). Keep biases on all
three projections and vanilla normal initialization with standard deviation 0.02.
"""

from model_10m import model as vanilla
from model_10m.model import ModelConfig
from torch import nn
from torch.nn import functional as F

__all__ = ["GPT", "ModelConfig", "SwiGLU"]
BASE_MODEL_SOURCE = vanilla.__file__


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.hidden_width = (8 * width + 2) // 3
        self.gate = nn.Linear(width, self.hidden_width)
        self.up = nn.Linear(width, self.hidden_width)
        self.down = nn.Linear(self.hidden_width, width)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GPT(vanilla.GPT):
    def __init__(self, config):
        super().__init__(config)
        for block in self.blocks:
            block.mlp = SwiGLU(config.n_embd)
            block.mlp.apply(self.initialize)
