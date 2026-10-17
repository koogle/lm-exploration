"""10M RMSNorm ablation: preserve vanilla attention, MLPs, positions and initialization.

Replace both block norms and the final norm. Match vanilla LayerNorm's epsilon
(1e-5), so the comparison changes the normalization rule rather than epsilon too.
"""

import torch
from model_10m import model as vanilla
from model_10m.model import ModelConfig
from torch import nn

__all__ = ["GPT", "ModelConfig", "RMSNorm"]
BASE_MODEL_SOURCE = vanilla.__file__


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        # Accumulate low-precision statistics in FP32; retain FP64 for reference checks.
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        mean_square = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(mean_square + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class GPT(vanilla.GPT):
    def __init__(self, config):
        super().__init__(config)
        for block in self.blocks:
            block.attention_norm = RMSNorm(config.n_embd)
            block.mlp_norm = RMSNorm(config.n_embd)
        self.final_norm = RMSNorm(config.n_embd)
