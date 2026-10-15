"""Architecture only: the 10M vanilla GPT scaled to roughly 100M parameters.

Lineage: every module is imported unchanged from 10m/model.py, which holds the
explained definition. Only the default dimensions below differ: 14 layers, width
768, 12 heads and context 128 (10M: 6 layers, width 384, 6 heads, context 256).
The exact count depends on context length and vocabulary; trainer/loop.py records it.
Training, data loading, checkpoints and CLI code deliberately live elsewhere.
"""

from dataclasses import dataclass

from model_10m import model as vanilla
from model_10m.model import GPT, Attention, Block, causal_attention

__all__ = ["GPT", "Attention", "Block", "ModelConfig", "causal_attention"]


@dataclass(frozen=True)
class ModelConfig(vanilla.ModelConfig):
    block_size: int = 128
    n_layer: int = 14
    n_head: int = 12
    n_embd: int = 768
