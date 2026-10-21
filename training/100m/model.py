"""100M-family configuration of the shared modern decoder.

Version 1 retains historical vanilla checkpoint compatibility.
"""

from dataclasses import dataclass

from model_10m import model as backbone
from model_10m.model import (
    GPT,
    Attention,
    Block,
    RMSNorm,
    SwiGLU,
    causal_attention,
    rotary,
)

__all__ = [
    "GPT",
    "Attention",
    "Block",
    "ModelConfig",
    "RMSNorm",
    "SwiGLU",
    "causal_attention",
    "rotary",
]
MODEL_DEPENDENCIES = {
    "model_10m_definition": backbone.__file__,
    **backbone.MODEL_DEPENDENCIES,
}


@dataclass(frozen=True)
class ModelConfig(backbone.ModelConfig):
    n_layer: int = 14
    n_head: int = 12
    n_embd: int = 768
