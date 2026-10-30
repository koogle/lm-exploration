"""100M-family configuration of the shared modern decoder."""

from dataclasses import dataclass

from model_10m import model as backbone
from model_10m.model import (
    GPT,
    Attention,
    Block,
    Documents,
    RMSNorm,
    SwiGLU,
    document_attention,
    find_documents,
    rotary,
)

__all__ = [
    "GPT",
    "Attention",
    "Block",
    "Documents",
    "ModelConfig",
    "RMSNorm",
    "SwiGLU",
    "document_attention",
    "find_documents",
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
