"""Weight-tying-only variant of 100m/model.py; depth, width and blocks are unchanged.

Initialize the vanilla model first, then reuse its input embedding Parameter as
the output projection. This preserves all other initial weights under the same
seed and registers the shared matrix once in model.parameters().
"""

from model_10m import model as vanilla
from model_100m import model as baseline
from model_100m.model import ModelConfig

__all__ = ["GPT", "ModelConfig"]
MODEL_DEPENDENCIES = {
    "model_100m_definition": baseline.__file__,
    "model_10m_definition": vanilla.__file__,
}


class GPT(baseline.GPT):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head.weight = self.token_embedding.weight
