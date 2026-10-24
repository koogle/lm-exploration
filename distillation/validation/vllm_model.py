"""The student decoder (``training/10m/model.py``) written for vLLM, loading ``model.pt``.

Same modules and parameter names as the PyTorch definition: RMSNorm, interleaved
(GPT-J style) RoPE, biased QKV/output/SwiGLU projections and tied embeddings. vLLM
supplies paged attention and batching; weights load straight from the training
checkpoint (``{"model": state_dict, "config": ..., "settings": ...}``) with
``load_format="pt"``. Single GPU only (replicated linear layers).
"""

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

ARCHITECTURE = "SimpleLLMForCausalLM"


class StudentAttention(nn.Module):
    def __init__(self, config, cache_config, prefix):
        super().__init__()
        width, heads = config.hidden_size, config.num_attention_heads
        self.width, self.head_dim = width, width // heads
        self.qkv = ReplicatedLinear(width, 3 * width, bias=True, prefix=f"{prefix}.qkv")
        self.projection = ReplicatedLinear(
            width, width, bias=True, prefix=f"{prefix}.projection"
        )
        self.rotary = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=False,  # adjacent feature pairs, as in model_10m.model.rotary
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )
        self.attn = Attention(
            heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions, x):
        qkv, _ = self.qkv(x)
        q, k, v = qkv.split(self.width, dim=-1)
        q, k = self.rotary(positions, q, k)
        out, _ = self.projection(self.attn(q, k, v))
        return out


class StudentMLP(nn.Module):
    def __init__(self, width, prefix):
        super().__init__()
        hidden = (8 * width + 2) // 3
        self.gate = ReplicatedLinear(width, hidden, bias=True, prefix=f"{prefix}.gate")
        self.up = ReplicatedLinear(width, hidden, bias=True, prefix=f"{prefix}.up")
        self.down = ReplicatedLinear(hidden, width, bias=True, prefix=f"{prefix}.down")

    def forward(self, x):
        out, _ = self.down(F.silu(self.gate(x)[0]) * self.up(x)[0])
        return out


class StudentBlock(nn.Module):
    def __init__(self, config, cache_config, prefix):
        super().__init__()
        width = config.hidden_size
        self.attention_norm = RMSNorm(width, eps=1e-5)
        self.attention = StudentAttention(config, cache_config, f"{prefix}.attention")
        self.mlp_norm = RMSNorm(width, eps=1e-5)
        self.mlp = StudentMLP(width, f"{prefix}.mlp")

    def forward(self, positions, x):
        x = x + self.attention(positions, self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class SimpleLLMForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        self.token_embedding = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.blocks = nn.ModuleList(
            StudentBlock(config, cache_config, f"blocks.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=1e-5)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size
        ).tie_weights(self.token_embedding)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.token_embedding(input_ids)

    get_input_embeddings = embed_input_ids

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        x = self.token_embedding(input_ids) if inputs_embeds is None else inputs_embeds
        for block in self.blocks:
            x = block(positions, x)
        return self.final_norm(x)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``model`` entry of a training checkpoint; other entries are metadata."""
        parameters = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            if name != "model":
                continue
            for key, tensor in value.items():
                if key == "lm_head.weight":  # tied to token_embedding.weight
                    continue
                parameter = parameters[key]
                getattr(parameter, "weight_loader", default_weight_loader)(
                    parameter, tensor
                )
                loaded.add(key)
        # vLLM's own attention scales (``*.attn.*``) are not checkpoint weights.
        missing = {n for n in set(parameters) - loaded if ".attn." not in n}
        if missing:
            raise ValueError(f"checkpoint is missing parameters: {sorted(missing)[:5]}")
        return loaded
