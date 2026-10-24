"""Export a student checkpoint as a Llama checkpoint that vLLM serves natively.

The student (``training/10m/model.py``) is a Llama-style decoder with biases on all
attention and MLP projections and tied embeddings. Its RoPE rotates adjacent feature
pairs (0, 1), (2, 3), ...; Llama rotates (i, i + d/2). Reordering each head's query
and key rows to [0, 2, 4, ..., 1, 3, 5, ...] makes the two equivalent; values and the
output projection are unchanged.
"""

import json
from pathlib import Path

import numpy as np
import torch


def rope_order(head_dim):
    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])


def llama_state(state, config):
    """Map a student state dict to Llama tensor names (Q/K rows reordered per head)."""
    heads, width = config["n_head"], config["n_embd"]
    head_dim = width // heads
    order = torch.cat([h * head_dim + rope_order(head_dim) for h in range(heads)])
    out = {
        "model.embed_tokens.weight": state["token_embedding.weight"],
        "model.norm.weight": state["final_norm.weight"],
    }
    for layer in range(config["n_layer"]):
        source, target = f"blocks.{layer}.", f"model.layers.{layer}."
        weight = state[source + "attention.qkv.weight"]
        bias = state[source + "attention.qkv.bias"]
        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
            rows = slice(index * width, (index + 1) * width)
            w, b = weight[rows], bias[rows]
            if name != "v_proj":
                w, b = w[order], b[order]
            out[target + f"self_attn.{name}.weight"] = w
            out[target + f"self_attn.{name}.bias"] = b
        out[target + "self_attn.o_proj.weight"] = state[
            source + "attention.projection.weight"
        ]
        out[target + "self_attn.o_proj.bias"] = state[
            source + "attention.projection.bias"
        ]
        for name in ("gate", "up", "down"):
            out[target + f"mlp.{name}_proj.weight"] = state[
                source + f"mlp.{name}.weight"
            ]
            out[target + f"mlp.{name}_proj.bias"] = state[source + f"mlp.{name}.bias"]
        out[target + "input_layernorm.weight"] = state[source + "attention_norm.weight"]
        out[target + "post_attention_layernorm.weight"] = state[
            source + "mlp_norm.weight"
        ]
    return {
        name: tensor.detach().to(torch.float32).contiguous()
        for name, tensor in out.items()
    }


def llama_config(config, bos, eos):
    width, heads = config["n_embd"], config["n_head"]
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config["vocab_size"],
        "hidden_size": width,
        "intermediate_size": (8 * width + 2) // 3,
        "num_hidden_layers": config["n_layer"],
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "head_dim": width // heads,
        "hidden_act": "silu",
        "max_position_embeddings": config["block_size"],
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "attention_bias": True,
        "mlp_bias": True,
        "tie_word_embeddings": True,
        "bos_token_id": bos,
        "eos_token_id": eos,
        "torch_dtype": "float32",
    }


def export(checkpoint, vocab, eos, out_dir):
    """Write ``config.json``, ``model.safetensors`` and ``vocab.npz`` to ``out_dir``."""
    from safetensors.torch import save_file

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = llama_state(saved["model"], saved["config"])  # lm_head is tied: not saved
    save_file(tensors, out_dir / "model.safetensors", metadata={"format": "pt"})
    config = llama_config(saved["config"], vocab.bos, eos)
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    arrays = {
        "table": vocab.table,
        "native": vocab.native,
        "bos": np.asarray(vocab.bos),
    }
    if vocab.pieces is not None:
        arrays.update(pieces=vocab.pieces, offsets=vocab.offsets, lengths=vocab.lengths)
    np.savez(out_dir / "vocab.npz", **arrays)
    return out_dir
