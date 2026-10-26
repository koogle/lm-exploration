"""Incremental decoding for HybridLM in plain PyTorch (CPU or GPU, no fla/Flex).

Each layer keeps a cache: full attention its rotated keys and values; KDA the last
`kda_conv_size - 1` projections for its short convolutions and its fp32 recurrent
state. One token then costs one pass through the weights instead of a forward over
the whole prefix. The arithmetic mirrors `HybridLM.forward` (tests check logits
against it). Generation covers one document: it stops at EOS, which would end the
document and reset every cache.
"""

import math

import torch
from torch.nn import functional as F

from .kda import kda_gate
from .model import HybridLM


class Cache:
    def __init__(self, lm):
        self.position = 0
        self.layers = [{} for _ in lm.blocks]


def _attention_step(attention, x, cache, position, cos, sin):
    query, key, value = attention.project(x, (cos[position][None, None], sin[position][None, None]))
    if "key" in cache:
        key = torch.cat((cache["key"], key), 2)
        value = torch.cat((cache["value"], value), 2)
    cache["key"], cache["value"] = key, value
    groups = query.size(1) // key.size(1)
    keys = key.repeat_interleave(groups, 1).float()
    values = value.repeat_interleave(groups, 1).float()
    scores = query.float() @ keys.transpose(-1, -2) / math.sqrt(query.size(-1))
    lse = scores.logsumexp(-1)
    attended = (scores - lse[..., None]).exp() @ values
    kept = torch.sigmoid(lse - attention.sink.float()[:, None])
    attended = (attended * kept[..., None]).to(x.dtype).transpose(1, 2).reshape(1, 1, -1)
    gate = torch.sigmoid(attention.gate_proj(x) + attention.gate_bias)
    return attention.o_proj(attended * gate.to(attended.dtype))


def _convolution_step(x, weight, history, position):
    """One step of `kda.short_convolution`; history holds earlier inputs, oldest first."""
    width = weight.size(-1)
    out = x * weight[:, 0, -1]
    for shift in range(1, width):
        if position >= shift and len(history) >= shift:
            out = out + history[-shift] * weight[:, 0, -1 - shift]
    history.append(x)
    del history[: -(width - 1)]
    return F.silu(out)


def _kda_step(kda, x, cache, position):
    shape = (kda.heads, kda.head_dim)
    if not cache:
        cache["history"] = {"q": [], "k": [], "v": []}
        cache["state"] = torch.zeros(kda.heads, kda.head_dim, kda.head_dim, device=x.device)
    histories = cache["history"]
    q = _convolution_step(kda.q_proj(x)[0, 0], kda.q_conv, histories["q"], position).float()
    k = _convolution_step(kda.k_proj(x)[0, 0], kda.k_conv, histories["k"], position).float()
    v = _convolution_step(kda.v_proj(x)[0, 0], kda.v_conv, histories["v"], position).float()
    q = F.normalize(q.view(shape), dim=-1) * kda.head_dim**-0.5
    k = F.normalize(k.view(shape), dim=-1)
    v = v.view(shape)
    beta = torch.sigmoid(kda.b_proj(x)[0, 0].float())
    decay = kda_gate(kda.f_proj(x)[0, 0].view(shape), kda.A_log, kda.dt_bias).exp()
    state = cache["state"] * decay[..., None]
    error = v - torch.einsum("hk,hkv->hv", k, state)
    state = state + torch.einsum("hk,hv->hkv", beta[:, None] * k, error)
    cache["state"] = state
    out = torch.einsum("hk,hkv->hv", q, state)
    out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + kda.norm_eps)
    out = out * kda.o_norm_weight.float()
    gate = torch.sigmoid(kda.g_proj(x)[0, 0].float().view(shape))
    return kda.o_proj((out * gate).to(x.dtype).reshape(1, 1, -1))


@torch.no_grad()
def step(lm: HybridLM, token: int, cache: Cache):
    """Logits [vocab] for the next token after feeding `token` at the cache's position."""
    position = cache.position
    if position >= lm.config.block_size:
        raise ValueError("context is full")
    weight = lm.token_embedding.weight
    x = lm.token_embedding(torch.tensor([[token]], device=weight.device))
    for block, layer in zip(lm.blocks, cache.layers, strict=True):
        normed = block.mixer_norm(x)
        if block.kind == "kda":
            mixed = _kda_step(block.mixer, normed, layer, position)
        else:
            mixed = _attention_step(block.mixer, normed, layer, position, lm.rope_cos, lm.rope_sin)
        x = x + mixed
        x = x + block.mlp(block.mlp_norm(x))
    cache.position += 1
    return F.linear(lm.final_norm(x), lm.output_weight)[0, 0].float()


@torch.no_grad()
def generate(lm, prompt_ids, max_new_tokens, eos_id, temperature=0.0, top_p=1.0, generator=None):
    """Yield new token ids; greedy at temperature 0, else nucleus sampling."""
    cache = Cache(lm)
    if not prompt_ids:
        raise ValueError("the prompt must contain at least one token")
    for token in prompt_ids[:-1]:
        step(lm, token, cache)
    token = prompt_ids[-1]
    for _ in range(max_new_tokens):
        if cache.position >= lm.config.block_size:
            return
        logits = step(lm, token, cache)
        if temperature <= 0:
            token = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, -1)
            ordered, index = probs.sort(descending=True)
            keep = ordered.cumsum(0) - ordered < top_p
            ordered = ordered * keep
            token = int(index[torch.multinomial(ordered / ordered.sum(), 1, generator=generator)])
        yield token
        if token == eos_id:
            return


def load_export(folder, dtype=torch.float32, device="cpu"):
    """The inference trunk from a bf16 export folder (config.json, model.safetensors).

    Exports from hybrid-720m-40b-01 predate `HybridConfig.eos_id` and carry its
    former `attention_kernel` field; both are reconciled here, with EOS from the
    pinned GLM tokenizer the run trained with.
    """
    import json
    from dataclasses import fields
    from pathlib import Path

    from safetensors.torch import load_file

    from . import tokenizer as glm
    from .model import HybridConfig

    folder = Path(folder)
    saved = json.loads((folder / "config.json").read_text())
    if saved.get("tokenizer_sha256", glm.SHA256) != glm.SHA256:
        raise ValueError("export was trained with a different tokenizer")
    known = {f.name for f in fields(HybridConfig)}
    settings = {k: tuple(v) if isinstance(v, list) else v for k, v in saved.items() if k in known}
    settings.update(eos_id=saved.get("eos_id", glm.EOS_ID), mtp=False)
    lm = HybridLM(HybridConfig(**settings))
    lm.load_state_dict(load_file(folder / "model.safetensors"))
    return lm.to(device=device, dtype=dtype).eval()
