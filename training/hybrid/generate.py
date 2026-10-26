"""Incremental decoding for HybridLM in plain PyTorch (CPU or GPU, no fla/Flex).

Each layer keeps a cache per row: full attention its rotated keys and values; KDA
the last `kda_conv_size - 1` projections for its short convolutions and its fp32
recurrent state. One token per row then costs one pass through the weights instead
of a forward over the whole prefix. The arithmetic mirrors `HybridLM.forward`
(tests check logits against it), including documents: after a row feeds EOS, its
next token starts a new document (attention reads only keys from there on, the
convolutions and state reset and RoPE positions restart). `generate_batch` uses
that to left-pad prompts of different lengths with EOS.
"""

import math

import torch
from torch.nn import functional as F

from .kda import kda_gate
from .model import HybridLM


class Cache:
    def __init__(self, lm, rows=1):
        device = lm.token_embedding.weight.device
        self.length = 0
        self.positions = torch.zeros(rows, dtype=torch.long, device=device)
        # Rows whose previous token ended a document (all rows before the first token).
        self.fresh = torch.ones(rows, dtype=torch.bool, device=device)
        # Cache index where each row's current document starts.
        self.starts = torch.zeros(rows, dtype=torch.long, device=device)
        self.layers = [{} for _ in lm.blocks]


def _attention_step(attention, x, cache, layer, cos, sin):
    query, key, value = attention.project(x, (cos[:, None, None], sin[:, None, None]))
    if "key" in layer:
        key = torch.cat((layer["key"], key), 2)
        value = torch.cat((layer["value"], value), 2)
    layer["key"], layer["value"] = key, value
    groups = query.size(1) // key.size(1)
    keys = key.repeat_interleave(groups, 1).float()
    values = value.repeat_interleave(groups, 1).float()
    scores = query.float() @ keys.transpose(-1, -2) / math.sqrt(query.size(-1))
    index = torch.arange(key.size(2), device=x.device)
    other_documents = index[None, :] < cache.starts[:, None]
    scores = scores.masked_fill(other_documents[:, None, None], float("-inf"))
    lse = scores.logsumexp(-1)
    attended = (scores - lse[..., None]).exp() @ values
    kept = torch.sigmoid(lse - attention.sink.float()[:, None])
    rows = x.size(0)
    attended = (attended * kept[..., None]).to(x.dtype).transpose(1, 2).reshape(rows, 1, -1)
    gate = torch.sigmoid(attention.gate_proj(x) + attention.gate_bias)
    return attention.o_proj(attended * gate.to(attended.dtype))


def _convolution_step(x, weight, layer, name, fresh):
    """One step of `kda.short_convolution` for x [B, C]; history holds earlier inputs."""
    width = weight.size(-1)
    history = layer.get(name)
    if history is None:
        history = x.new_zeros(x.size(0), width - 1, x.size(1))
    history = history.masked_fill(fresh[:, None, None], 0)
    out = x * weight[:, 0, -1]
    for shift in range(1, width):
        out = out + history[:, -shift] * weight[:, 0, -1 - shift]
    layer[name] = torch.cat((history[:, 1:], x[:, None]), 1)
    return F.silu(out)


def _kda_step(kda, x, layer, fresh):
    rows = x.size(0)
    shape = (rows, kda.heads, kda.head_dim)
    point = x[:, 0]
    q = _convolution_step(kda.q_proj(point), kda.q_conv, layer, "q", fresh).float()
    k = _convolution_step(kda.k_proj(point), kda.k_conv, layer, "k", fresh).float()
    v = _convolution_step(kda.v_proj(point), kda.v_conv, layer, "v", fresh).float()
    q = F.normalize(q.view(shape), dim=-1) * kda.head_dim**-0.5
    k = F.normalize(k.view(shape), dim=-1)
    v = v.view(shape)
    beta = torch.sigmoid(kda.b_proj(point).float())
    decay = kda_gate(kda.f_proj(point).view(shape), kda.A_log, kda.dt_bias).exp()
    state = layer.get("state")
    if state is None:
        state = x.new_zeros(rows, kda.heads, kda.head_dim, kda.head_dim, dtype=torch.float32)
    state = state.masked_fill(fresh[:, None, None, None], 0) * decay[..., None]
    error = v - torch.einsum("bhk,bhkv->bhv", k, state)
    state = state + torch.einsum("bhk,bhv->bhkv", beta[..., None] * k, error)
    layer["state"] = state
    out = torch.einsum("bhk,bhkv->bhv", q, state)
    out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + kda.norm_eps)
    out = out * kda.o_norm_weight.float()
    gate = torch.sigmoid(kda.g_proj(point).float().view(shape))
    return kda.o_proj((out * gate).to(x.dtype).reshape(rows, 1, -1))


@torch.no_grad()
def step(lm: HybridLM, tokens, cache: Cache):
    """Logits [rows, vocab] for the next token after feeding `tokens` (one id per row)."""
    tokens = torch.as_tensor(tokens, device=cache.positions.device).view(-1)
    fresh = cache.fresh
    cache.positions = torch.where(fresh, 0, cache.positions)
    cache.starts = torch.where(fresh, cache.length, cache.starts)
    if int(cache.positions.max()) >= lm.config.block_size:
        raise ValueError("context is full")
    x = lm.token_embedding(tokens[:, None])
    cos, sin = lm.rope_cos[cache.positions], lm.rope_sin[cache.positions]
    for block, layer in zip(lm.blocks, cache.layers, strict=True):
        normed = block.mixer_norm(x)
        if block.kind == "kda":
            mixed = _kda_step(block.mixer, normed, layer, fresh)
        else:
            mixed = _attention_step(block.mixer, normed, cache, layer, cos, sin)
        x = x + mixed
        x = x + block.mlp(block.mlp_norm(x))
    cache.length += 1
    cache.positions = cache.positions + 1
    cache.fresh = tokens == lm.config.eos_id
    return F.linear(lm.final_norm(x), lm.output_weight)[:, 0].float()


def _sample(logits, temperature, top_p, generator):
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, -1)
    ordered, index = probs.sort(-1, descending=True)
    ordered = ordered * (ordered.cumsum(-1) - ordered < top_p)
    choice = torch.multinomial(ordered / ordered.sum(-1, keepdim=True), 1, generator=generator)
    return index.gather(-1, choice)[:, 0]


@torch.no_grad()
def generate_batch(
    lm, prompts, max_new_tokens, stop_ids, temperature=0.0, top_p=1.0, generator=None, done=None
):
    """New token ids per prompt, each up to and including its first stop id.

    Prompts are left-padded with EOS, which ends a document, so each row decodes
    exactly as it would alone. `done(row, ids)` may end a row early (for example on
    a stop string).
    """
    if not prompts or not all(prompts):
        raise ValueError("every prompt must contain at least one token")
    width = max(len(p) for p in prompts)
    padded = [[lm.config.eos_id] * (width - len(p)) + list(p) for p in prompts]
    if width + max_new_tokens > lm.config.block_size + 1:
        max_new_tokens = lm.config.block_size + 1 - width
    cache = Cache(lm, len(prompts))
    columns = torch.tensor(padded, device=cache.positions.device).T
    for column in columns[:-1]:
        step(lm, column, cache)
    token = columns[-1]
    outputs = [[] for _ in prompts]
    finished = [False] * len(prompts)
    stops = torch.tensor(sorted(stop_ids), device=columns.device)
    for _ in range(max(0, max_new_tokens)):
        token = _sample(step(lm, token, cache), temperature, top_p, generator)
        stopped = torch.isin(token, stops).tolist()
        for row, value in enumerate(token.tolist()):
            if finished[row]:
                continue
            outputs[row].append(value)
            finished[row] = stopped[row] or bool(done and done(row, outputs[row]))
        if all(finished):
            break
    return outputs


@torch.no_grad()
def generate(lm, prompt_ids, max_new_tokens, stop_ids, temperature=0.0, top_p=1.0, generator=None):
    """Yield new token ids up to and including one of `stop_ids`; greedy at temperature 0,
    else nucleus sampling."""
    if not prompt_ids:
        raise ValueError("the prompt must contain at least one token")
    cache = Cache(lm)
    for token in prompt_ids[:-1]:
        step(lm, [token], cache)
    token = prompt_ids[-1]
    for _ in range(max_new_tokens):
        if int(cache.positions[0]) >= lm.config.block_size:
            return
        token = int(_sample(step(lm, [token], cache), temperature, top_p, generator)[0])
        yield token
        if token in stop_ids:
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
