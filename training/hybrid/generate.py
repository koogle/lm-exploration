"""Incremental decoding for HybridLM in plain PyTorch (CPU or GPU, no fla/Flex).

Each layer keeps a cache per row: full attention its rotated keys and values; KDA
the last `kda_conv_size - 1` projections for its short convolutions and its fp32
recurrent state. One token per row then costs one pass through the weights instead
of a forward over the whole prefix. The arithmetic mirrors `HybridLM.forward`
(tests check logits against it), including documents: after a row feeds EOS, its
next token starts a new document (attention reads only keys from there on, the
convolutions and state reset and RoPE positions restart).

`prefill` reads each prompt with one forward pass instead and builds the same cache
for every row, left-padded to the longest prompt (pad keys are masked through each
row's document start), so `generate_batch` costs one forward per prompt plus one
step per new token.
"""

import math

import torch
from torch.nn import functional as F

from .kda import kda_gate
from .model import DocumentMask, HybridLM

PREFILL_MIN = 1024


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
    # Query heads grouped by their shared key head: [rows, kv heads, group, width].
    rows, heads, kv_heads = x.size(0), query.size(1), key.size(1)
    grouped = query.view(rows, kv_heads, heads // kv_heads, query.size(-1))
    scores = (grouped @ key.transpose(-1, -2)).float() / math.sqrt(query.size(-1))
    index = torch.arange(key.size(2), device=x.device)
    other_documents = index[None, :] < cache.starts[:, None]
    scores = scores.masked_fill(other_documents[:, None, None], float("-inf"))
    lse = scores.logsumexp(-1)
    weights = (scores - lse[..., None]).exp().to(value.dtype)
    attended = (weights @ value).float().view(rows, heads, 1, -1)
    kept = torch.sigmoid(lse.view(rows, heads) - attention.sink.float())[..., None, None]
    attended = (attended * kept).to(x.dtype).transpose(1, 2).reshape(rows, 1, -1)
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
    if top_p >= 1:
        return torch.multinomial(probs, 1, generator=generator)[:, 0]
    ordered, index = probs.sort(-1, descending=True)
    ordered = ordered * (ordered.cumsum(-1) - ordered < top_p)
    choice = torch.multinomial(ordered / ordered.sum(-1, keepdim=True), 1, generator=generator)
    return index.gather(-1, choice)[:, 0]


def _prefill_row(lm, prompt, device):
    """Next-token logits [vocab] after one prompt and its per-layer cache entries.

    On CUDA the prompt is padded to a power of two (at least PREFILL_MIN, at most the
    context), so compiled attention sees few shapes; the padding is one second
    document and changes nothing in the prompt.
    """
    length = len(prompt)
    width = length
    if lm.token_embedding.weight.is_cuda:
        width = min(max(PREFILL_MIN, 1 << (length - 1).bit_length()), lm.config.block_size)
    tokens = torch.tensor([list(prompt) + [lm.config.eos_id] * (width - length)], device=device)
    documents = (torch.arange(width, device=device) >= length).long()[None]
    mask = DocumentMask(documents)
    rope = (lm.rope_cos[mask.positions][:, None], lm.rope_sin[mask.positions][:, None])
    x = lm.token_embedding(tokens)
    layers = []
    for block in lm.blocks:
        normed = block.mixer_norm(x)
        layer = {}
        if block.kind == "kda":
            mixed = block.mixer(normed, rope, mask, cache=layer)
        else:
            _, key, value = block.mixer.project(normed, rope)
            layer["key"], layer["value"] = key[:, :, :length].clone(), value[:, :, :length].clone()
            mixed = block.mixer(normed, rope, mask)
        x = x + mixed
        x = x + block.mlp(block.mlp_norm(x))
        layers.append(layer)
    logits = F.linear(lm.final_norm(x[:, length - 1]), lm.output_weight)[0].float()
    return logits, layers


@torch.no_grad()
def prefill(lm, prompts):
    """Logits [rows, vocab] after each prompt and the batched Cache that continues them.

    Each prompt is one document (it may not contain EOS).
    """
    if not prompts or not all(prompts):
        raise ValueError("every prompt must contain at least one token")
    if any(lm.config.eos_id in p for p in prompts):
        raise ValueError("prompts are single documents and may not contain EOS")
    if max(len(p) for p in prompts) > lm.config.block_size:
        raise ValueError("a prompt is longer than the model's context")
    device = lm.token_embedding.weight.device
    rows = [_prefill_row(lm, p, device) for p in prompts]
    width = max(len(p) for p in prompts)
    cache = Cache(lm, len(prompts))
    cache.length = width
    cache.positions = torch.tensor([len(p) for p in prompts], device=device)
    cache.starts = width - cache.positions
    cache.fresh = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    for index, layer in enumerate(cache.layers):
        for name in rows[0][1][index]:
            parts = [entry[index][name] for _, entry in rows]
            if name in ("key", "value"):
                parts = [F.pad(part, (0, 0, width - part.size(2), 0)) for part in parts]
            layer[name] = torch.cat(parts)
    return torch.stack([logits for logits, _ in rows]), cache


@torch.no_grad()
def generate_batch(
    lm, prompts, max_new_tokens, stop_ids, temperature=0.0, top_p=1.0, generator=None, done=None
):
    """New token ids per prompt, each up to and including its first stop id.

    Prompts are read by `prefill` (one document each), so each row decodes as it
    would alone. `done(row, ids)` may end a row early (for example on a stop string).
    """
    logits, cache = prefill(lm, prompts)
    room = lm.config.block_size + 1 - max(len(p) for p in prompts)
    outputs = [[] for _ in prompts]
    finished = [False] * len(prompts)
    stops = torch.tensor(sorted(stop_ids), device=logits.device)
    limit = max(0, min(max_new_tokens, room))
    for index in range(limit):
        token = _sample(logits, temperature, top_p, generator)
        stopped = torch.isin(token, stops).tolist()
        for row, value in enumerate(token.tolist()):
            if finished[row]:
                continue
            outputs[row].append(value)
            finished[row] = stopped[row] or bool(done and done(row, outputs[row]))
        if all(finished) or index == limit - 1:
            break
        logits = step(lm, token, cache)
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
