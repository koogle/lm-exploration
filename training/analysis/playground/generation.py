"""Key/value-cached sampling for the decoder in `model.py` (copied from training/10m)."""

import torch
from model import rotary
from torch.nn import functional as F


@torch.no_grad()
def step(model, tokens, cache, offset):
    """Logits after `tokens` at positions `offset:`, extending each layer's cache."""
    x = model.token_embedding(torch.tensor([tokens]))
    for layer, block in enumerate(model.blocks):
        attention = block.attention
        h = block.attention_norm(x)
        batch, length, width = h.shape
        heads = attention.n_head
        q, k, v = (
            t.view(batch, length, heads, width // heads).transpose(1, 2)
            for t in attention.qkv(h).chunk(3, -1)
        )
        q, k = rotary(q, offset), rotary(k, offset)
        if cache[layer] is not None:
            k = torch.cat([cache[layer][0], k], 2)
            v = torch.cat([cache[layer][1], v], 2)
        cache[layer] = (k, v)
        # Queries attend to every cached position and causally among themselves.
        mask = torch.ones(length, k.size(2), dtype=torch.bool).tril(k.size(2) - length)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + attention.projection(attended.transpose(1, 2).reshape(batch, length, width))
        x = x + block.mlp(block.mlp_norm(x))
    return model.lm_head(model.final_norm(x[:, -1]))[0]


def choose(logits, generated, temperature, top_k, top_p, penalty, generator):
    """Repetition penalty, then greedy at temperature 0, else top-k and nucleus sampling."""
    logits = logits.clone()
    if penalty != 1.0 and generated:
        seen = torch.tensor(sorted(set(generated)))
        values = logits[seen]
        logits[seen] = torch.where(values > 0, values / penalty, values * penalty)
    if temperature == 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        cutoff = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    probabilities = logits.softmax(-1)
    if top_p < 1.0:
        ordered, order = probabilities.sort(descending=True)
        # Keep the tokens up to and including the one that crosses top_p.
        drop = ordered.cumsum(-1) - ordered >= top_p
        probabilities[order[drop]] = 0
        probabilities /= probabilities.sum()
    return int(torch.multinomial(probabilities, 1, generator=generator))


def complete(
    model, ids, tokens, temperature, top_k, top_p, penalty, seed, eos, block_size, stop=None
):
    """Up to `tokens` generated IDs after prompt `ids`, ending before EOS or once `stop`."""
    ids = ids[-(block_size - tokens) :] if len(ids) + tokens > block_size else ids
    generator = torch.Generator().manual_seed(seed)
    cache = [None] * len(model.blocks)
    logits, generated = step(model, ids, cache, 0), []
    for _ in range(tokens):
        token = choose(logits, generated, temperature, top_k, top_p, penalty, generator)
        if token == eos:
            break
        generated.append(token)
        if stop is not None and stop(generated):
            break
        logits = step(model, [token], cache, len(ids) + len(generated) - 1)
    return generated
