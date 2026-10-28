"""Batched key/value-cached decoding for our decoders (`model_10m.model.GPT` family).

The one sampler the repository uses for generation: checkpoint benchmarks, the playground
Space and RL rollouts. Prompts of different lengths are left-padded into one batch. Each
row's positions count from its first real token, padded keys are masked out, and every
new token is computed once and cached, so a row's logits equal those of the model run on
that row alone (up to numerical precision). It runs on CPU or CUDA (BF16 autocast).

Choosing a token: the repetition penalty divides positive and multiplies negative
logits of tokens the row already generated; temperature 0 is greedy; otherwise logits
are divided by the temperature, cut to the `top_k` largest (0 keeps all) and to the
smallest set whose probability reaches `top_p` (keeping the token that crosses it), and
sampled. With `temperature` above 0 the sampler also returns its log-probability of each
sampled token and, with `record_top_k`, its `record_top_k` likeliest tokens and their
log-probabilities, all from the distribution the token was drawn from: what RL needs to
compare sampler and learner (`trainer/rl/`).
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def rotate(x, positions):
    """RoPE as in the model's `rotary`, with a position per row and token."""
    width = x.size(-1)
    frequencies = 10000.0 ** (
        -torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width
    )
    angles = positions.float()[:, None, :, None] * frequencies
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


def forward(model, tokens, positions, cache, start, valid):
    """Last-token logits after writing `tokens` to cache slots `start:`.

    `valid[b, j]` says whether cache slot j of row b holds a real token.
    """
    batch, length = tokens.shape
    end = start + length
    slots = torch.arange(end, device=tokens.device)
    queries = start + torch.arange(length, device=tokens.device)
    causal = slots[None, :] <= queries[:, None]
    # A padding query attends to itself only, so no attention row is empty.
    mask = (valid[:, None, None, :end] & causal) | (slots[None, :] == queries[:, None])
    x = model.token_embedding(tokens)
    for layer, block in enumerate(model.blocks):
        attention = block.attention
        h = block.attention_norm(x)
        heads = attention.n_head
        q, k, v = (
            t.view(batch, length, heads, -1).transpose(1, 2) for t in attention.qkv(h).chunk(3, -1)
        )
        q, k = rotate(q, positions), rotate(k, positions)
        if cache[layer] is None:
            shape = (batch, heads, valid.size(1), k.size(-1))
            cache[layer] = (k.new_zeros(shape), v.new_zeros(shape))
        keys, values = cache[layer]
        keys[:, :, start:end], values[:, :, start:end] = k, v
        attended = F.scaled_dot_product_attention(
            q, keys[:, :, :end], values[:, :, :end], attn_mask=mask
        )
        x = x + attention.projection(attended.transpose(1, 2).reshape(batch, length, -1))
        x = x + block.mlp(block.mlp_norm(x))
    return model.lm_head(model.final_norm(x[:, -1])).float()


def choose(logits, seen, temperature, top_k, top_p, penalty, generator):
    """(next token per row, log-probabilities `[batch, vocab]` it was drawn from, None
    when greedy) from `[batch, vocab]` logits; `seen` marks generated tokens."""
    if penalty != 1.0:
        logits = torch.where(
            seen, torch.where(logits > 0, logits / penalty, logits * penalty), logits
        )
    if temperature == 0:
        return logits.argmax(-1), None
    logits = logits / temperature
    if top_k:
        cutoff = logits.topk(min(top_k, logits.size(-1)), -1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    probabilities = logits.softmax(-1)
    if top_p < 1.0:
        ordered, order = probabilities.sort(-1, descending=True)
        # Keep the tokens up to and including the one that crosses top_p.
        drop = ordered.cumsum(-1) - ordered >= top_p
        probabilities = probabilities.scatter(-1, order, ordered.masked_fill(drop, 0))
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    chosen = torch.multinomial(probabilities, 1, generator=generator)[:, 0]
    return chosen, probabilities.log()


@dataclass
class Samples:
    """New token IDs per prompt (without EOS) and whether each ended in EOS; when
    sampling, the sampler's log-probability of each new token (EOS included) `logprobs`
    `[batch, steps]`, and with `record_top_k` its likeliest tokens `top_ids` and their
    `top_logprobs` `[batch, steps, record_top_k]`."""

    completions: list
    finished: list
    logprobs: torch.Tensor | None = None
    top_ids: torch.Tensor | None = None
    top_logprobs: torch.Tensor | None = None


@torch.no_grad()
def sample(
    model,
    prompts,
    max_new_tokens,
    eos,
    temperature=1.0,
    generator=None,
    top_k=0,
    top_p=1.0,
    penalty=1.0,
    stop=None,
    record_top_k=0,
):
    """Decode up to `max_new_tokens` after each prompt (token ID lists); see `Samples`.

    A row also ends once `stop(generated IDs)` is true; its completion keeps those IDs.
    """
    device = model.token_embedding.weight.device
    batch, width = len(prompts), max(map(len, prompts))
    total = width + max_new_tokens
    if total > model.config.block_size:
        raise ValueError("Prompt plus new tokens exceed the model's context")
    tokens = torch.zeros((batch, width), dtype=torch.long, device=device)
    valid = torch.zeros((batch, total), dtype=torch.bool, device=device)
    for row, prompt in enumerate(prompts):
        tokens[row, width - len(prompt) :] = torch.tensor(prompt, device=device)
        valid[row, width - len(prompt) : width] = True
    positions = (valid[:, :width].long().cumsum(1) - 1).clamp(min=0)
    lengths = torch.tensor([len(p) for p in prompts], device=device)
    cache = [None] * len(model.blocks)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    generated = torch.zeros((batch, max_new_tokens), dtype=torch.long, device=device)
    seen = torch.zeros((batch, model.config.vocab_size), dtype=torch.bool, device=device)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    ended = torch.zeros(batch, dtype=torch.bool, device=device)
    record = temperature != 0
    if record:
        logprobs = torch.zeros((batch, max_new_tokens), device=device)
        top_ids = torch.zeros(
            (batch, max_new_tokens, record_top_k), dtype=torch.long, device=device
        )
        top_logprobs = torch.zeros((batch, max_new_tokens, record_top_k), device=device)
    with autocast:
        logits = forward(model, tokens, positions, cache, 0, valid)
    steps = 0
    for step in range(max_new_tokens):
        chosen, scores = choose(logits, seen, temperature, top_k, top_p, penalty, generator)
        if record:
            logprobs[:, step] = scores.gather(1, chosen[:, None])[:, 0]
            if record_top_k:
                top_logprobs[:, step], top_ids[:, step] = scores.topk(record_top_k, -1)
        chosen = torch.where(done | ended, eos, chosen)
        generated[:, step] = chosen
        seen[torch.arange(batch, device=device), chosen] = True
        done |= (chosen == eos) & ~ended
        steps = step + 1
        if stop is not None:
            for row in (~done & ~ended).nonzero()[:, 0].tolist():
                if stop(generated[row, :steps].tolist()):
                    ended[row] = True
        if bool((done | ended).all()):
            break
        valid[:, width + step] = True
        with autocast:
            logits = forward(
                model, chosen[:, None], (lengths + step)[:, None], cache, width + step, valid
            )
    completions = []
    for row in generated[:, :steps].tolist():
        completions.append(row[: row.index(eos)] if eos in row else row)
    if not record:
        return Samples(completions, done.tolist())
    recorded = (logprobs[:, :steps], top_ids[:, :steps], top_logprobs[:, :steps])
    if not record_top_k:
        recorded = (recorded[0], None, None)
    return Samples(completions, done.tolist(), *recorded)
