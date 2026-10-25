"""Batched key/value-cached sampling for our decoders (`model_10m.model.GPT` family).

Prompts of different lengths are left-padded into one batch. Each row's positions count
from its first real token, padded keys are masked out, and every new token is computed
once and cached, so a row's logits equal those of the model run on that row alone (up to
numerical precision). Sampling uses the plain softmax at `temperature` (no top-k or
nucleus), so samples follow the policy's own probabilities; temperature 0 is greedy.

Two engines implement the same sampler. `eager` runs the PyTorch modules step by step
and works on any device; `graph` (`kernels/decode.py`, CUDA only) replays one captured
CUDA graph per decode step built from custom kernels. They agree up to BF16 rounding;
`compare_engines` measures how far each engine's log-probabilities are from the
learner's full forward pass.

When sampling, the sampler also returns its own log-probability of each sampled token
and, with `top_k`, its `top_k` most likely tokens and their log-probabilities: the
sampler-side information that Score Centering (`trainer/rl.py`) needs.
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


@dataclass
class Samples:
    """New token IDs per prompt (without EOS), whether each ended in EOS, and for
    sampled (not greedy) tokens the sampler's log-probabilities: of each new token
    (EOS included) `[batch, steps]`, and of its `top_k` likeliest tokens
    `top_ids`/`top_logprobs` `[batch, steps, top_k]` (None without `top_k`)."""

    completions: list
    finished: list
    logprobs: torch.Tensor | None = None
    top_ids: torch.Tensor | None = None
    top_logprobs: torch.Tensor | None = None


ENGINES = ("eager", "graph")


@torch.no_grad()
def sample(
    model, prompts, max_new_tokens, eos, temperature=1.0, generator=None, top_k=0, engine="eager"
):
    """Sample up to `max_new_tokens` after each prompt; see `Samples`."""
    if engine == "graph":
        return graph_sample(model, prompts, max_new_tokens, eos, temperature, generator, top_k)
    if engine != "eager":
        raise ValueError(f"engine must be one of {ENGINES}")
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
    greedy = temperature == 0
    logprobs = None if greedy else torch.zeros((batch, max_new_tokens), device=device)
    if top_k and not greedy:
        top_ids = torch.zeros((batch, max_new_tokens, top_k), dtype=torch.long, device=device)
        top_logprobs = torch.zeros((batch, max_new_tokens, top_k), device=device)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    with autocast:
        logits = forward(model, tokens, positions, cache, 0, valid)
    steps = 0
    for step in range(max_new_tokens):
        if greedy:
            chosen = logits.argmax(-1)
        else:
            scores = torch.log_softmax(logits / temperature, -1)
            chosen = torch.multinomial(scores.exp(), 1, generator=generator)[:, 0]
            logprobs[:, step] = scores.gather(1, chosen[:, None])[:, 0]
            if top_k:
                top_logprobs[:, step], top_ids[:, step] = scores.topk(top_k, -1)
        chosen = torch.where(done, eos, chosen)
        generated[:, step] = chosen
        done |= chosen == eos
        steps = step + 1
        if bool(done.all()):
            break
        valid[:, width + step] = True
        with autocast:
            logits = forward(
                model, chosen[:, None], (lengths + step)[:, None], cache, width + step, valid
            )
    completions = []
    for row in generated[:, :steps].tolist():
        completions.append(row[: row.index(eos)] if eos in row else row)
    result = Samples(completions, done.tolist())
    if not greedy:
        result.logprobs = logprobs[:, :steps]
        if top_k:
            result.top_ids, result.top_logprobs = top_ids[:, :steps], top_logprobs[:, :steps]
    return result


def graph_sample(model, prompts, max_new_tokens, eos, temperature, generator, top_k):
    from kernels import decode

    width = max(map(len, prompts))
    needed = width + max_new_tokens
    if needed > model.config.block_size:
        raise ValueError("Prompt plus new tokens exceed the model's context")
    step = decode.CAPACITY_STEP
    capacity = min(-(-needed // step) * step, model.config.block_size)
    greedy = temperature == 0
    runner = decode.decoder(model, len(prompts), capacity, top_k, greedy)
    steps = runner.run(prompts, max_new_tokens, eos, temperature, generator)
    completions = []
    for row in runner.generated[:, :steps].tolist():
        completions.append(row[: row.index(eos)] if eos in row else row)
    result = Samples(completions, runner.done.tolist())
    if not greedy:
        result.logprobs = runner.logprobs[:, :steps].clone()
        if top_k:
            result.top_ids = runner.top_ids[:, :steps].clone()
            result.top_logprobs = runner.top_logprobs[:, :steps].clone()
    return result


@torch.no_grad()
def compare_engines(model, prompts, max_new_tokens=64):
    """Per engine, the mean and mean absolute difference between the sampler's and the
    learner's (full forward, BF16 autocast) log-probabilities of the sampled tokens."""
    device = model.token_embedding.weight.device
    results = {}
    for engine in ENGINES:
        generator = torch.Generator(device=device).manual_seed(0)
        samples = sample(model, prompts, max_new_tokens, -1, 1.0, generator, engine=engine)
        differences = []
        for row, prompt in enumerate(prompts):
            answer = samples.completions[row]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(torch.tensor([prompt + answer], device=device))[0].float()
            learner = torch.log_softmax(logits[len(prompt) - 1 : -1], -1)
            learner = learner.gather(1, torch.tensor(answer, device=device)[:, None])[:, 0]
            differences.append(samples.logprobs[row, : len(answer)] - learner)
        joined = torch.cat(differences)
        results[engine] = {"mean": joined.mean().item(), "mean_abs": joined.abs().mean().item()}
    return results
