"""Block-buffered sparse probability targets; teacher token IDs remain native."""

import torch


def compact_targets(logits, tokens, k=64):
    """Top-k probabilities retain full-vocabulary normalization and residual mass."""
    normalizer = logits.logsumexp(-1)
    values, ids = logits.topk(k, dim=-1)
    logp = values - normalizer[:, None]
    selected = logits.gather(-1, tokens[:, None]).squeeze(-1) - normalizer
    return {
        "topk_ids": ids.cpu().tolist(),
        "topk_logprobs": logp.cpu().tolist(),
        "selected_logprobs": selected.cpu().tolist(),
        "tail_mass": (1 - logp.exp().sum(-1)).clamp_min(0).cpu().tolist(),
        "logsumexp": normalizer.cpu().tolist(),
    }


@torch.inference_mode()
def trajectory(
    decoder,
    ids,
    max_tokens=1024,
    eos_ids=(),
    seed=0,
    block_tokens=64,
    temperature=0.6,
    sampling_top_k=20,
    top_p=0.95,
):
    if not ids or len(ids) + max_tokens > decoder.max_seq_len or block_tokens < 1:
        raise ValueError("Invalid context or block size")
    decoder.reset()
    for token in ids:
        logits = decoder.step(token)
    generator = torch.Generator(device=decoder.head.device).manual_seed(seed)
    outputs, payload = (
        [],
        {
            k: []
            for k in [
                "topk_ids",
                "topk_logprobs",
                "selected_logprobs",
                "tail_mass",
                "logsumexp",
            ]
        },
    )
    finish = "length"
    while len(outputs) < max_tokens:
        count = min(block_tokens, max_tokens - len(outputs))
        rows = torch.empty(
            (count, decoder.head.shape[0]),
            device=decoder.head.device,
            dtype=torch.float32,
        )
        tokens = torch.empty(count, device=decoder.head.device, dtype=torch.int64)
        for i in range(count):
            rows[i].copy_(logits)
            if temperature == 0:
                torch.argmax(logits, dim=0, out=tokens[i])
            else:
                values, candidates = logits.topk(sampling_top_k)
                scaled = values / temperature
                probs = scaled.softmax(-1)
                scaled = scaled.masked_fill(
                    probs.cumsum(-1) - probs > top_p, -torch.inf
                )
                noise = (
                    torch.empty_like(scaled).exponential_(generator=generator).log_()
                )
                tokens[i].copy_(candidates[(scaled - noise).argmax()])
            if i + 1 < count:
                logits = decoder.step_device(tokens[i])
        batch = tokens.cpu().tolist()
        stop = next((i + 1 for i, t in enumerate(batch) if t in eos_ids), count)
        compact = compact_targets(rows[:stop], tokens[:stop])
        for key, values in payload.items():
            values.extend(compact[key])
        outputs.extend(batch[:stop])
        if batch[stop - 1] in eos_ids:
            finish = "stop"
            break
        if len(outputs) < max_tokens:
            logits = decoder.step_device(tokens[-1])
    return {"token_ids": outputs, "targets": payload, "finish_reason": finish}
