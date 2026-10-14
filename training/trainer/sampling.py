"""Sampling shared by training previews and the generation CLI."""

import torch


@torch.no_grad()
def sample_tokens(model, tokenizer, prompt, max_new_tokens, seed):
    # Yields each new token id as soon as it is sampled, so callers can stream output.
    context = tokenizer.encode(prompt) or [0]
    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            tokens = torch.tensor([context[-model.config.block_size :]], device=device)
            logits = model(tokens)
            probabilities = logits[0, -1].softmax(dim=-1).cpu()
            token = torch.multinomial(probabilities, 1, generator=generator).item()
            context.append(token)
            yield token
    finally:
        model.train(was_training)


def sample(model, tokenizer, prompt, max_new_tokens, seed):
    generated = list(sample_tokens(model, tokenizer, prompt, max_new_tokens, seed))
    return prompt + tokenizer.decode(generated)
