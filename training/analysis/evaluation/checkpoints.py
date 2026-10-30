"""The model interface the benchmarks score (`scoring.py`), for a checkpoint of any stage
loaded by `trainer.checkpoints.load`."""

from types import SimpleNamespace

# Padding sits after every scored position; PAD is ID 0 in our tokenizers.
PAD_TOKEN_ID = 0


def scorer(checkpoint):
    """The model interface `tasks.py` expects, running in BF16 autocast on CUDA."""
    import torch

    def forward(tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return checkpoint.model(tokens)

    tokenizer = checkpoint.tokenizer
    return SimpleNamespace(
        forward=forward,
        network=checkpoint.model,
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        bos_token_id=checkpoint.special["bos"],
        eos_token_id=checkpoint.special["eos"],
        pad_token_id=PAD_TOKEN_ID,
        block_size=checkpoint.config["block_size"],
        device="cuda",
    )
