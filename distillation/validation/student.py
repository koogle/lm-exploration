"""From-scratch students: packed data stream, CE / CE+KD losses, training and decoding.

The architecture is the repository's modern decoder (``training/10m/model.py``:
RMSNorm, RoPE, SwiGLU, tied embeddings). The teacher signal is its top-K
distribution per position plus one residual bucket holding the remaining mass,
so the soft loss is an exact cross-entropy over the partition {top-K, rest}.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from model_10m.model import GPT, ModelConfig
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

PAD, BOS, UNK = 0, 1, 2

SIZES = {
    "small": {"n_layer": 10, "n_head": 8, "n_embd": 512},
    "large": {"n_layer": 18, "n_head": 13, "n_embd": 832},
}


class Vocabulary:
    """Native Qwen ID <-> student ID mapping for one arm."""

    def __init__(self, table, kept, bos):
        self.table = np.asarray(table, dtype=np.int32)  # native -> student
        self.native = np.asarray(
            kept, dtype=np.int64
        )  # student -> native (-1: special)
        self.bos = bos
        self.size = self.native.size

    @classmethod
    def trimmed(cls, table, kept):
        native = np.concatenate([[-1, -1, -1], kept]).astype(np.int64)
        return cls(table, native, BOS)

    @classmethod
    def full(cls, size, bos):
        return cls(np.arange(size, dtype=np.int32), np.arange(size), bos)

    def encode(self, native_ids):
        return self.table[np.asarray(native_ids, dtype=np.int64)]

    def decode_native(self, student_ids):
        native = self.native[np.asarray(student_ids, dtype=np.int64)]
        return [int(i) for i in native if i >= 0]


def pack_shard(shard, vocab):
    """Insert BOS before each document and an ignored label after it."""
    tokens = np.load(shard / "tokens.npy")
    offsets = np.load(shard / "offsets.npy")
    top_ids = np.load(shard / "topk_ids.npy")
    top_lp = np.load(shard / "topk_logprobs.npy")
    starts, ends = offsets[:-1], offsets[1:]
    student = vocab.encode(tokens)
    inputs = np.insert(student, starts, vocab.bos)
    labels = np.insert(student.astype(np.int64), ends, -100)
    mapped = vocab.encode(top_ids)
    ids = np.insert(mapped, ends, -1, axis=0)
    lps = np.insert(top_lp, ends, -np.inf, axis=0)
    native = np.insert(tokens, ends, -1)
    return inputs, labels, ids, lps, native


class Stream:
    """Fixed-order windows of ``window`` positions across shards, without repeats."""

    def __init__(self, shards, vocab, window):
        self.shards, self.vocab, self.window = list(shards), vocab, window

    def __iter__(self):
        carry = None
        for shard in self.shards:
            arrays = pack_shard(Path(shard), self.vocab)
            if carry is not None:
                arrays = tuple(np.concatenate([c, a]) for c, a in zip(carry, arrays))
            usable = arrays[0].size // self.window * self.window
            for start in range(0, usable, self.window):
                yield tuple(a[start : start + self.window] for a in arrays)
            carry = tuple(a[usable:] for a in arrays)


def batches(stream, batch_windows):
    group = []
    for window in stream:
        group.append(window)
        if len(group) == batch_windows:
            yield tuple(np.stack(parts) for parts in zip(*group))
            group = []


def soft_cross_entropy(logits, top_ids, top_lp):
    """Cross-entropy against the teacher's {top-K, residual} partition, per row.

    Returns (soft CE, teacher partition entropy); their difference is the KL.
    Duplicate student IDs (several native IDs mapped to UNK) are summed.
    """
    present = top_ids >= 0
    p = torch.where(
        present, top_lp.float().exp(), torch.zeros_like(top_lp, dtype=torch.float32)
    )
    dense = torch.zeros_like(logits).scatter_add_(1, top_ids.clamp(min=0), p)
    residual = (1 - dense.sum(-1)).clamp(min=0)
    lse = torch.logsumexp(logits, -1)
    in_top = dense > 0
    rest = torch.logsumexp(logits.masked_fill(in_top, -math.inf), -1)
    soft = -(dense * (logits - lse[:, None])).sum(-1) - residual * (rest - lse)
    entropy = (
        -(dense * dense.clamp(min=1e-30).log()).sum(-1)
        - residual * residual.clamp(min=1e-30).log()
    )
    return soft, entropy


def chunk_losses(model, features, labels, top_ids, top_lp):
    logits = model.lm_head(features).float()
    valid = labels != -100
    ce = F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
    soft, entropy = soft_cross_entropy(logits[valid], top_ids[valid], top_lp[valid])
    return torch.stack([ce, soft.sum(), entropy.sum()])


def losses(model, inputs, labels, top_ids, top_lp):
    """Summed (CE, soft CE, teacher entropy) with vocabulary logits recomputed per chunk."""
    # Several dense [chunk, vocabulary] FP32 tensors are live per chunk.
    chunk = 4096 if model.lm_head.out_features <= 65536 else 1024
    hidden = model.final_norm(model.blocks(model.token_embedding(inputs))).flatten(0, 1)
    parts = zip(
        hidden.split(chunk),
        labels.flatten().split(chunk),
        top_ids.flatten(0, 1).split(chunk),
        top_lp.flatten(0, 1).split(chunk),
    )
    return sum(checkpoint(chunk_losses, model, *p, use_reentrant=False) for p in parts)


def to_device(batch, device):
    inputs, labels, ids, lps = batch[:4]
    return (
        torch.from_numpy(inputs.astype(np.int64)).to(device, non_blocking=True),
        torch.from_numpy(labels).to(device, non_blocking=True),
        torch.from_numpy(ids.astype(np.int64)).to(device, non_blocking=True),
        torch.from_numpy(lps.astype(np.float32)).to(device, non_blocking=True),
    )


def learning_rate(step, steps, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(settings, train_shards, vocab, out_dir, log, deadline=None, device="cuda"):
    """Train one student for ``settings['positions']`` input positions; no repeats."""
    torch.manual_seed(settings["seed"])
    config = ModelConfig(
        vocab_size=vocab.size, block_size=settings["window"], **SIZES[settings["size"]]
    )
    model = GPT(config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    embedding = model.token_embedding.weight.numel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=device == "cuda",
    )
    per_step = settings["batch_windows"] * settings["window"]
    steps = settings["positions"] // per_step
    micro = settings["micro_windows"]
    accumulation = settings["batch_windows"] // micro
    warmup = max(20, steps // 50)
    weight = settings["kd_weight"]
    log(
        {
            "event": "start",
            "parameters": parameters,
            "embedding": embedding,
            "steps": steps,
        }
    )
    stream = batches(Stream(train_shards, vocab, settings["window"]), micro)
    totals = {"labels": 0, "ce": 0.0, "soft": 0.0, "entropy": 0.0}
    started = time.time()
    model.train()
    for step in range(steps):
        if deadline and time.time() > deadline:
            raise RuntimeError("campaign deadline reached during training")
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps, settings["lr"], warmup)
        sums = torch.zeros(3, device=device)
        try:
            micro_batches = [
                to_device(next(stream), device) for _ in range(accumulation)
            ]
        except StopIteration:
            raise ValueError(
                f"training data exhausted at step {step} of {steps}"
            ) from None
        count = sum(int((b[1] != -100).sum()) for b in micro_batches)
        for batch in micro_batches:
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                part = losses(model, *batch)
            loss = ((1 - weight) * part[0] + weight * part[1]) / count
            loss.backward()
            sums += part.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ce, soft, entropy = (sums / count).tolist()
        totals["labels"] += count
        if step % settings["log_every"] == 0 or step == steps - 1:
            elapsed = time.time() - started
            log(
                {
                    "event": "step",
                    "step": step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "ce": ce,
                    "kl": soft - entropy,
                    "grad_norm": float(norm),
                    "labels": totals["labels"],
                    "positions": (step + 1) * per_step,
                    "positions_per_second": (step + 1) * per_step / elapsed,
                }
            )
    torch.save(
        {"model": model.state_dict(), "config": config.__dict__, "settings": settings},
        Path(out_dir) / "model.pt",
    )
    return model, {
        "parameters": parameters,
        "embedding_parameters": embedding,
        "backbone_parameters": parameters - embedding,
        "steps": steps,
        "positions": steps * per_step,
        "labels": totals["labels"],
        "train_seconds": time.time() - started,
    }


@torch.no_grad()
def validate(
    model,
    shards,
    vocab,
    byte_lengths,
    positions,
    window=2048,
    device="cuda",
    loss_fn=None,
):
    """Held-out CE and KL per token, and CE per UTF-8 byte of the native targets."""
    loss_fn = loss_fn or losses
    model.eval()
    sums = torch.zeros(3, dtype=torch.float64, device=device)
    labels_seen, bytes_seen, seen = 0, 0, 0
    for batch in batches(Stream(shards, vocab, window), 8):
        inputs, labels, ids, lps = to_device(batch, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            sums += loss_fn(model, inputs, labels, ids, lps).double()
        labels_seen += int((labels != -100).sum())
        native = batch[4]
        bytes_seen += int(byte_lengths[native[native >= 0]].sum())
        seen += inputs.numel()
        if seen >= positions:
            break
    ce, soft, entropy = (sums / labels_seen).tolist()
    return {
        "labels": labels_seen,
        "bytes": bytes_seen,
        "ce_per_token": ce,
        "kl_per_token": soft - entropy,
        "ce_per_byte": float(sums[0]) / bytes_seen,
    }


def rope(x, positions):
    """RoPE for explicit per-row positions; matches ``model_10m.model.rotary``."""
    width = x.size(-1)
    frequencies = 10000.0 ** (
        -torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width
    )
    angles = positions.float()[:, None, :, None] * frequencies
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


def cached_forward(model, tokens, positions, mask, cache, start):
    """Run ``tokens`` at cache slots ``start:``; ``mask`` is [B, 1, Tq, Tk] bool."""
    x = model.token_embedding(tokens)
    for layer, block in enumerate(model.blocks):
        attention = block.attention
        batch, length, width = x.shape
        heads = attention.n_head
        q, k, v = attention.qkv(block.attention_norm(x)).chunk(3, -1)
        q, k, v = [
            t.view(batch, length, heads, width // heads).transpose(1, 2)
            for t in (q, k, v)
        ]
        q, k = rope(q, positions), rope(k, positions)
        keys, values = cache[layer]
        keys[:, :, start : start + length] = k
        values[:, :, start : start + length] = v
        end = start + length
        attended = F.scaled_dot_product_attention(
            q, keys[:, :, :end], values[:, :, :end], attn_mask=mask
        )
        x = x + attention.projection(
            attended.transpose(1, 2).reshape(batch, length, width)
        )
        x = x + block.mlp(block.mlp_norm(x))
    return model.lm_head(model.final_norm(x[:, -1]))


@torch.no_grad()
def generate(model, prompts, max_new, eos, finished, device="cuda"):
    """Greedy decoding of left-padded prompts (lists of student IDs).

    ``finished(ids)`` is checked every 16 steps on each row's generated IDs.
    """
    model.eval()
    batch = len(prompts)
    longest = max(map(len, prompts))
    total = longest + max_new
    pad = torch.tensor([longest - len(p) for p in prompts], device=device)
    tokens = torch.full((batch, longest), PAD, dtype=torch.long, device=device)
    for row, prompt in enumerate(prompts):
        tokens[row, longest - len(prompt) :] = torch.tensor(prompt, device=device)
    slots = torch.arange(total, device=device)
    valid = slots[None, :] >= pad[:, None]
    positions = (slots[None, :] - pad[:, None]).clamp(min=0)
    config = model.config
    shape = (batch, config.n_head, total, config.n_embd // config.n_head)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    cache = [
        (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )
        for _ in model.blocks
    ]
    causal = slots[:longest, None] >= slots[None, :longest]
    mask = (causal[None] & valid[:, None, :longest]) | torch.eye(
        longest, dtype=torch.bool, device=device
    )
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        logits = cached_forward(
            model, tokens, positions[:, :longest], mask[:, None], cache, 0
        )
    generated = torch.zeros((batch, max_new), dtype=torch.long, device=device)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    lengths = [max_new] * batch
    for step in range(max_new):
        next_token = logits.float().argmax(-1)
        generated[:, step] = next_token
        stopped = (~done) & (next_token == eos)
        if (step + 1) % 16 == 0:
            rows = generated[:, : step + 1].tolist()
            for row in range(batch):
                if not done[row] and finished(rows[row]):
                    stopped[row] = True
        for row in stopped.nonzero().flatten().tolist():
            lengths[row] = step + 1
        done |= stopped
        if bool(done.all()):
            break
        slot = longest + step
        step_mask = valid[:, None, None, : slot + 1]
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = cached_forward(
                model,
                next_token[:, None],
                positions[:, slot : slot + 1],
                step_mask,
                cache,
                slot,
            )
    return [row[:n] for row, n in zip(generated.tolist(), lengths)]


def load_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
