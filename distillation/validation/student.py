"""From-scratch students: packed data stream, CE / CE+KD losses, training and validation.

The architecture is the repository's modern decoder (``training/10m/model.py``:
RMSNorm, RoPE, SwiGLU, tied embeddings). The teacher signal is its top-K
distribution per position plus one residual bucket holding the remaining mass,
so the soft loss is an exact cross-entropy over the partition {top-K, rest}.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
from model_10m.model import GPT, ModelConfig
from torch.utils.checkpoint import checkpoint

PAD, BOS, UNK = 0, 1, 2

SIZES = {
    "small": {"n_layer": 10, "n_head": 8, "n_embd": 512},
    "large": {"n_layer": 18, "n_head": 13, "n_embd": 832},
}


class Vocabulary:
    """Native Qwen ID <-> student ID mapping for one arm.

    ``table`` maps each native ID to one student ID (UNK when dropped). A covering
    vocabulary instead spells dropped native IDs as several kept pieces
    (``pieces``/``offsets``/``lengths``); ``table`` then holds the first piece.
    """

    def __init__(self, table, kept, bos, pieces=None, offsets=None, lengths=None):
        self.table = np.asarray(table, dtype=np.int32)  # native -> (first) student ID
        self.native = np.asarray(
            kept, dtype=np.int64
        )  # student -> native (-1: special)
        self.bos = bos
        self.size = self.native.size
        self.pieces, self.offsets, self.lengths = pieces, offsets, lengths

    @classmethod
    def trimmed(cls, table, kept):
        native = np.concatenate([[-1, -1, -1], kept]).astype(np.int64)
        return cls(table, native, BOS)

    @classmethod
    def covering(cls, kept, pieces, offsets, lengths):
        native = np.concatenate([[-1, -1, -1], kept]).astype(np.int64)
        return cls(pieces[offsets], native, BOS, pieces, offsets, lengths)

    @classmethod
    def full(cls, size, bos):
        return cls(np.arange(size, dtype=np.int32), np.arange(size), bos)

    def encode(self, native_ids):
        """One student ID per native ID (first piece for a covering vocabulary)."""
        return self.table[np.asarray(native_ids, dtype=np.int64)]

    def expand(self, native_ids):
        """Student sequence for native IDs and the index of each piece's native token."""
        native_ids = np.asarray(native_ids, dtype=np.int64)
        if self.pieces is None:
            return self.encode(native_ids), np.arange(native_ids.size)
        counts = self.lengths[native_ids]
        source = np.repeat(np.arange(native_ids.size), counts)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        within = np.arange(source.size) - np.repeat(starts, counts)
        return self.pieces[self.offsets[native_ids][source] + within], source

    def decode_native(self, student_ids):
        native = self.native[np.asarray(student_ids, dtype=np.int64)]
        return [int(i) for i in native if i >= 0]


def pack_shard(shard, vocab):
    """Insert BOS before each document and an ignored label after it.

    With a covering vocabulary, the first piece of each native token keeps the
    teacher's top-K (candidates mapped to their first piece); continuation pieces
    get a one-hot target on the piece itself.
    """
    tokens = np.load(shard / "tokens.npy")
    offsets = np.load(shard / "offsets.npy")
    top_ids = np.load(shard / "topk_ids.npy")
    top_lp = np.load(shard / "topk_logprobs.npy")
    student, source = vocab.expand(tokens)
    first = np.ones(student.size, dtype=bool)
    first[1:] = source[1:] != source[:-1]
    ids = np.full((student.size, top_ids.shape[1]), -1, dtype=np.int32)
    lps = np.full((student.size, top_ids.shape[1]), -np.inf, dtype=top_lp.dtype)
    ids[first] = vocab.encode(top_ids)
    lps[first] = top_lp
    ids[~first, 0] = student[~first]
    lps[~first, 0] = 0
    native = np.where(first, tokens[source], -1)
    boundaries = np.searchsorted(source, offsets)  # document offsets in pieces
    starts, ends = boundaries[:-1], boundaries[1:]
    inputs = np.insert(student, starts, vocab.bos)
    labels = np.insert(student.astype(np.int64), ends, -100)
    ids = np.insert(ids, ends, -1, axis=0)
    lps = np.insert(lps, ends, -np.inf, axis=0)
    native = np.insert(native, ends, -1)
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


def soft_cross_entropy(logits, top_ids, top_lp, lse=None):
    """Cross-entropy against the teacher's {top-K, residual} partition, per row.

    Returns (soft CE, teacher partition entropy); their difference is the KL.
    Duplicate student IDs (several native IDs mapped to one student ID) are
    summed. Only the K gathered logits and the row log-sum-exp are needed, so
    no [rows, vocabulary] temporaries are built.
    """
    if lse is None:
        lse = torch.logsumexp(logits, -1)
    present = top_ids >= 0
    ids = top_ids.clamp(min=0)
    p = torch.where(
        present, top_lp.float().exp(), torch.zeros_like(top_lp, dtype=torch.float32)
    )
    log_q = logits.gather(1, ids) - lse[:, None]
    same = (
        (ids[:, :, None] == ids[:, None, :]) & present[:, :, None] & present[:, None, :]
    )
    earlier = torch.ones_like(same[0], dtype=torch.bool).tril(-1)
    first = present & ~(same & earlier).any(-1)  # first slot of each distinct ID
    grouped = (same * p[:, None, :]).sum(-1)  # teacher mass of each slot's ID
    q_top = torch.where(first, log_q.exp(), torch.zeros_like(log_q)).sum(-1)
    residual = (1 - p.sum(-1)).clamp(min=0)
    log_rest = torch.log1p(-q_top.clamp(max=1 - 1e-6))
    soft = -(p * log_q).sum(-1) - residual * log_rest
    entropy = (
        -torch.where(first, grouped * grouped.clamp(min=1e-30).log(), 0.0).sum(-1)
        - residual * residual.clamp(min=1e-30).log()
    )
    return soft, entropy


def chunk_losses(model, features, labels, top_ids, top_lp):
    logits = model.lm_head(features).float()
    valid = labels != -100
    lse = torch.logsumexp(logits, -1)
    target = logits.gather(1, labels.clamp(min=0)[:, None])[:, 0]
    ce = torch.where(valid, lse - target, 0.0).sum()
    soft, entropy = soft_cross_entropy(logits, top_ids, top_lp, lse)
    return torch.stack(
        [
            ce,
            torch.where(valid, soft, 0.0).sum(),
            torch.where(valid, entropy, 0.0).sum(),
        ]
    )


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
