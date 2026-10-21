"""Shared-tokenizer data, embedding transfer and exact full-vocabulary KL."""

import zlib

from distillation.data import CodeWindows


class SharedCodeWindows(CodeWindows):
    def __init__(self, *args, frozen_tokenizer, eos_id, **kwargs):
        self.frozen_tokenizer = frozen_tokenizer
        self.eos_id = eos_id
        super().__init__(*args, **kwargs)

    def _windows(self):
        rows = self.db.execute(
            """SELECT hash,parent,text,loss_tokens FROM documents
            WHERE source='code' AND split=? AND conflict=0 AND hash>=? ORDER BY hash""",
            (self.split, self.state["hash"]),
        )
        for key, parent, raw, expected in rows:
            if self.split == "train" and (
                key in self.excluded_hashes or parent in self.excluded_parents
            ):
                continue
            text = zlib.decompress(raw).decode("utf-8")
            if (
                len(self.frozen_tokenizer.encode(text, add_special_tokens=False).ids)
                + 1
                != expected
            ):
                raise ValueError("Original corpus token count mismatch")
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            tokens = [self.eos_id, *ids, self.eos_id]
            start = self.state["offset"] if key == self.state["hash"] else 0
            for offset in range(start, len(tokens) - 1, self.context):
                y = tokens[offset + 1 : offset + self.context + 1]
                x = tokens[offset : offset + len(y)]
                self.state = {"hash": key, "offset": offset + len(y)}
                yield x, y


def transfer_embeddings(old_weight, old_tokenizer, new_tokenizer, new_weight, eos_id):
    """Copy byte-identical tokens; initialize other pieces from old-token means."""
    import torch

    from distillation.alignment import vocabulary

    old_pieces, _ = vocabulary(old_tokenizer.to_str(), old_weight.shape[0])
    new_pieces, _ = vocabulary(
        new_tokenizer.backend_tokenizer.to_str(), new_weight.shape[0]
    )
    exact = {piece: i for i, piece in enumerate(old_pieces) if piece is not None}
    byte_ids = {piece[0]: i for piece, i in exact.items() if len(piece) == 1}
    stats = {"exact": 0, "composed": 0, "random_special_or_unused": 0}
    with torch.no_grad():
        for i, piece in enumerate(new_pieces):
            if piece is None:
                stats["random_special_or_unused"] += 1
                continue
            if piece in exact:
                new_weight[i].copy_(old_weight[exact[piece]])
                stats["exact"] += 1
            else:
                try:
                    ids = old_tokenizer.encode(
                        piece.decode("utf-8"), add_special_tokens=False
                    ).ids
                    if b"".join(old_pieces[t] or b"" for t in ids) != piece:
                        raise ValueError("Non-exact encoding")
                except (UnicodeDecodeError, ValueError):
                    ids = [byte_ids[b] for b in piece]
                new_weight[i].copy_(old_weight[ids].mean(0))
                stats["composed"] += 1
        new_weight[eos_id].copy_(old_weight[2])
    return stats


def token_losses(student_logits, teacher_logits, labels):
    """Return summed CE and KL(teacher || student), with padding excluded."""
    import torch
    import torch.nn.functional as F

    mask = labels.flatten() != -100
    if not mask.any():
        raise ValueError("No loss-bearing targets")
    s = student_logits.flatten(0, 1)[mask].float()
    t = teacher_logits.flatten(0, 1)[mask].detach().float()
    logp = F.log_softmax(t, dim=-1)
    logq = F.log_softmax(s, dim=-1)
    ce = F.nll_loss(logq, labels.flatten()[mask], reduction="sum")
    kl = torch.sum(logp.exp() * (logp - logq))
    return ce, kl


def take_batch(data, max_windows, remaining):
    """Consume at most remaining targets, preserving an exact partial cursor."""
    batch = []
    for _ in range(max_windows):
        if remaining == 0:
            break
        try:
            x, y = data.next()
        except StopIteration:
            break
        if len(y) > remaining:
            excess = len(y) - remaining
            data.state["offset"] -= excess
            x, y = x[:remaining], y[:remaining]
        batch.append((x, y))
        remaining -= len(y)
    return batch
