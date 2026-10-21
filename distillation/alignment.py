"""Byte-level BPE alignment without lossy per-token Unicode decoding.

The objective matches the first-byte marginal, not the full token distribution.
Special tokens are one separate bucket; unused vocabulary rows join that bucket.
"""

import json


def byte_decoder():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes = values.copy()
    for value in range(256):
        if value not in values:
            values.append(value)
            codes.append(256 + len(codes) - 188)
    return dict(zip(map(chr, codes), values))


def vocabulary(tokenizer_json, size):
    spec = json.loads(tokenizer_json)
    if spec["model"]["type"] != "BPE" or spec["decoder"]["type"] != "ByteLevel":
        raise ValueError("Only verified ByteLevel BPE tokenizers are supported")
    decoder = byte_decoder()
    specials = {t["id"] for t in spec["added_tokens"] if t["special"]}
    pieces = [None] * size
    for spelling, index in spec["model"]["vocab"].items():
        if index not in specials:
            pieces[index] = bytes(decoder[c] for c in spelling)
    buckets = [piece[0] if piece else 256 for piece in pieces]
    return pieces, buckets


def shared_boundaries(student_ids, teacher_ids, student_pieces, teacher_pieces):
    """Return positions whose inclusive input prefixes contain identical bytes.

    No empty-prefix match: teacher has no BOS. Special student BOS contributes
    zero bytes. Compare reconstruction before aligning to prevent silent drift.
    """

    def ends(ids, pieces):
        raw = bytearray()
        positions = {}
        for i, token in enumerate(ids):
            piece = pieces[token]
            if piece is not None:
                raw.extend(piece)
                positions[len(raw)] = i
        return bytes(raw), positions

    student_raw, student_ends = ends(student_ids, student_pieces)
    teacher_raw, teacher_ends = ends(teacher_ids, teacher_pieces)
    if student_raw != teacher_raw:
        raise ValueError("Teacher/student text bytes differ")
    offsets = sorted(student_ends.keys() & teacher_ends.keys())
    return [student_ends[x] for x in offsets], [teacher_ends[x] for x in offsets]


def byte_probabilities(logits, buckets):
    """Full softmax followed by exact first-byte/special marginalization."""
    import torch

    probabilities = logits.float().softmax(-1)
    indices = torch.as_tensor(buckets, dtype=torch.long, device=logits.device)
    indices = indices.expand_as(probabilities)
    return probabilities.new_zeros((*probabilities.shape[:-1], 257)).scatter_add(
        -1, indices, probabilities
    )


def marginal_kl(student_logits, teacher_probs, buckets):
    import torch.nn.functional as F

    student = byte_probabilities(student_logits, buckets)
    return F.kl_div(
        student.clamp_min(1e-30).log(), teacher_probs, reduction="batchmean"
    )
