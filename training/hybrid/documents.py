"""Document boundaries inside packed training windows.

Loaders pack documents back to back, each ending in EOS, so a new document starts
at every position whose previous input token is EOS. Ids restart at 0 in each row;
padding after a final EOS forms its own document, and its targets are ignored.
"""

import torch

IGNORE_INDEX = -100


def document_ids(input_ids, eos_id):
    """[B, T] long ids; position t starts a new document when input t-1 is EOS."""
    starts = torch.zeros_like(input_ids, dtype=torch.long)
    starts[:, 1:] = (input_ids[:, :-1] == eos_id).long()
    return starts.cumsum(1)


def document_positions(documents):
    """Position of each token within its own document, [B, T]."""
    index = torch.arange(documents.size(1), device=documents.device).expand_as(documents)
    start = torch.zeros_like(documents, dtype=torch.bool)
    start[:, 0] = True
    start[:, 1:] = documents[:, 1:] != documents[:, :-1]
    return index - torch.where(start, index, 0).cummax(1).values


def cu_seqlens(documents):
    """int32 offsets of every document in the flattened [B * T] batch.

    Row starts are always boundaries, so varlen kernels never join rows.
    """
    batch, length = documents.shape
    boundary = torch.ones_like(documents, dtype=torch.bool)
    boundary[:, 1:] = documents[:, 1:] != documents[:, :-1]
    starts = boundary.flatten().nonzero().squeeze(1)
    end = torch.tensor([batch * length], device=documents.device)
    return torch.cat((starts, end)).to(torch.int32)


def mtp_labels(inputs, labels, eos_id):
    """Targets x_(t+2) for the MTP module at each position t, [B, T].

    The target is labels[t + 1]. It is ignored when x_t or x_(t+1) is EOS (it
    would belong to the next document) and at the last position, whose x_(t+2)
    lies outside the window.
    """
    targets = torch.full_like(labels, IGNORE_INDEX)
    within = (inputs[:, :-1] != eos_id) & (inputs[:, 1:] != eos_id)
    targets[:, :-1] = labels[:, 1:].masked_fill(~within, IGNORE_INDEX)
    return targets
