"""Document boundaries inside packed training windows.

Loaders pack documents back to back, each ending in EOS, so a new document starts
at every position whose previous input token is EOS. Ids restart at 0 in each row;
padding after a final EOS forms its own document, and its targets are ignored.
"""

import torch


def document_ids(input_ids, eos_id):
    """[B, T] long ids; position t starts a new document when input t-1 is EOS."""
    starts = torch.zeros_like(input_ids, dtype=torch.long)
    starts[:, 1:] = (input_ids[:, :-1] == eos_id).long()
    return starts.cumsum(1)


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
