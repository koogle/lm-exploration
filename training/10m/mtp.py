"""Experimental one-step sequential MTP; not wired into the distributed trainer."""

import torch
from torch import nn
from torch.nn import functional as F

from .model import GPT, Block, RMSNorm


class SequentialMTP(nn.Module):
    """Combine h_t and the known x_(t+1) to predict x_(t+2)."""

    def __init__(self, config):
        super().__init__()
        width = config.n_embd
        self.hidden_norm = RMSNorm(width)
        self.embedding_norm = RMSNorm(width)
        self.fuse = nn.Linear(2 * width, width, bias=False)
        self.block = Block(config)
        self.output_norm = RMSNorm(width)

    def forward(self, hidden, next_embeddings):
        fused = self.fuse(
            torch.cat((self.hidden_norm(hidden), self.embedding_norm(next_embeddings)), dim=-1)
        )
        return self.output_norm(self.block(fused))


class GPTWithMTP(GPT):
    """Ordinary GPT inference plus a separate, teacher-forced training objective.

    Inherited forward() remains next-token-only. Use training_losses() explicitly
    for MTP. Each batch row must contain ONE document, optionally right-padded;
    packed multi-document rows are not supported by this prototype.
    """

    def __init__(self, config):
        super().__init__(config)
        self.mtp = SequentialMTP(config)
        self.mtp.apply(self.initialize)

    def training_losses(self, sequence, lengths):
        """Return differentiable loss sums and separate valid-target counts.

        sequence: [B, T+1] token IDs, including the last target token. Padding
        must use an in-vocabulary ID. lengths: [B] actual document lengths,
        including the final target, excluding padding. Require at least three
        columns (individual documents may be shorter). No Qwen logits needed.

        Single-device objective: ntp_sum / ntp_count + weight * mtp_sum / mtp_count
        (clamp each denominator to >=1). Under DDP, aggregate counts globally
        across the whole accumulation window before scaling each loss sum.
        """
        if sequence.ndim != 2 or not 3 <= sequence.size(1) <= self.config.block_size + 1:
            raise ValueError("sequence must have shape [B, T+1], with 2 <= T <= block_size")
        if lengths.shape != (sequence.size(0),) or lengths.dtype != torch.long:
            raise ValueError("lengths must be a long tensor with shape [B]")
        if lengths.device != sequence.device:
            raise ValueError("lengths and sequence must be on the same device")
        if ((lengths < 1) | (lengths > sequence.size(1))).any():
            raise ValueError("document lengths must be within the supplied sequence")

        tokens = sequence[:, :-1]
        hidden = self.blocks(self.token_embedding(tokens))
        next_logits = self.lm_head(self.final_norm(hidden))

        # h_t sees only x_<=t. The auxiliary branch additionally sees x_(t+1),
        # never its target x_(t+2). Attention within the branch remains causal.
        future_hidden = self.mtp(hidden[:, :-1], self.token_embedding(tokens[:, 1:]))
        future_logits = self.lm_head(future_hidden)  # same tied vocabulary table

        def loss_sum(logits, targets, offset):
            positions = torch.arange(targets.size(1), device=sequence.device) + offset
            valid = positions[None, :] < lengths[:, None]
            targets = targets.masked_fill(~valid, -100)
            loss = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="sum"
            )
            return loss, valid.sum()

        ntp_sum, ntp_count = loss_sum(next_logits, sequence[:, 1:], 1)
        mtp_sum, mtp_count = loss_sum(future_logits, sequence[:, 2:], 2)
        return {
            "ntp_sum": ntp_sum,
            "ntp_count": ntp_count,
            "mtp_sum": mtp_sum,
            "mtp_count": mtp_count,
        }
