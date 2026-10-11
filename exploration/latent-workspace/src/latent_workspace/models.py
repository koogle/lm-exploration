"""Trainable heads; frozen Qwen embedding and output weights stay external."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class Workspace(nn.Module):
    def __init__(self, hidden, cfg):
        super().__init__()
        self.project = nn.Linear(hidden, cfg.width)
        self.depth = nn.Parameter(torch.randn(len(cfg.layers), cfg.width) * 0.02)
        self.position = nn.Parameter(torch.randn(cfg.memory_tokens, cfg.width) * 0.02)
        self.slots = nn.Parameter(torch.randn(cfg.slots, cfg.width) * 0.02)
        self.attend = nn.MultiheadAttention(
            cfg.width, cfg.heads, batch_first=True, add_zero_attn=cfg.null_attention
        )
        self.layer_dropout = cfg.layer_dropout
        self.dropout_seed = cfg.seed
        self.dropout_step = 0
        self.null_attention = cfg.null_attention
        self.norm = nn.LayerNorm(cfg.width)
        self.ff = nn.Sequential(
            nn.Linear(cfg.width, cfg.width * 4), nn.GELU(), nn.Linear(cfg.width * 4, cfg.width)
        )

    def forward(self, memory):
        # [batch, depth, context tail, hidden], all strictly before the target.
        b, depths, positions, _ = memory.shape
        x = self.project(memory.float())
        x = x + self.depth[:depths][None, :, None] + self.position[:positions][None, None]
        x = x.reshape(b, depths * positions, -1)
        slots = self.slots.unsqueeze(0).expand(b, -1, -1)
        padding = None
        if self.training and self.layer_dropout:
            # Independent, step-indexed RNG preserves corruption/sampling and resumes.
            generator = torch.Generator().manual_seed(self.dropout_seed + 9191 + self.dropout_step)
            dropped = torch.rand(b, depths, generator=generator) < self.layer_dropout
            if not self.null_attention:
                dropped[dropped.all(dim=1), -1] = False
            padding = dropped.repeat_interleave(positions, dim=1).to(x.device)
        # add_zero_attn appends zero K/V *after* projection, with a valid mask entry.
        slots = self.norm(
            slots + self.attend(slots, x, x, key_padding_mask=padding, need_weights=False)[0]
        )
        return slots + self.ff(slots)


class BlockHead(nn.Module):
    def __init__(self, hidden, cfg, variant):
        super().__init__()
        self.cfg, self.variant = cfg, variant
        self.kind, self.condition = variant.split(":")
        self.position = nn.Parameter(torch.randn(cfg.block_size, cfg.width) * 0.02)
        self.out_norm = nn.Identity() if cfg.residual_readout else nn.LayerNorm(hidden)
        self.out = (
            nn.ModuleList([nn.Linear(cfg.width, hidden) for _ in range(cfg.block_size)])
            if cfg.position_readout
            else nn.Linear(cfg.width, hidden)
        )
        if cfg.residual_readout:
            for projection in self.out if cfg.position_readout else [self.out]:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
        if self.kind == "probe":
            self.probe = nn.Sequential(
                nn.Linear(hidden, cfg.width), nn.GELU(), nn.Linear(cfg.width, cfg.width), nn.GELU()
            )
            return
        self.workspace = Workspace(hidden, cfg)
        self.token_project = nn.Linear(hidden, cfg.width)
        self.token_norm = (
            nn.RMSNorm(hidden, eps=1e-6, elementwise_affine=False)
            if cfg.normalize_token_inputs
            else nn.Identity()
        )
        self.mask = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.time = nn.Sequential(
            nn.Linear(1, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width)
        )
        layer = nn.TransformerDecoderLayer(
            cfg.width, cfg.heads, cfg.width * 4, dropout=0, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, cfg.decoder_layers)
        # Identical capacity across conditioning ablations.

    def forward(self, memory, token_embeddings=None, masked=None, noise=None):
        b = memory.shape[0]
        base = memory[:, len(self.cfg.layers) - 1, -1].float()[:, None]
        if self.kind == "probe":
            index = self.cfg.layers.index(int(self.condition.removeprefix("attention")))
            if self.condition.startswith("attention"):
                index += len(self.cfg.layers)
            x = self.probe(memory[:, index, -1].float())[:, None] + self.position
        else:
            depth = len(self.cfg.layers)
            selected = [
                self.cfg.layers.index(i) for i in (self.cfg.selected_layers or self.cfg.layers)
            ]
            if self.condition == "final":
                memory = memory[:, depth - 1 : depth]
            elif self.condition == "all":
                memory = memory[:, :depth]
            elif self.condition == "attention_all":
                memory = memory[:, depth : 2 * depth]
            elif self.condition == "best":
                index = selected[0]
                memory = memory[:, index : index + 1]
            elif self.condition == "multi":
                memory = memory[:, selected]
            elif self.condition == "attention_multi":
                memory = memory[:, [i + depth for i in selected]]
            elif self.condition == "jspace":
                memory = memory[:, [i + 2 * depth for i in selected]]
            elif self.condition.startswith("layer"):
                index = self.cfg.layers.index(int(self.condition[5:]))
                memory = memory[:, index : index + 1]
            elif self.condition.startswith("attention"):
                index = depth + self.cfg.layers.index(int(self.condition[9:]))
                memory = memory[:, index : index + 1]
            slots = self.workspace(memory)
            x = self.position.unsqueeze(0).expand(b, -1, -1)
            if self.kind != "parallel":
                if token_embeddings is None or masked is None or noise is None:
                    raise ValueError("diffusion needs corrupted tokens, mask, and noise level")
                visible = self.token_project(self.token_norm(token_embeddings.float()))
                x = x + torch.where(masked[..., None], self.mask, visible)
                x = x + self.time(noise[:, None].float())[:, None]
            for _ in range(self.cfg.refinement_rounds):
                x = self.decoder(x, slots)
        projected = (
            torch.stack([projection(x[:, i]) for i, projection in enumerate(self.out)], dim=1)
            if self.cfg.position_readout
            else self.out(x)
        )
        delta = self.out_norm(projected)
        return base + delta if self.cfg.residual_readout else delta


def corrupt(target, generator=None):
    """Uniform nonempty mask count; loss averages masked positions per example.

    Deliberately a masked-denoising objective, not a BD3-LM ELBO reproduction.
    """
    b, length = target.shape
    count = torch.randint(1, length + 1, (b,), device=target.device, generator=generator)
    order = torch.rand(target.shape, device=target.device, generator=generator).argsort(-1)
    mask = order.argsort(-1) < count[:, None]
    return mask, count.float() / length


def loss_and_logits(head, backbone, memory, target, *, all_masked=False):
    if head.kind in ("parallel", "probe"):
        mask = torch.ones_like(target, dtype=torch.bool)
        features = head(memory)
    else:
        if all_masked:
            mask = torch.ones_like(target, dtype=torch.bool)
            noise = torch.ones(target.shape[0], device=target.device)
        else:
            mask, noise = corrupt(target)
        # Masked target IDs cannot leak through embeddings, even accidentally.
        ids = target.masked_fill(mask, 0)
        features = head(memory, backbone.embed(ids), mask, noise)
    logits = backbone.project(features)
    ce = F.cross_entropy(logits.float().transpose(1, 2), target, reduction="none")
    loss = ((ce * mask).sum(-1) / mask.sum(-1)).mean()
    return loss, logits, mask


@torch.no_grad()
def sample_block(
    head, backbone, memory, temperature=0.0, steps=None, anchor_first=False, left_to_right=False
):
    b = memory.shape[0]
    length = head.cfg.block_size
    ids = torch.zeros((b, length), dtype=torch.long, device=memory.device)
    if head.kind in ("parallel", "probe"):
        logits = backbone.project(head(memory)).float()
        return draw(logits, temperature)
    mask = torch.ones_like(ids, dtype=torch.bool)
    if anchor_first:
        final = memory[:, len(head.cfg.layers) - 1, -1]
        ids[:, 0] = draw(backbone.project(final).float(), temperature)
        mask[:, 0] = False
    masked_length = length - int(anchor_first)
    steps = head.cfg.diffusion_steps if steps is None else steps
    if steps <= 0:
        raise ValueError("steps must be positive")
    for step in range(steps):
        noise = mask.float().mean(-1)
        logits = backbone.project(head(memory, backbone.embed(ids), mask, noise)).float()
        proposals = draw(logits, temperature)
        confidence = logits.softmax(-1).gather(-1, proposals[..., None]).squeeze(-1)
        confidence = confidence.masked_fill(~mask, -float("inf"))
        remaining = math.ceil(masked_length * (1 - (step + 1) / steps))
        to_commit = (mask.sum(-1) - remaining).clamp(min=0)
        if left_to_right:
            order = torch.arange(length, device=memory.device)[None].expand(b, -1)
            rank = order.masked_fill(~mask, length).argsort(-1).argsort(-1)
        else:
            rank = confidence.argsort(-1, descending=True).argsort(-1)
        commit = mask & (rank < to_commit[:, None])
        ids = torch.where(commit, proposals, ids)
        mask &= ~commit
    return ids


def draw(logits, temperature):
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if temperature == 0:
        return logits.argmax(-1)
    shape = logits.shape[:-1]
    probs = (logits / temperature).softmax(-1).reshape(-1, logits.shape[-1])
    return torch.multinomial(probs, 1).reshape(shape)
