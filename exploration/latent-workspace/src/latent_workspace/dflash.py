"""DFlash-style Qwen2 draft layers and greedy verified decoding.

Architecture adapted from Z Lab's MIT-licensed DFlash reference, commit
07ebd93db9f472af339b644bb70221ad8428328a. See references/dflash-LICENSE.
No workspace, J-lens, predicted hidden states, or learned layer selection.
"""

import time
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm, Qwen2RotaryEmbedding


@dataclass
class DraftConfig:
    model: str = "Qwen/Qwen2.5-0.5B"
    model_revision: str = "060db6499f32faf8b98477b0a26969ef7d8b9987"
    layers: tuple[int, ...] = (2, 22)  # Fixed, one-based post-block features.
    draft_layers: int = 2
    block_size: int = 8  # One target-supplied anchor plus seven NEW proposals.
    gamma: float = 4.0
    train_prompts: int = 384
    validation_prompts: int = 48
    test_prompts: int = 48
    prompt_tokens: int = 128
    response_tokens: int = 64
    batch_size: int = 4
    anchors_per_sequence: int = 8
    steps: int = 1000
    eval_every: int = 100
    learning_rate: float = 0.0006
    warmup_ratio: float = 0.04
    seed: int = 42
    max_seconds: int = 720
    continue_from: str = ""
    historical_corpus: str = "/runs/data-large-01/corpus.pt"
    prompt_manifest: str = ""

    def validate(self):
        if self.model != "Qwen/Qwen2.5-0.5B" or len(self.model_revision) != 40:
            raise ValueError("baseline requires pinned Qwen2.5-0.5B")
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("invalid fixed feature layers")
        if self.block_size < 2 or self.response_tokens < self.block_size or self.gamma <= 0:
            raise ValueError("invalid block/response/loss decay")
        if any(
            x < 3 or x % 3 for x in (self.train_prompts, self.validation_prompts, self.test_prompts)
        ):
            raise ValueError("prompt splits must be balanced positive multiples of three")
        if (
            min(
                self.steps,
                self.eval_every,
                self.batch_size,
                self.anchors_per_sequence,
                self.prompt_tokens,
                self.draft_layers,
                self.max_seconds,
            )
            < 1
        ):
            raise ValueError("counts must be positive")
        if self.learning_rate <= 0 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("invalid optimizer configuration")
        return self

    def to_dict(self):
        return asdict(self)


def features(output, layers):
    return torch.cat([output.hidden_states[i] for i in layers], dim=-1).detach()


def rotate(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return x * cos[:, None] + torch.cat((-b, a), dim=-1) * sin[:, None]


class DraftLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_size
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = h // self.heads
        self.q = nn.Linear(h, h, bias=True)
        self.k = nn.Linear(h, self.kv_heads * self.dim, bias=True)
        self.v = nn.Linear(h, self.kv_heads * self.dim, bias=True)
        self.o = nn.Linear(h, h, bias=False)
        self.norm = Qwen2RMSNorm(h, eps=config.rms_norm_eps)
        self.post_norm = Qwen2RMSNorm(h, eps=config.rms_norm_eps)
        self.mlp = Qwen2MLP(config)

    def kv(self, x, cos, sin):
        b, t, _ = x.shape
        k = self.k(x).view(b, t, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, self.dim).transpose(1, 2)
        return rotate(k, cos, sin), v

    def forward(self, x, context_kv, cos, sin, mask=None):
        residual = x
        z = self.norm(x)
        q = self.q(z).view(x.shape[0], x.shape[1], self.heads, self.dim).transpose(1, 2)
        q = rotate(q, cos, sin)
        k, v = self.kv(z, cos, sin)
        k = torch.cat((context_kv[0], k), dim=-2)
        v = torch.cat((context_kv[1], v), dim=-2)
        repeats = self.heads // self.kv_heads
        z = (
            F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(repeats, dim=1),
                v.repeat_interleave(repeats, dim=1),
                attn_mask=mask,
                dropout_p=0.0,
            )
            .transpose(1, 2)
            .reshape_as(x)
        )
        x = residual + self.o(z)
        return x + self.mlp(self.post_norm(x))


class DFlashDrafter(nn.Module):
    def __init__(self, target_config, cfg):
        super().__init__()
        self.cfg = cfg
        if max(cfg.layers) >= target_config.num_hidden_layers or min(cfg.layers) < 1:
            raise ValueError("features must be intermediate post-block states")
        h = target_config.hidden_size
        self.fc = nn.Linear(len(cfg.layers) * h, h, bias=False)
        self.hidden_norm = Qwen2RMSNorm(h, eps=target_config.rms_norm_eps)
        self.layers = nn.ModuleList([DraftLayer(target_config) for _ in range(cfg.draft_layers)])
        self.norm = Qwen2RMSNorm(h, eps=target_config.rms_norm_eps)
        self.rotary = Qwen2RotaryEmbedding(target_config)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=target_config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def context_cache(self, memory, positions, previous=None):
        x = self.hidden_norm(self.fc(memory.to(self.fc.weight.dtype)))
        cos, sin = self.rotary(x, positions)
        cache = []
        for i, layer in enumerate(self.layers):
            kv = layer.kv(x, cos, sin)
            if previous is not None:
                kv = tuple(torch.cat((old, new), dim=-2) for old, new in zip(previous[i], kv))
            cache.append(kv)
        return cache

    def forward(self, embeddings, positions, context_cache, mask=None):
        x = embeddings.to(self.fc.weight.dtype)
        cos, sin = self.rotary(x, positions)
        for layer, kv in zip(self.layers, context_cache):
            x = layer(x, kv, cos, sin, mask)
        return self.norm(x)


def pack_blocks(sequences, anchors, block_size, mask_id):
    """Strictly pre-anchor context; bidirectional within each isolated draft block."""
    b, length = sequences.shape
    count = anchors.shape[1]
    positions = anchors[..., None] + torch.arange(block_size, device=sequences.device)
    labels = sequences.gather(1, positions.reshape(b, -1))
    masked = (torch.arange(count * block_size, device=sequences.device) % block_size) != 0
    noisy = labels.masked_fill(masked[None], mask_id)
    context_mask = (
        torch.arange(length, device=sequences.device)[None, None]
        < (anchors.repeat_interleave(block_size, dim=1)[..., None])
    )
    ids = torch.arange(count, device=sequences.device).repeat_interleave(block_size)
    within = (ids[:, None] == ids[None, :])[None].expand(b, -1, -1)
    attention_mask = torch.cat((context_mask, within), dim=-1)[:, None]
    return noisy, labels, positions.reshape(b, -1), attention_mask, masked


def block_loss(drafter, target, sequences, memory, anchors, mask_id):
    cfg = drafter.cfg
    noisy, labels, positions, mask, selected = pack_blocks(
        sequences, anchors, cfg.block_size, mask_id
    )
    context_positions = torch.arange(sequences.shape[1], device=sequences.device)[None].expand(
        len(sequences), -1
    )
    cache = drafter.context_cache(memory, context_positions)
    hidden = drafter(target.get_input_embeddings()(noisy).detach(), positions, cache, mask)
    logits = target.get_output_embeddings()(hidden[:, selected].to(target.dtype)).float()
    labels = labels[:, selected]
    ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    weights = torch.exp(-torch.arange(cfg.block_size - 1, device=ce.device) / cfg.gamma)
    weights = weights.repeat(anchors.shape[1])
    loss = (ce * weights).sum() / (len(sequences) * weights.sum())
    return loss, ce, logits.argmax(-1), labels


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def verified_generate(drafter, target, prompt, max_new_tokens, mask_id, eos_id=None):
    """Greedy-only verifier. Anchor is excluded from accepted-draft statistics."""
    if max_new_tokens < 1 or prompt.shape[0] != 1:
        raise ValueError("positive generation length and batch one required")
    drafter.eval()
    synchronize(prompt.device)
    start_time = time.perf_counter()
    output = target(prompt, use_cache=True, output_hidden_states=True, logits_to_keep=1)
    cache = output.past_key_values
    anchor = output.logits[:, -1].argmax(-1)
    positions = torch.arange(prompt.shape[1], device=prompt.device)[None]
    draft_cache = drafter.context_cache(features(output, drafter.cfg.layers), positions)
    synchronize(prompt.device)
    prefill_seconds = time.perf_counter() - start_time
    decode_start = time.perf_counter()
    emitted, accepted, proposals = [], [], []
    calls = 0
    while len(emitted) < max_new_tokens:
        remaining = max_new_tokens - len(emitted)
        if remaining == 1 or (eos_id is not None and anchor.item() == eos_id):
            emitted.append(anchor.item())
            break
        width = min(drafter.cfg.block_size, remaining)
        start = prompt.shape[1] + len(emitted)
        pos = torch.arange(start, start + width, device=prompt.device)[None]
        block = torch.full((1, width), mask_id, device=prompt.device, dtype=torch.long)
        block[:, 0] = anchor
        hidden = drafter(target.get_input_embeddings()(block), pos, draft_cache)
        block[:, 1:] = target.get_output_embeddings()(hidden[:, 1:].to(target.dtype)).argmax(-1)
        output = target(block, past_key_values=cache, use_cache=True, output_hidden_states=True)
        posterior = output.logits.argmax(-1)
        k = int((block[:, 1:] == posterior[:, :-1]).long().cumprod(-1).sum())
        commit = block[0, : k + 1].tolist()
        stopped = eos_id is not None and eos_id in commit
        if stopped:
            commit = commit[: commit.index(eos_id) + 1]
        emitted.extend(commit)
        accepted.append(len(commit) - 1)
        proposals.append(width - 1)
        calls += 1
        if stopped or len(emitted) == max_new_tokens:
            break
        cache = output.past_key_values
        cache.crop(start + len(commit))
        draft_cache = drafter.context_cache(
            features(output, drafter.cfg.layers)[:, : len(commit)],
            pos[:, : len(commit)],
            draft_cache,
        )
        anchor = posterior[:, k]
    synchronize(prompt.device)
    return {
        "tokens": emitted,
        "accepted_new_draft_tokens": accepted,
        "proposed_new_tokens": proposals,
        "verification_calls": calls,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": time.perf_counter() - decode_start,
        "total_seconds": time.perf_counter() - start_time,
        "mean_accepted_beyond_anchor": sum(accepted) / len(accepted) if accepted else 0.0,
    }
