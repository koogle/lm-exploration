"""CUDA-graph decoding with the custom kernels of `decode_ops.cu` (`engine="graph"`).

`GraphDecoder` keeps BF16 copies of the weights, a preallocated key/value cache and one
captured CUDA graph per (batch, capacity, top-k, greedy) shape; the prompt prefill runs
eagerly through `trainer.rollout.forward` into that cache. One replay of the graph is
one decode step for every row:

    x = embed(token)                                     # FP32 residual stream
    per block:
        q = rope_append(qkv(RMSNorm(x)))                 # writes k, v to the cache slot
        x = x + projection(attention(q, cache[start:slot+1]))
        x = x + down(silu(gate(RMSNorm(x))) * up(RMSNorm(x)))
    logits = lm_head(RMSNorm(x))
    token, log p = sample(logits / T, uniform[step])     # or argmax when greedy
    slot, step, positions += 1

The graph advances its own slot, step and positions on the device, so the host only
replays it and checks for EOS every `CHECK_EVERY` steps. Shapes are rounded up
(`CAPACITY_STEP`) so graphs are reused across batches of different prompt lengths; the
weights are copied into the same buffers before every rollout, so they stay current.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

_ops = None
CAPACITY_STEP = 256
CHECK_EVERY = 16
MAX_GRAPHS = 4


def ops():
    global _ops
    if _ops is None:
        from torch.utils.cpp_extension import load

        _ops = load(
            name="decode_ops",
            sources=[str(Path(__file__).with_name("decode_ops.cu"))],
            extra_cuda_cflags=["-O3"],
        )
    return _ops


def norm_ops():
    from kernels import fused

    return fused.ops()


class GraphDecoder:
    def __init__(self, model, batch, capacity, top_k, greedy):
        self.model, self.batch, self.capacity = model, batch, capacity
        self.top_k, self.greedy = top_k, greedy
        config, device = model.config, model.token_embedding.weight.device
        self.heads = model.blocks[0].attention.n_head
        if config.n_embd // self.heads != 64:
            raise ValueError("The decode kernels assume a head dimension of 64")
        shape = (batch, self.heads, capacity, 64)
        self.keys = [torch.zeros(shape, dtype=torch.bfloat16, device=device) for _ in model.blocks]
        self.values = [torch.zeros_like(k) for k in self.keys]
        self.weights = {
            name: torch.empty_like(p, dtype=torch.bfloat16)
            for name, p in model.named_parameters()
            if p.dim() == 2 and "token_embedding" not in name
        }
        self.weights["lm_head"] = torch.empty_like(model.lm_head.weight, dtype=torch.bfloat16)
        self.biases = {
            name: torch.empty_like(p, dtype=torch.bfloat16)
            for name, p in model.named_parameters()
            if p.dim() == 1 and "norm" not in name
        }
        width = config.n_embd // self.heads
        self.frequencies = 10000.0 ** (
            -torch.arange(0, width, 2, device=device, dtype=torch.float32) / width
        )
        integer = {"dtype": torch.int32, "device": device}
        self.token = torch.zeros(batch, dtype=torch.long, device=device)
        self.positions = torch.zeros(batch, **integer)
        self.start = torch.zeros(batch, **integer)
        self.slot = torch.zeros(1, **integer)
        self.step = torch.zeros(1, **integer)
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)
        self.eos = torch.zeros((), dtype=torch.long, device=device)
        self.uniforms = torch.zeros((capacity, batch), device=device)
        # What each step produced, written by the graph at column `step`.
        self.generated = torch.zeros((batch, capacity), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((batch, capacity), device=device)
        self.top_ids = torch.zeros(
            (batch, capacity, max(top_k, 1)), dtype=torch.long, device=device
        )
        self.top_logprobs = torch.zeros((batch, capacity, max(top_k, 1)), device=device)
        self.graph = None

    def copy_weights(self):
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, buffer in self.weights.items():
                source = self.model.lm_head.weight if name == "lm_head" else parameters[name]
                buffer.copy_(source)
            for name, buffer in self.biases.items():
                buffer.copy_(parameters[name])

    def linear(self, x, name):
        bias = self.biases.get(f"{name}.bias")
        return F.linear(x, self.weights[f"{name}.weight"], bias)

    def rope_append(self, qkv, index):
        """Rotated queries `[batch, heads, 64]`; rotated keys and values go to the cache."""
        return ops().rope_append(
            qkv, self.positions, self.slot, self.frequencies, self.keys[index], self.values[index]
        )

    def attend(self, q, index):
        """Attention of `q` over cache slots `[start, slot]` of layer `index`."""
        return ops().attention(q, self.keys[index], self.values[index], self.start, self.slot)

    def logits(self):
        """Next-token logits for `self.token` at `self.positions` and cache `self.slot`."""
        norms = norm_ops()
        x = self.model.token_embedding.weight[self.token]
        for index, block in enumerate(self.model.blocks):
            prefix = f"blocks.{index}"
            h = norms.rmsnorm_forward(x, block.attention_norm.weight, block.attention_norm.eps)[0]
            qkv = self.linear(h, f"{prefix}.attention.qkv")
            attended = self.attend(self.rope_append(qkv, index), index)
            x = x + self.linear(attended.view(self.batch, -1), f"{prefix}.attention.projection")
            h = norms.rmsnorm_forward(x, block.mlp_norm.weight, block.mlp_norm.eps)[0]
            hidden = F.silu(self.linear(h, f"{prefix}.mlp.gate")) * self.linear(
                h, f"{prefix}.mlp.up"
            )
            x = x + self.linear(hidden, f"{prefix}.mlp.down")
        final = self.model.final_norm
        h = norms.rmsnorm_forward(x, final.weight, final.eps)[0]
        return F.linear(h, self.weights["lm_head"]).float()

    def choose(self, logits, temperature):
        """Pick and record the token of column `step` from `logits`; it becomes `token`."""
        column = self.step.long()
        if self.greedy:
            chosen = logits.argmax(-1)
        else:
            chosen, logprob, normalizer = ops().sample(
                logits, self.uniforms, self.step, temperature
            )
            self.logprobs.index_copy_(1, column, logprob[:, None])
            if self.top_k:
                values, ids = logits.topk(self.top_k, -1)
                self.top_ids.index_copy_(1, column, ids[:, None])
                self.top_logprobs.index_copy_(
                    1, column, (values / temperature - normalizer[:, None])[:, None]
                )
        chosen = torch.where(self.done, self.eos, chosen)
        self.done |= chosen == self.eos
        self.generated.index_copy_(1, column, chosen[:, None])
        self.token.copy_(chosen)
        self.step += 1

    def decode_step(self, temperature):
        """Cache `token` at `positions` and `slot`, then choose the next token."""
        self.choose(self.logits(), temperature)
        self.positions += 1
        self.slot += 1

    def capture(self, temperature):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        saved = [t.clone() for t in (self.token, self.positions, self.slot, self.step, self.done)]
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.decode_step(temperature)
        torch.cuda.current_stream().wait_stream(stream)
        for target, value in zip(
            (self.token, self.positions, self.slot, self.step, self.done), saved, strict=True
        ):
            target.copy_(value)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.decode_step(temperature)
        self.temperature = temperature

    @torch.no_grad()
    def prefill(self, prompts):
        """Run the prompts into the cache eagerly; their next-token logits. Sets the
        positions, start and slot of the first decode step."""
        from trainer.rollout import forward

        device = self.token.device
        width = max(map(len, prompts))
        tokens = torch.zeros((self.batch, width), dtype=torch.long, device=device)
        valid = torch.zeros((self.batch, self.capacity), dtype=torch.bool, device=device)
        for row, prompt in enumerate(prompts):
            tokens[row, width - len(prompt) :] = torch.tensor(prompt, device=device)
            valid[row, width - len(prompt) : width] = True
        lengths = torch.tensor([len(p) for p in prompts], dtype=torch.int32, device=device)
        self.copy_weights()
        cache = list(zip(self.keys, self.values, strict=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            positions = (valid[:, :width].long().cumsum(1) - 1).clamp(min=0)
            logits = forward(self.model, tokens, positions, cache, 0, valid)
        self.positions.copy_(lengths)
        self.start.copy_(width - lengths)
        self.slot.fill_(width)
        return logits

    @torch.no_grad()
    def run(self, prompts, max_new_tokens, eos, temperature, generator):
        """Steps taken; tokens and sampler records are in `generated`, `logprobs`,
        `top_ids`, `top_logprobs` columns `[0, steps)` and `done`."""
        if not self.greedy:
            self.uniforms.uniform_(generator=generator)
        self.eos.fill_(eos)
        self.done.zero_()
        self.step.zero_()
        logits = self.prefill(prompts)
        self.choose(logits, temperature)
        if self.graph is None or self.temperature != temperature:
            self.capture(temperature)
        steps = 1
        while steps < max_new_tokens:
            if steps % CHECK_EVERY == 0 and bool(self.done.all()):
                break
            self.graph.replay()
            steps += 1
        return steps


def decoder(model, batch, capacity, top_k, greedy):
    """A cached `GraphDecoder` for this model and shape."""
    cache = model.__dict__.setdefault("_graph_decoders", {})
    key = (batch, capacity, top_k, greedy)
    if key not in cache:
        if len(cache) >= MAX_GRAPHS:
            cache.pop(next(iter(cache)))
            torch.cuda.empty_cache()
        cache[key] = GraphDecoder(model, batch, capacity, top_k, greedy)
    return cache[key]
