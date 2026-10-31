"""Generation for HybridLM on CUDA: one-pass prefill, then a CUDA-graph decode step.

`prefill` reads each prompt with one forward pass of the model (the training kernels;
the KDA kernels return each document's final state) and loads the rows into a
`Decoder`, left-padded to the longest prompt. The Decoder keeps preallocated buffers
per layer: full attention its rotated keys and values, KDA the last
`kda_conv_size - 1` q/k/v projections for its short convolutions and its fp32
recurrent state. One step feeds one token per row: projections and the MLP through
cuBLAS, the KDA update and the attention read through the kernels of
kernels/decode.cu, the whole step captured once as a CUDA graph and replayed.

Rows follow the model's document rule: after a row feeds EOS, its next token starts
a new document (attention reads only keys from there on, the convolutions and state
reset and RoPE positions restart).
"""

import torch
from torch.nn import functional as F

from .cuda_extension import load as load_kernel
from .model import DocumentMask, HybridLM

PREFILL_MIN = 1024
# Decoders kept per model, by (rows, capacity); capacity is rounded up to this step.
CAPACITY_STEP = 256
MAX_DECODERS = 2


class Decoder:
    """Decode state for `rows` rows and up to `capacity` cached tokens, and the CUDA
    graph of one step. The graph reads the model's parameters in place, so it follows
    in-place weight updates."""

    def __init__(self, lm: HybridLM, rows, capacity):
        if not lm.token_embedding.weight.is_cuda:
            raise ValueError("decoding runs on CUDA")
        self.lm, self.rows, self.capacity = lm, rows, capacity
        config, device = lm.config, lm.token_embedding.weight.device
        dtype = lm.token_embedding.weight.dtype
        self.ops = load_kernel("decode")
        long = {"dtype": torch.long, "device": device}
        self.token = torch.zeros(rows, **long)
        self.positions = torch.zeros(rows, **long)
        self.starts = torch.zeros(rows, **long)
        self.slot = torch.zeros(1, **long)
        self.fresh = torch.ones(rows, dtype=torch.bool, device=device)
        self.layers = []
        for block in lm.blocks:
            mixer = block.mixer
            if block.kind == "kda":
                inner = mixer.heads * mixer.head_dim
                history = (rows, config.kda_conv_size - 1, inner)
                self.layers.append(
                    {
                        **{n: torch.zeros(history, device=device) for n in "qkv"},
                        "state": torch.zeros(
                            rows, mixer.heads, mixer.head_dim, mixer.head_dim, device=device
                        ),
                    }
                )
            else:
                shape = (rows, mixer.n_kv_head, capacity, mixer.head_dim)
                self.layers.append(
                    {
                        "key": torch.zeros(shape, dtype=dtype, device=device),
                        "value": torch.zeros(shape, dtype=dtype, device=device),
                    }
                )
        self.logits = torch.zeros(rows, config.vocab_size, device=device)
        self.used = 0  # cache slots holding tokens, on the host
        self.parameters = parameter_pointers(lm)
        self.graph = self._capture()

    def _kda(self, kda, x, layer):
        point = x[:, 0]
        projections = [p(point).float() for p in (kda.q_proj, kda.k_proj, kda.v_proj)]
        out = self.ops.kda_step(
            *projections,
            kda.f_proj(point).float(),
            kda.b_proj(point).float(),
            kda.g_proj(point).float(),
            *(w.float()[:, 0].contiguous() for w in (kda.q_conv, kda.k_conv, kda.v_conv)),
            layer["q"],
            layer["k"],
            layer["v"],
            layer["state"],
            self.fresh,
            kda.A_log.float(),
            kda.dt_bias.float(),
            kda.o_norm_weight.float(),
            kda.norm_eps,
        )
        return kda.o_proj(out.to(x.dtype))[:, None]

    def _attention(self, attention, x, layer, rope):
        query, key, value = attention.project(x, rope)
        layer["key"].index_copy_(2, self.slot, key)
        layer["value"].index_copy_(2, self.slot, value)
        attended = self.ops.attention(
            query[:, :, 0].contiguous(),
            layer["key"],
            layer["value"],
            self.starts,
            self.slot,
            attention.sink.float(),
        )
        attended = attended.to(x.dtype).reshape(self.rows, 1, -1)
        gate = torch.sigmoid(attention.gate_proj(x) + attention.gate_bias)
        return attention.o_proj(attended * gate.to(attended.dtype))

    def _step(self):
        lm = self.lm
        self.positions.masked_fill_(self.fresh, 0)
        self.starts.copy_(torch.where(self.fresh, self.slot, self.starts))
        x = lm.token_embedding(self.token[:, None])
        cos, sin = lm.rope_cos[self.positions], lm.rope_sin[self.positions]
        rope = (cos[:, None, None], sin[:, None, None])
        for block, layer in zip(lm.blocks, self.layers, strict=True):
            normed = block.mixer_norm(x)
            if block.kind == "kda":
                x = x + self._kda(block.mixer, normed, layer)
            else:
                x = x + self._attention(block.mixer, normed, layer, rope)
            x = x + block.mlp(block.mlp_norm(x))
        self.logits.copy_(F.linear(lm.final_norm(x), lm.output_weight)[:, 0])
        self.slot += 1
        self.positions += 1
        self.fresh.copy_(self.token == lm.config.eos_id)

    @torch.no_grad()
    def _capture(self):
        # Warm up on a side stream (cuBLAS handles, kernel loading), then capture; the
        # buffers are reset afterwards and filled by `load`.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step()
        return graph

    @torch.no_grad()
    def load(self, entries, lengths, width):
        """Fill the buffers from prefilled rows: `entries` [layer][name] batched over rows,
        keys and values left-padded to `width`, the next slot at `width`."""
        for layer, entry in zip(self.layers, entries, strict=True):
            if "key" in layer:
                layer["key"].zero_()
                layer["value"].zero_()
                layer["key"][:, :, :width].copy_(entry["key"])
                layer["value"][:, :, :width].copy_(entry["value"])
            else:
                for name in ("q", "k", "v", "state"):
                    layer[name].copy_(entry[name])
        lengths = torch.as_tensor(lengths, device=self.token.device)
        self.positions.copy_(lengths)
        self.starts.copy_(width - lengths)
        self.slot.fill_(width)
        self.fresh.zero_()
        self.used = width

    @torch.no_grad()
    def step(self, tokens):
        """Logits [rows, vocab] for the next token after feeding `tokens` (one id per row)."""
        if self.used >= self.capacity:
            raise ValueError("the decoder's cache is full")
        self.token.copy_(torch.as_tensor(tokens).view(-1))
        self.graph.replay()
        self.used += 1
        return self.logits


def parameter_pointers(lm):
    return tuple(p.data_ptr() for p in lm.parameters())


def decoder(lm, rows, capacity):
    """A Decoder for `lm` with at least `capacity` slots, reused across calls while the
    model's parameters stay in the same memory."""
    capacity = -(-capacity // CAPACITY_STEP) * CAPACITY_STEP
    decoders = lm.__dict__.setdefault("_decoders", {})
    if decoders and next(iter(decoders.values())).parameters != parameter_pointers(lm):
        decoders.clear()
    key = (rows, capacity)
    if key not in decoders:
        while len(decoders) >= MAX_DECODERS:
            decoders.pop(next(iter(decoders)))
        decoders[key] = Decoder(lm, rows, capacity)
    return decoders[key]


def _sample(logits, temperature, top_p, generator):
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, -1)
    if top_p >= 1:
        return torch.multinomial(probs, 1, generator=generator)[:, 0]
    ordered, index = probs.sort(-1, descending=True)
    ordered = ordered * (ordered.cumsum(-1) - ordered < top_p)
    choice = torch.multinomial(ordered / ordered.sum(-1, keepdim=True), 1, generator=generator)
    return index.gather(-1, choice)[:, 0]


def _prefill_row(lm, prompt, device):
    """Next-token logits [vocab] after one prompt and its per-layer cache entries.

    On CUDA the prompt is padded to a power of two (at least PREFILL_MIN, at most the
    context), so compiled attention sees few shapes; the padding is one second
    document and changes nothing in the prompt.
    """
    length = len(prompt)
    width = length
    if lm.token_embedding.weight.is_cuda:
        width = min(max(PREFILL_MIN, 1 << (length - 1).bit_length()), lm.config.block_size)
    tokens = torch.tensor([list(prompt) + [lm.config.eos_id] * (width - length)], device=device)
    documents = (torch.arange(width, device=device) >= length).long()[None]
    mask = DocumentMask(documents)
    rope = (lm.rope_cos[mask.positions][:, None], lm.rope_sin[mask.positions][:, None])
    x = lm.token_embedding(tokens)
    layers = []
    for block in lm.blocks:
        normed = block.mixer_norm(x)
        layer = {}
        if block.kind == "kda":
            mixed = block.mixer(normed, rope, mask, cache=layer)
        else:
            _, key, value = block.mixer.project(normed, rope)
            layer["key"], layer["value"] = key[:, :, :length].clone(), value[:, :, :length].clone()
            mixed = block.mixer(normed, rope, mask)
        x = x + mixed
        x = x + block.mlp(block.mlp_norm(x))
        layers.append(layer)
    logits = F.linear(lm.final_norm(x[:, length - 1]), lm.output_weight)[0].float()
    return logits, layers


@torch.no_grad()
def prefill(lm, prompts, new_tokens):
    """Logits [rows, vocab] after each prompt and the Decoder that continues them for up
    to `new_tokens` steps. Each prompt is one document (it may not contain EOS)."""
    if not prompts or not all(prompts):
        raise ValueError("every prompt must contain at least one token")
    if any(lm.config.eos_id in p for p in prompts):
        raise ValueError("prompts are single documents and may not contain EOS")
    if max(len(p) for p in prompts) > lm.config.block_size:
        raise ValueError("a prompt is longer than the model's context")
    device = lm.token_embedding.weight.device
    # Each distinct prompt is read once (RL samples a group of answers per prompt).
    read = {}
    for prompt in map(tuple, prompts):
        if prompt not in read:
            read[prompt] = _prefill_row(lm, prompt, device)
    rows = [read[tuple(p)] for p in prompts]
    width = max(len(p) for p in prompts)
    entries = []
    for index in range(len(lm.blocks)):
        entry = {}
        for name in rows[0][1][index]:
            parts = [layers[index][name] for _, layers in rows]
            if name in ("key", "value"):
                parts = [F.pad(part, (0, 0, width - part.size(2), 0)) for part in parts]
            entry[name] = torch.cat(parts)
        entries.append(entry)
    state = decoder(lm, len(prompts), width + max(new_tokens, 1))
    state.load(entries, [len(p) for p in prompts], width)
    return torch.stack([logits for logits, _ in rows]), state


@torch.no_grad()
def generate_batch(
    lm, prompts, max_new_tokens, stop_ids, temperature=0.0, top_p=1.0, generator=None, done=None
):
    """New token ids per prompt, each up to and including its first stop id.

    Prompts are read by `prefill` (one document each), so each row decodes as it
    would alone. `done(row, ids)` may end a row early (for example on a stop string).
    """
    room = lm.config.block_size + 1 - max(len(p) for p in prompts)
    limit = max(0, min(max_new_tokens, room))
    logits, state = prefill(lm, prompts, limit)
    outputs = [[] for _ in prompts]
    finished = [False] * len(prompts)
    stops = torch.tensor(sorted(stop_ids), device=logits.device)
    for index in range(limit):
        token = _sample(logits, temperature, top_p, generator)
        stopped = torch.isin(token, stops).tolist()
        for row, value in enumerate(token.tolist()):
            if finished[row]:
                continue
            outputs[row].append(value)
            finished[row] = stopped[row] or bool(done and done(row, outputs[row]))
        if all(finished) or index == limit - 1:
            break
        logits = state.step(token)
    return outputs


@torch.no_grad()
def generate(lm, prompt_ids, max_new_tokens, stop_ids, temperature=0.0, top_p=1.0, generator=None):
    """Yield new token ids up to and including one of `stop_ids`; greedy at temperature 0,
    else nucleus sampling. The prompt is one document (it may not contain EOS)."""
    limit = max(0, min(max_new_tokens, lm.config.block_size + 1 - len(prompt_ids)))
    logits, state = prefill(lm, [list(prompt_ids)], limit)
    for index in range(limit):
        token = _sample(logits, temperature, top_p, generator)
        yield int(token[0])
        if int(token[0]) in stop_ids or index == limit - 1:
            return
        logits = state.step(token)


def load_export(folder, dtype=torch.float32, device="cpu"):
    """The inference trunk from a bf16 export folder (config.json, model.safetensors).

    Exports from hybrid-720m-40b-01 predate `HybridConfig.eos_id` and carry its
    former `attention_kernel` field; both are reconciled here, with EOS from the
    pinned GLM tokenizer the run trained with.
    """
    import json
    from dataclasses import fields
    from pathlib import Path

    from safetensors.torch import load_file

    from tokenizer import glm

    from .model import HybridConfig

    folder = Path(folder)
    saved = json.loads((folder / "config.json").read_text())
    if saved.get("tokenizer_sha256", glm.SHA256) != glm.SHA256:
        raise ValueError("export was trained with a different tokenizer")
    known = {f.name for f in fields(HybridConfig)}
    settings = {k: tuple(v) if isinstance(v, list) else v for k, v in saved.items() if k in known}
    settings.update(eos_id=saved.get("eos_id", glm.EOS_ID), mtp=False)
    lm = HybridLM(HybridConfig(**settings))
    lm.load_state_dict(load_file(folder / "model.safetensors"))
    return lm.to(device=device, dtype=dtype).eval()
