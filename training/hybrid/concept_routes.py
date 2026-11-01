"""FP32 cache-transplant factorial experiment on a frozen KDA hybrid checkpoint."""

import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from training_data.packed_shards import heldout_index, shard_paths

from . import generate, kda
from .concept_probes import save_json
from .model import HybridConfig, HybridLM
from .tokenizer import load

SOURCE = "kda-concepts-1793489611"
PAIRS = {
    "location": [("Paris", "Rome"), ("London", "Berlin"), ("Tokyo", "Sydney")],
    "color": [("red", "blue"), ("green", "yellow"), ("black", "white")],
    "boolean": [("True", "False")],
    "animal": [("cat", "dog"), ("horse", "cow")],
}
TEMPLATES = {
    "location": [
        ("{n} lives in {v}.", " The city where {n} lives is"),
        ("{n}'s home city is {v}.", " {n} lives in the city of"),
    ],
    "color": [
        ("The {n} is {v}.", " The color of the {n} is"),
        ("The {n} has been painted {v}.", " The paint on the {n} is"),
    ],
    "boolean": [
        ("{n} = {v}\n", "copy = {n}\n# The value of copy is"),
        ("flag = {v}\n{n} = flag\n", "# The value of {n} is"),
    ],
    "animal": [
        ("{n} owns a {v}.", " The animal owned by {n} is a"),
        ("{n}'s pet is a {v}.", " The pet belonging to {n} is a"),
    ],
}
NAMES = {
    "location": ["Iris", "Jack", "Kelly", "Liam"],
    "animal": ["Iris", "Jack", "Kelly", "Liam"],
    "color": ["coat", "fence", "bottle", "truck"],
    "boolean": ["available", "complete", "selected", "connected"],
}
ROUTES = [s for n in range(4) for s in itertools.combinations("KCA", n)]
MODES = (
    ["none"]
    + ["".join(r) for r in ROUTES if r]
    + [
        "city_head",
        "city_column",
        "other_column",
        "random_column",
        "random_head",
        "attention_3",
        "attention_7",
        "attention_11",
    ]
)


def examples(tok):
    enc = lambda x: tok.encode(x, add_special_tokens=False).ids
    filler = enc(" A separate note describes the weather and the passing days.")
    for task, pairs in PAIRS.items():
        for values in pairs:
            answers = [enc(" " + v) for v in values]
            assert all(len(a) == 1 for a in answers)
            for template, (prefix, query) in enumerate(TEMPLATES[task]):
                for entity, n in enumerate(NAMES[task]):
                    pre = [enc(prefix.format(n=n, v=v)) for v in values]
                    assert len(pre[0]) == len(pre[1]), (task, values, n)
                    suffix = enc(query.format(n=n))
                    for lag in [0, 32, 128]:
                        tail = (filler * (lag // len(filler) + 1))[:lag]
                        yield {
                            "task": task,
                            "values": values,
                            "template": template,
                            "entity": entity,
                            "name": n,
                            "lag": lag,
                            "prefixes": [p + tail for p in pre],
                            "suffix": suffix,
                            "answers": [a[0] for a in answers],
                        }


class ProbeDecoder(generate.Decoder):
    """Dimension-independent FP32 recurrence, captured by main's cache/graph driver."""

    def _kda(self, mixer, x, layer):
        point = x[:, 0]
        vectors = []
        for name, projection, weight in zip(
            "qkv",
            (mixer.q_proj, mixer.k_proj, mixer.v_proj),
            (mixer.q_conv, mixer.k_conv, mixer.v_conv),
            strict=True,
        ):
            history = layer[name]
            history.masked_fill_(self.fresh[:, None, None], 0)
            window = torch.cat((history, projection(point)[:, None]), dim=1)
            vector = F.silu((window * weight[:, 0].T[None]).sum(1))
            vectors.append(vector.view(self.rows, mixer.heads, mixer.head_dim))
            history.copy_(window[:, 1:])
        q, key, value = vectors
        q = F.normalize(q, dim=-1) / mixer.head_dim**0.5
        key = F.normalize(key, dim=-1)
        g = mixer.f_proj(point).view(self.rows, mixer.heads, mixer.head_dim)
        decay = kda.kda_gate(g, mixer.A_log, mixer.dt_bias).exp()
        state = layer["state"]
        state.masked_fill_(self.fresh[:, None, None, None], 0)
        faded = state * decay[..., None]
        error = value - torch.einsum("bhk,bhkv->bhv", key, faded)
        beta = mixer.b_proj(point).sigmoid()
        state.copy_(faded + torch.einsum("bhk,bhv->bhkv", beta[..., None] * key, error))
        read = torch.einsum("bhk,bhkv->bhv", q, state)
        read = read * torch.rsqrt(read.square().mean(-1, keepdim=True) + mixer.norm_eps)
        read = read * mixer.o_norm_weight
        gate = mixer.g_proj(point).view_as(read).sigmoid()
        return mixer.o_proj((read * gate).flatten(1))[:, None]

    def _attention(self, attention, x, layer, rope):
        query, key, value = attention.project(x, rope)
        layer["key"].index_copy_(2, self.slot, key)
        layer["value"].index_copy_(2, self.slot, value)
        groups = query.shape[1] // layer["key"].shape[1]
        keys = layer["key"].repeat_interleave(groups, dim=1)
        values = layer["value"].repeat_interleave(groups, dim=1)
        scores = (query @ keys.transpose(-1, -2)) / attention.head_dim**0.5
        indices = torch.arange(self.capacity, device=x.device)
        valid = (indices[None] >= self.starts[:, None]) & (indices[None] <= self.slot)
        scores = scores.masked_fill(~valid[:, None, None], -torch.inf)
        lse = scores.logsumexp(-1)
        read = scores.softmax(-1) @ values
        kept = torch.sigmoid(lse - attention.sink[None, :, None])
        read = (read * kept[..., None]).transpose(1, 2).flatten(2)
        gate = torch.sigmoid(attention.gate_proj(x) + attention.gate_bias)
        return attention.o_proj(read * gate)


@torch.no_grad()
def continuation(model, cache, prefix_len, suffix, answers):
    if not hasattr(model, "_route_decoder"):
        model._route_decoder = ProbeDecoder(model, 1, 512)
    dec = model._route_decoder
    dec.load(cache, [prefix_len], prefix_len)
    for token in suffix:
        logits = dec.step(torch.tensor([token], device="cuda"))
    return float(logits[0, answers[0]] - logits[0, answers[1]])


def transplant(target, donor, mode, noise_seed, noise_sign):
    """K: all KDA states, C: all convolution histories, A: all full-attention KV."""
    cache = [{k: v.clone() for k, v in layer.items()} for layer in target]
    if mode in ["none"] + ["".join(r) for r in ROUTES if r]:
        for i, layer in enumerate(cache):
            for name in layer:
                route = "A" if name in ["key", "value"] else "K" if name == "state" else "C"
                if route in mode:
                    layer[name].copy_(donor[i][name])
        return cache
    if mode.startswith("attention_"):
        index = int(mode.split("_")[1])
        for name in cache[index]:
            cache[index][name].copy_(donor[index][name])
        return cache
    # Frozen localization from the original discovery set, never reselected here.
    s = cache[8]["state"]
    d = donor[8]["state"] - target[8]["state"]
    if mode == "city_head":
        s[:, 3] += d[:, 3]
    if mode == "city_column":
        s[:, 3, :, 13] += d[:, 3, :, 13]
    if mode == "other_column":
        delta = d[:, 3, :, 14]
        delta = delta * d[:, 3, :, 13].norm() / delta.norm().clamp_min(1e-12)
        s[:, 3, :, 14] += delta
    if mode in ["random_head", "random_column"]:
        region = d[:, 3] if mode == "random_head" else d[:, 3, :, 13]
        rng = torch.Generator(device="cuda").manual_seed(noise_seed)
        noise = torch.randn(region.shape, generator=rng, device="cuda")
        noise *= noise_sign * region.norm() / noise.norm().clamp_min(1e-12)
        if mode == "random_head":
            s[:, 3] += noise
        else:
            s[:, 3, :, 13] += noise
    return cache


@torch.no_grad()
def precision_check(model, ex):
    out = []
    for prefix in ex["prefixes"]:
        x = torch.tensor([prefix + ex["suffix"]], device="cuda")
        ans = ex["answers"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.hidden_states(x)[:, -1]
            low = F.linear(hidden, model.output_weight).float()[0]
        full_readout = F.linear(hidden.float(), model.output_weight.float())[0]
        native = F.linear(model.hidden_states(x)[:, -1], model.output_weight)[0]
        out.append(
            {
                "bf16_readout": float(low[ans[0]] - low[ans[1]]),
                "fp32_readout_bf16_trunk": float(full_readout[ans[0]] - full_readout[ans[1]]),
                "fp32": float(native[ans[0]] - native[ans[1]]),
            }
        )
    return out


@torch.no_grad()
def natural_atlas(model, tok, folder, volume, deadline):
    """One strongest example per original document, with fixed discovery coordinates."""
    recipe = json.loads((Path(__file__).parent / "recipes/hybrid-40b-v1.json").read_text())
    root = Path("/data") / recipe["name"]
    coordinates = [(8, 3, 13), (8, 3, 12), (8, 3, 14), (8, 2, 13), (0, 0, 27)]
    records = {
        f"{l}:{h}:{c}": {"layer": l, "head": h, "column": c, "positive": [], "negative": []}
        for l, h, c in coordinates
    }
    original = kda.kda_kernel
    indices = [i for i, b in enumerate(model.blocks) if b.kind == "kda"]
    count = 0
    for source, config in recipe["sources"].items():
        bp, dp, _ = shard_paths(root, source, heldout_index(config))
        docs = np.load(dp)
        ends = docs[:, 0].cumsum()
        starts = np.r_[0, ends[:-1]]
        tokens = np.fromfile(bp, dtype=np.uint32)
        eligible = np.flatnonzero(docs[:, 0] >= 128)
        rng = np.random.default_rng(2941)
        for doc in rng.choice(eligible, min(100, len(eligible)), replace=False):
            if time.time() > deadline - 120:
                break
            length = min(256, int(docs[doc, 0]) - 1)
            ids = tokens[int(starts[doc]) : int(starts[doc]) + length].astype(np.int64).tolist()
            calls = 0

            def kernel(
                q,
                k,
                v,
                g,
                beta,
                A,
                bias,
                documents,
                final_state=False,
                ids=ids,
                source=source,
                doc=int(doc),
            ):
                nonlocal calls
                layer = indices[calls]
                calls += 1
                out = original(q, k, v, g, beta, A, bias, documents, final_state)
                for l, h, c in coordinates:
                    if l != layer:
                        continue
                    values = out[0, : len(ids), h, c].float()
                    entry = records[f"{l}:{h}:{c}"]
                    for pole, sign in [("positive", 1), ("negative", -1)]:
                        t = int((values * sign).argmax())
                        entry[pole].append(
                            {
                                "activation": float(values[t]),
                                "source": source,
                                "document": doc,
                                "position": t,
                                "prefix": tok.decode(ids[max(0, t - 48) : t + 1]),
                                "following": tok.decode(ids[t + 1 : t + 17]),
                            }
                        )
                        entry[pole] = sorted(
                            entry[pole], key=lambda r: sign * r["activation"], reverse=True
                        )[:12]
                return out

            kda.kda_kernel = kernel
            try:
                model.hidden_states(
                    torch.tensor([ids + [model.config.eos_id] * (256 - len(ids))], device="cuda")
                )
            finally:
                kda.kda_kernel = original
            count += 1
        save_json(
            folder / "atlas.json",
            {"documents": count, "coordinates": list(records.values())},
            volume,
        )
        del tokens
    return count


@torch.no_grad()
def run(folder, volume, deadline, provenance):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.config.recompile_limit = 128
    ck = torch.load(Path("/runs") / SOURCE / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = HybridLM(HybridConfig(**ck["config"])).cuda().eval()
    model.load_state_dict(ck["model"])
    trained_tokens = ck["tokens"]
    del ck
    tok = load("/project/glm-tokenizer.json")
    generate.PREFILL_MIN = 256
    result = {
        "status": "running",
        "source": SOURCE,
        "trained_tokens": trained_tokens,
        "provenance": provenance,
        "precision": "FP32 weights, activations and readout; TF32 disabled",
        "rows": [],
        "qualification": [],
        "precision_comparison": [],
    }
    for index, ex in enumerate(examples(tok)):
        if time.time() > deadline - 600:
            result["status"] = "deadline"
            break
        caches = [generate._prefill_row(model, p, "cuda")[1] for p in ex["prefixes"]]
        length = len(ex["prefixes"][0])
        suffix = ex["suffix"]
        answers = ex["answers"]
        base = [continuation(model, c, length, suffix, answers) for c in caches]
        if index in [0, 24, 72, 144]:
            precision = precision_check(model, ex)
            error = max(abs(a["fp32"] - b) for a, b in zip(precision, base, strict=True))
            check = {"example": index, "prefill_decode_vs_full_error": error}
            result["qualification"].append(check)
            assert error < 0.01, check
            result["precision_comparison"].append(
                {"example": index, "task": ex["task"], "values": ex["values"], "scores": precision}
            )
        for direction in [0, 1]:
            sign = 1 if direction == 0 else -1
            donor, target = caches[direction], caches[1 - direction]
            for mode in MODES:
                cache = transplant(target, donor, mode, 1793 + index, sign)
                changed = continuation(model, cache, length, suffix, answers)
                if mode == "none":
                    assert abs(changed - base[1 - direction]) < 1e-5
                if mode == "KCA":
                    assert abs(changed - base[direction]) < 1e-5
                row = {k: v for k, v in ex.items() if k not in ["prefixes", "suffix", "answers"]}
                row.update(
                    direction=direction,
                    mode=mode,
                    donor_logodds=base[direction],
                    target_logodds=base[1 - direction],
                    patched_logodds=changed,
                    oriented_effect=sign * (changed - base[1 - direction]),
                    clean_gap=sign * (base[direction] - base[1 - direction]),
                    paired_correct=sign * base[direction] > 0 and sign * base[1 - direction] < 0,
                )
                result["rows"].append(row)
        if index % 12 == 0:
            save_json(folder / "routes.json", result, volume)
            print("ROUTES", index + 1, "rows", len(result["rows"]), flush=True)
    else:
        result["status"] = "complete"
    save_json(folder / "routes.json", result, volume)
    result["atlas_documents"] = natural_atlas(model, tok, folder, volume, deadline)
    return {k: v for k, v in result.items() if k not in ["rows", "provenance"]}
