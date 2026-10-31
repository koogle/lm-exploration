"""Native KDA channel atlas and held-out, bidirectional state patching.

No learned feature decomposition. The recurrence is instrumented only for probes;
training uses the unchanged main-branch implementation.
"""

import contextlib
import json
import time

import torch
from torch.nn import functional as F

from . import kda
from .documents import document_positions

LAYERS = (0, 4, 8)
TASKS = {
    "color": ("red", "blue", "The {noun} is {value}.", " The color of the {noun} is"),
    "location": ("Paris", "Rome", "{noun} lives in {value}.", " The city where {noun} lives is"),
    "boolean": ("True", "False", "{noun} = {value}\n", "\ncopy = {noun}\n# The value of copy is"),
}
NAMES = {
    "color": ("box", "cup", "ball", "car", "hat", "bag", "door", "chair"),
    "location": ("Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry"),
    "boolean": ("ready", "active", "valid", "enabled", "visible", "found", "open_flag", "done"),
}


def recurrence(q, k, v, g, beta, A, bias, documents, boundary, patch=None):
    """FP32 reference: replace state AFTER the prefix update, BEFORE any suffix read."""
    with torch.autocast("cuda", enabled=False):
        q = F.normalize(q.float(), dim=-1) / q.shape[-1] ** 0.5
        k = F.normalize(k.float(), dim=-1)
        v, beta = v.float(), beta.float().sigmoid()
        decay = kda.kda_gate(g, A, bias).exp()
        positions = document_positions(documents)
        state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3], device=q.device)
        outputs, saved = [], None
        for t in range(q.shape[1]):
            state = state * (positions[:, t] > 0)[:, None, None, None] * decay[:, t, :, :, None]
            error = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)
            state = state + torch.einsum("bhk,bhv->bhkv", beta[:, t, :, None] * k[:, t], error)
            outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
            if t == boundary:
                saved = state.clone()
                if patch is not None:
                    state = patch(state)
        return torch.stack(outputs, 1), saved


@contextlib.contextmanager
def instrument(model, layer, boundary, patch=None):
    """Replace only the requested layer's recurrence; preserve all convolutions."""
    original = kda.kda_kernel
    indices = [i for i, b in enumerate(model.blocks) if isinstance(b.mixer, kda.KimiDeltaAttention)]
    captured, calls = {}, 0

    def kernel(q, k, v, g, beta, A, bias, documents, final_state=False):
        nonlocal calls
        current = indices[calls]
        calls += 1
        assert not final_state
        if current != layer:
            return original(q, k, v, g, beta, A, bias, documents)
        out, state = recurrence(q, k, v, g, beta, A, bias, documents, boundary, patch)
        captured["state"] = state
        return out.to(v.dtype)

    kda.kda_kernel = kernel
    try:
        yield captured
    finally:
        kda.kda_kernel = original


@torch.no_grad()
def score(model, ids, answers, layer=None, boundary=0, patch=None):
    ctx = (
        instrument(model, layer, boundary, patch)
        if layer is not None
        else contextlib.nullcontext({})
    )
    with ctx as capture, torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.hidden_states(torch.tensor([ids], device="cuda"))[:, -1]
        logits = F.linear(hidden, model.output_weight).float()[0]
    return float(logits[answers[0]] - logits[answers[1]]), capture.get("state")


def prompts(tok, task, index, lag):
    a, b, prefix, query = TASKS[task]
    noun = NAMES[task][index]
    encode = lambda s: tok.encode(s, add_special_tokens=False).ids
    answers = [encode(" " + word) for word in (a, b)]
    assert all(len(x) == 1 for x in answers), (task, answers)
    # Exactly lag token distractors, identical across the counterfactual pair.
    filler = encode(" A separate note describes the weather and the passing days.")
    suffix = (filler * (lag // len(filler) + 1))[:lag] + encode(query.format(noun=noun))
    prefixes = [encode(prefix.format(noun=noun, value=value)) for value in (a, b)]
    return [(p + suffix, len(p) - 1) for p in prefixes], [x[0] for x in answers]


def save_json(path, value, volume):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)
    volume.commit()


@torch.no_grad()
def causal_probes(model, tok, folder, volume, deadline, quick=False):
    """Choose a head/column on four discovery entities; test on four different ones."""
    model.eval()
    result = {
        "status": "running",
        "selection": [],
        "rows": [],
        "qualification": [],
        "note": "Log odds of first single-token alternative. Native state swaps; no SAE.",
    }
    path = folder / "causal.json"
    for task in TASKS:
        for layer in LAYERS:
            if time.time() > deadline - 120:
                result["status"] = "deadline"
                save_json(path, result, volume)
                return result
            differences = []
            for index in range(4):
                pair, answers = prompts(tok, task, index, 0)
                states = [score(model, ids, answers, layer, boundary)[1] for ids, boundary in pair]
                differences.append(states[0] - states[1])
            energy = torch.stack(differences).square().mean(0)[0]
            head = int(energy.sum((1, 2)).argmax())
            column = int(energy[head].sum(0).argmax())
            result["selection"].append(
                {"task": task, "layer": layer, "head": head, "column": column}
            )
            # Native-kernel/ref and self-patch checks precede interpretation.
            pair, answers = prompts(tok, task, 4, 0)
            ids, boundary = pair[0]
            native, _ = score(model, ids, answers)
            ref, own = score(model, ids, answers, layer, boundary)
            self_score, _ = score(model, ids, answers, layer, boundary, lambda s, own=own: own)
            check = {
                "task": task,
                "layer": layer,
                "native_ref_logodds_error": abs(native - ref),
                "self_patch_error": abs(ref - self_score),
            }
            result["qualification"].append(check)
            assert abs(native - ref) < 0.08 and abs(ref - self_score) < 1e-5, check
            for index in range(4, 6 if quick else 8):
                for lag in [0] if quick else [0, 32, 128]:
                    if time.time() > deadline - 120:
                        result["status"] = "deadline"
                        save_json(path, result, volume)
                        return result
                    pair, answers = prompts(tok, task, index, lag)
                    base = [score(model, ids, answers, layer, boundary) for ids, boundary in pair]
                    for direction in (0, 1):
                        donor, target = base[direction], base[1 - direction]
                        ids, boundary = pair[1 - direction]
                        delta = donor[1] - target[1]
                        generator = torch.Generator(device="cuda").manual_seed(1000 + index + lag)
                        noise = torch.randn(delta.shape, generator=generator, device="cuda")
                        noise *= delta.norm() / noise.norm().clamp_min(1e-8)

                        def patch(s, mode, delta=delta, noise=noise, head=head, column=column):
                            s = s.clone()
                            if mode == "full_state":
                                return s + delta
                            if mode == "random_direction":
                                return s + noise
                            if mode == "head":
                                s[:, head] += delta[:, head]
                            if mode == "other_head":
                                h = (head + 1) % s.shape[1]
                                s[:, h] += delta[:, h]
                            if mode == "column":
                                s[:, head, :, column] += delta[:, head, :, column]
                            if mode == "zero_head":
                                s[:, head] = 0
                            return s

                        for mode in (
                            "full_state",
                            "head",
                            "column",
                            "other_head",
                            "random_direction",
                            "zero_head",
                        ):
                            changed, _ = score(
                                model,
                                ids,
                                answers,
                                layer,
                                boundary,
                                lambda s, mode=mode: patch(s, mode),
                            )
                            sign = 1 if direction == 0 else -1
                            result["rows"].append(
                                {
                                    "task": task,
                                    "entity": index,
                                    "lag": lag,
                                    "layer": layer,
                                    "direction": direction,
                                    "mode": mode,
                                    "donor_logodds": donor[0],
                                    "target_logodds": target[0],
                                    "patched_logodds": changed,
                                    "oriented_effect": sign * (changed - target[0]),
                                    "clean_gap": sign * (donor[0] - target[0]),
                                    "donor_prefers_correct": sign * donor[0] > 0,
                                    "target_prefers_correct": sign * target[0] < 0,
                                }
                            )
                    save_json(path, result, volume)
    result["status"] = "complete"
    save_json(path, result, volume)
    return result


@torch.no_grad()
def atlas(model, tok, windows, folder, volume, deadline):
    """Positive AND negative examples of fixed native read coordinates, no cherry picking."""
    original = kda.kda_kernel
    layers = [i for i, b in enumerate(model.blocks) if isinstance(b.mixer, kda.KimiDeltaAttention)]
    records = {}
    model.eval()
    for number, (source, ids) in enumerate(windows):
        if time.time() > deadline - 120:
            break
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
            number=number,
        ):
            nonlocal calls
            layer = layers[calls]
            calls += 1
            out = original(q, k, v, g, beta, A, bias, documents, final_state)
            assert not final_state
            flat = out[0].float().flatten(1)
            for channel in torch.linspace(0, flat.shape[-1] - 1, 16).long().tolist():
                key = f"{layer}:{channel}"
                entry = records.setdefault(
                    key,
                    {
                        "layer": layer,
                        "head": channel // v.shape[-1],
                        "value_channel": channel % v.shape[-1],
                        "positive": [],
                        "negative": [],
                    },
                )
                for pole, sign in [("positive", 1), ("negative", -1)]:
                    _, positions = (sign * flat[:, channel]).topk(min(3, len(ids)))
                    for t in positions.tolist():
                        # Both sides are shown, but future tokens did not cause this activation.
                        entry[pole].append(
                            {
                                "activation": float(flat[t, channel]),
                                "source": source,
                                "window": number,
                                "position": t,
                                "prefix": tok.decode(ids[max(0, t - 32) : t + 1]),
                                "following": tok.decode(ids[t + 1 : t + 17]),
                            }
                        )
                    entry[pole] = sorted(
                        entry[pole], key=lambda r: sign * r["activation"], reverse=True
                    )[:6]
            return out

        kda.kda_kernel = kernel
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.hidden_states(torch.tensor([ids], device="cuda"))
        finally:
            kda.kda_kernel = original
    result = {
        "windows": number + 1,
        "coordinates": list(records.values()),
        "note": "Raw q-transpose-state read before output norm/gate. Signed native coordinates; descriptive, not semantic labels.",
    }
    save_json(folder / "atlas.json", result, volume)
    return result


@torch.no_grad()
def qualify_recurrence():
    """Check reference equivalence, suffix-only intervention, and document reset."""
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(913)
        shape = (1, 13, 2, 64)
        q, k, v, g = [torch.randn(shape, device="cuda") for _ in range(4)]
        beta = torch.randn(1, 13, 2, device="cuda")
        A = torch.zeros(2, device="cuda")
        bias = torch.full((128,), -4.0, device="cuda")
        docs = torch.zeros(1, 13, device="cuda", dtype=torch.long)
        docs[:, 9:] = 1
        args = (q, k, v, g, beta, A, bias, docs)
        out, state = recurrence(*args, 4)
        expected = kda.kda_reference(q, k, v, g, beta, A, bias, document_positions(docs))
        error = float((out - expected).abs().max())
        assert error < 1e-5, error
        patched, _ = recurrence(*args, 4, lambda s: s + 1)
        assert torch.equal(out[:, :5], patched[:, :5]), "Patch changed prefix outputs"
        assert not torch.equal(out[:, 5:9], patched[:, 5:9]), "Patch did not affect suffix"
        assert torch.equal(out[:, 9:], patched[:, 9:]), "Patch crossed document reset"
        noop, _ = recurrence(*args, 4, lambda s: state)
        assert torch.equal(out, noop)
        return {
            "max_reference_error": error,
            "prefix_unchanged": True,
            "document_reset": True,
            "self_patch": True,
        }
