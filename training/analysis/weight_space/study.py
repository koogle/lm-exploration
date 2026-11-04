"""Gradient-free, reversible sparse MLP interventions on a downloaded Qwen checkpoint.

Inputs are explicit, task-disjoint calibration/test prompts. The paired novel and
repeated contexts contain the same target command; their activation contrast is a
hypothesis about repetition, not a discovered causal feature until tested.
"""

import hashlib
import json
import shlex
import time
from difflib import SequenceMatcher
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(path):
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    tokenizer.pad_token_id = 151643
    model = (
        AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa")
        .cuda()
        .eval()
    )
    return model, tokenizer


def command_metrics(text, seen):
    from training_data.qwen_agent_chat import parse_assistant

    message = parse_assistant(text)
    calls = message.get("tool_calls", [])
    commands = []
    for call in calls:
        if call["function"]["name"] == "bash":
            args = json.loads(call["function"]["arguments"])
            if isinstance(args.get("command"), str):
                commands.append(args["command"])
    valid = len(commands) == 1 and len(calls) == 1
    repeated = valid and commands[0] in seen
    similarity = (
        max(
            (SequenceMatcher(None, commands[0], old, autojunk=False).ratio() for old in seen),
            default=0,
        )
        if valid
        else 0
    )
    internal_repeat = False
    if valid:
        try:
            lexer = shlex.shlex(commands[0], posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            parts, part = [], []
            for token in lexer:
                if token in (";", "&&", "||"):
                    if len(part) >= 3:
                        parts.append(tuple(part))
                    part = []
                else:
                    part.append(token)
            if len(part) >= 3:
                parts.append(tuple(part))
            internal_repeat = len(set(parts)) < len(parts)
        except ValueError:
            pass  # Shell syntax validity requires execution, especially for heredocs.
    near_repeat = valid and similarity >= 0.85
    return {
        "valid": valid,
        "repeated": repeated,
        "valid_novel": valid and not repeated,
        "near_repeat": near_repeat,
        "internal_repeat": internal_repeat,
        "valid_distinct": valid and not near_repeat and not internal_repeat,
        "max_command_similarity": similarity,
        "command": commands[0] if valid else None,
        "premature_submit": valid and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in commands[0],
    }


@torch.inference_mode()
def generate(model, tokenizer, examples, *, seed=2026, max_tokens=512, batch_size=8, on_batch=None):
    rows = []
    for begin in range(0, len(examples), batch_size):
        batch = examples[begin : begin + batch_size]
        inputs = tokenizer([x["prompt"] for x in batch], return_tensors="pt", padding=True).to(
            "cuda"
        )
        torch.manual_seed(seed + begin)
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_new_tokens=max_tokens,
            eos_token_id=[151645, 151643],
            pad_token_id=151643,
            repetition_penalty=1.0,
        )[:, inputs.input_ids.shape[1] :]
        for example, ids in zip(batch, outputs, strict=True):
            values = ids.tolist()
            ends = [i for i, x in enumerate(values) if x in (151643, 151645)]
            stopped = bool(ends)
            values = values[: ends[0]] if stopped else values
            text = tokenizer.decode(values, skip_special_tokens=False)
            grams = [tuple(values[i : i + 8]) for i in range(max(0, len(values) - 7))]
            rows.append(
                {
                    "id": example["id"],
                    "text": text,
                    "tokens": len(values),
                    "truncated": not stopped,
                    "repeated_8gram_fraction": 1 - len(set(grams)) / max(1, len(grams)),
                    **command_metrics(text, example["seen_commands"]),
                }
            )
        print(json.dumps({"generated": len(rows), "total": len(examples)}), flush=True)
        if on_batch is not None:
            on_batch(rows)
    return rows


def summary(rows):
    return {
        "n": len(rows),
        **{
            key: sum(r[key] for r in rows) / max(1, len(rows))
            for key in (
                "valid",
                "repeated",
                "valid_novel",
                "near_repeat",
                "internal_repeat",
                "valid_distinct",
                "premature_submit",
                "truncated",
                "tokens",
                "repeated_8gram_fraction",
            )
        },
    }


@torch.inference_mode()
def collect(model, tokenizer, examples):
    """Contrast neuron activity at matched command positions in paired contexts."""
    layers = model.model.layers
    stats = [{"loop": [], "novel": []} for _ in layers]
    current = {}
    handles = []
    for index, layer in enumerate(layers):

        def hook(module, args, output, index=index):
            # Full prompt plus a common command prefix: only the command's tokens.
            start = current["start"]
            values = args[0][0, start:].float()
            stats[index][current["kind"]].append(values.mean(0).cpu())

        handles.append(layer.mlp.down_proj.register_forward_hook(hook))
    try:
        for example in examples:
            target = tokenizer.encode(example["target"], add_special_tokens=False)[:128]
            for kind, key in (("loop", "prompt"), ("novel", "novel_prompt")):
                prefix = tokenizer.encode(example[key], add_special_tokens=False)
                current.update(kind=kind, start=len(prefix))
                ids = torch.tensor([prefix + target], device="cuda")
                model(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    proposals, audit = [], []
    for index, stat in enumerate(stats):
        loop = torch.stack(stat["loop"]).mean(0)
        novel = torch.stack(stat["novel"]).mean(0)
        w = layers[index].mlp.down_proj.weight.detach().float().cpu()
        delta = loop - novel
        direction = w @ delta
        direction /= direction.norm().clamp_min(1e-8)
        # Columns that write along the contrast more in repeated contexts.
        contribution = (direction @ w) * delta
        columns = contribution.topk(64).indices
        proposal = {"layer": index, "columns": columns, "direction": direction}
        proposals.append(proposal)
        audit.append(
            {
                "layer": index,
                "contrast_norm": (w @ delta).norm().item(),
                "selected_contribution": contribution[columns].sum().item(),
                "columns": columns.tolist(),
            }
        )
    return proposals, audit


@torch.inference_mode()
def attention_proposals(model, tokenizer, examples):
    """Rank heads by excess target-logit contribution in repeated contexts.

    Attribution is a hypothesis; causal ablations below test it separately.
    """
    layers = model.model.layers
    heads, width = model.config.num_attention_heads, model.config.head_dim
    stats = [{"loop": [], "novel": []} for _ in layers]
    current, handles = {}, []
    for index, layer in enumerate(layers):

        def hook(module, args, output, index=index):
            start, target = current["start"], current["target"]
            activations = args[0][0, start - 1 : start - 1 + len(target)].float()
            embedding = model.get_output_embeddings().weight[target].float()
            readout = embedding @ module.weight.float()
            attribution = (activations * readout).view(-1, heads, width).sum(-1).mean(0)
            stats[index][current["kind"]].append(attribution.cpu())

        handles.append(layer.self_attn.o_proj.register_forward_hook(hook))
    try:
        for example in examples:
            target = tokenizer.encode(example["target"], add_special_tokens=False)[:128]
            for kind, key in (("loop", "prompt"), ("novel", "novel_prompt")):
                prefix = tokenizer.encode(example[key], add_special_tokens=False)
                current.update(kind=kind, start=len(prefix), target=target)
                model(
                    input_ids=torch.tensor([prefix + target], device="cuda"),
                    use_cache=False,
                    logits_to_keep=1,
                )
    finally:
        for handle in handles:
            handle.remove()
    ranking = []
    for index, stat in enumerate(stats):
        loop = torch.stack(stat["loop"]).mean(0)
        novel = torch.stack(stat["novel"]).mean(0)
        for head in range(heads):
            ranking.append(
                {"layer": index, "head": head, "contrast": (loop[head] - novel[head]).item()}
            )
    ranking.sort(key=lambda x: -x["contrast"])
    proposals = [
        {
            "layer": r["layer"],
            "head": r["head"],
            "module": "attention",
            "columns": torch.arange(r["head"] * width, (r["head"] + 1) * width),
        }
        for r in ranking[:12]
    ]
    return proposals, ranking


@torch.inference_mode()
def apply(model, proposals, strength=1.0):
    originals = []
    for p in proposals:
        layer = model.model.layers[p["layer"]]
        w = (
            layer.self_attn.o_proj.weight
            if p.get("module") == "attention"
            else layer.mlp.down_proj.weight
        )
        columns = p["columns"].to(w.device)
        old = w[:, columns].clone()
        if p.get("module") == "attention":
            edited = (1 - strength) * old.float()
        else:
            direction = p["direction"].to(w.device).float()
            edited = (
                old.float() - strength * direction[:, None] * (direction @ old.float())[None, :]
            )
        w[:, columns] = edited.to(w.dtype)
        originals.append((w, columns, old))
    return originals


@torch.inference_mode()
def restore(originals):
    for w, columns, old in originals:
        w[:, columns] = old


@torch.inference_mode()
def likelihoods(model, tokenizer, examples):
    out = []
    for example in examples:
        target = tokenizer.encode(example["target"], add_special_tokens=False)[:128]
        row = {}
        for kind, key in (("loop", "prompt"), ("novel", "novel_prompt")):
            prefix = tokenizer.encode(example[key], add_special_tokens=False)
            inputs = torch.tensor([prefix + target], device="cuda")
            logits = (
                model(input_ids=inputs, use_cache=False, logits_to_keep=len(target) + 1)
                .logits[0, :-1]
                .float()
            )
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(target, device="cuda"))
            row[kind] = loss.item()
        out.append(row)
    return out


def run(model_path, prompts_path, output, deadline, phase="scan", selection=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")

    def check():
        if time.time() > deadline - 120:
            raise TimeoutError("Experiment deadline reached")

    import transformers

    with (Path(model_path) / "model.safetensors").open("rb") as handle:
        model_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    save(
        f"provenance-{phase}-{int(time.time())}.json",
        {
            "model_path": model_path,
            "model_sha256": model_hash,
            "prompts_sha256": hashlib.sha256(Path(prompts_path).read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(),
            "phase": phase,
            "selection": selection,
            "deadline_unix": deadline,
        },
    )
    model, tokenizer = load(model_path)
    examples = json.loads(Path(prompts_path).read_text())
    calibration = [x for x in examples if x["split"] == "calibration"]
    if phase in ("scan", "attention-scan"):
        check()
        if phase == "scan":
            baseline = generate(model, tokenizer, calibration)
            save("baseline-calibration.json", baseline)
            print(json.dumps({"baseline": summary(baseline)}), flush=True)
            proposals, audit = collect(model, tokenizer, calibration[:16])
            torch.save(proposals, output / "proposals.pt")
        else:
            baseline = json.loads((output / "baseline-calibration.json").read_text())
            proposals, audit = attention_proposals(model, tokenizer, calibration[:16])
            torch.save(proposals, output / "attention-proposals.pt")
        save(f"{phase}-activation-audit.json", audit)
        probes = calibration[16:24]
        base_scores = likelihoods(model, tokenizer, probes)
        save("likelihood-baseline.json", base_scores)
        ranking = []
        for index, proposal in enumerate(proposals):
            check()
            originals = apply(model, [proposal])
            try:
                scores = likelihoods(model, tokenizer, probes)
            finally:
                restore(originals)
            loop_delta = sum(
                x["loop"] - b["loop"] for x, b in zip(scores, base_scores, strict=True)
            ) / len(probes)
            novel_delta = sum(
                x["novel"] - b["novel"] for x, b in zip(scores, base_scores, strict=True)
            ) / len(probes)
            row = {
                "index": index,
                "layer": proposal["layer"],
                "head": proposal.get("head"),
                "loop_nll_delta": loop_delta,
                "novel_nll_delta": novel_delta,
                "selectivity": loop_delta - max(0, novel_delta),
                "scores": scores,
            }
            ranking.append(row)
            save(f"{phase}-ranking.json", ranking)
            print(json.dumps({k: v for k, v in row.items() if k != "scores"}), flush=True)
        return {
            "baseline": summary(baseline),
            "ranking": sorted(ranking, key=lambda x: -x["selectivity"]),
        }
    proposals = torch.load(output / "proposals.pt", weights_only=True)
    attention = (
        torch.load(output / "attention-proposals.pt", weights_only=True)
        if (output / "attention-proposals.pt").exists()
        else []
    )
    specs = selection["candidates"]
    selected_examples = (
        calibration if phase == "candidates" else [x for x in examples if x["split"] == "test"]
    )
    results = {}
    for spec in specs:
        check()
        originals = apply(model, [proposals[i] for i in spec["layers"]], spec.get("strength", 1.0))
        originals += apply(
            model,
            [attention[i] for i in spec.get("attention", [])],
            spec.get("attention_strength", 1.0),
        )
        try:
            rows = generate(
                model,
                tokenizer,
                selected_examples,
                seed=selection.get("seed", 2026),
                max_tokens=selection.get("max_tokens", 512),
                on_batch=lambda partial, name=spec["name"]: save(
                    f"{phase}-{name}.partial.json", partial
                ),
            )
            save(f"{phase}-{spec['name']}.json", rows)
            results[spec["name"]] = summary(rows)
            print(json.dumps({spec["name"]: results[spec["name"]]}), flush=True)
            if spec.get("export"):
                target = output / spec["export"]
                model.save_pretrained(target)
                tokenizer.save_pretrained(target)
                save("export-spec.json", spec)
        finally:
            restore(originals)
    save(f"{phase}-summary.json", results)
    return results
