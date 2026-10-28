"""Equal-target natural context tests and explicit associative-recall probes."""

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .autoresearch_model import build
from .model import rope_tables
from .prepared_data import heldout_index, shard_paths


def extend_context(model, length):
    model.config = replace(model.config, block_size=length)
    cos, sin = rope_tables(
        length, model.config.head_dim, model.config.rope_base, model.token_embedding.weight.device
    )
    model.rope_cos, model.rope_sin = cos, sin


def evaluate_checkpoint(checkpoint, recipe, deadline, max_context=32768):
    torch.set_num_threads(4)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build(ck["config"], ck["variant"], ck["seed"]).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    extend_context(model, max_context)
    result = {
        "variant": ck["variant"],
        "seed": ck["seed"],
        "step": ck["step"],
        "trained_context": ck["config"]["block_size"],
        "natural": [],
        "retrieval": [],
        "note": "Extrapolation evaluation; these checkpoints were trained at 4K.",
    }
    del ck
    lengths = [n for n in (4096, 8192, 16384, 32768, 131072) if n <= max_context]

    def score(ids, target_start):
        x = torch.as_tensor(ids[:-1][None], device="cuda", dtype=torch.long)
        y = torch.as_tensor(ids[1:][None], device="cuda", dtype=torch.long)
        y[:, : max(0, target_start - 1)] = -100
        torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, y)
        torch.cuda.synchronize()
        return {
            "ce": out["ce"].item() / out["count"].item(),
            "count": out["count"].item(),
            "seconds": time.perf_counter() - tick,
        }

    root = Path("/data") / recipe["name"]
    for name, source in recipe["sources"].items():
        binpath, docpath, _ = shard_paths(root, name, heldout_index(source))
        sizes = np.load(docpath)[:, 0]
        offsets = np.concatenate(([0], np.cumsum(sizes)))
        tokens = np.fromfile(binpath, dtype=np.uint32)
        # Selection starts beyond the prefix used for training-time validation.
        eligible = np.flatnonzero((sizes >= max_context + 1) & (offsets[:-1] > 2_000_000))
        rng = np.random.default_rng(9124)
        chosen = rng.choice(eligible, size=min(2, len(eligible)), replace=False)
        if not len(chosen):
            result["natural"].append({"source": name, "status": "no_eligible_documents"})
        for index in chosen:
            end = int(offsets[index] + max_context + 1)
            for length in lengths:
                if time.time() > deadline - 180:
                    result["status"] = "deadline"
                    return result
                ids = tokens[end - length - 1 : end].astype(np.int64)
                assert not np.any(ids[:-1] == model.config.eos_id)
                try:
                    row = score(ids, len(ids) - 256)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row = {"status": "oom"}
                result["natural"].append(
                    {
                        "source": name,
                        "document": int(index),
                        "context": length,
                        "target_end": end,
                        "target_tokens": 256,
                        **row,
                    }
                )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file("/project/glm-tokenizer.json")
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
    filler = enc(" This paragraph is unrelated to the stored records. Trees grow near a river.")
    for length in lengths:
        for task in ("single", "many", "update", "order"):
            if time.time() > deadline - 180:
                result["status"] = "deadline"
                return result
            count = 32 if task == "many" else 4
            records = "".join(
                f"key_{i:03d} = value_{(i * 17 + 11) % 997:03d}\n" for i in range(count)
            )
            target = "value_011"
            query = "\nkey_000 ="
            suffix = ""
            if task == "update":
                suffix = "\nkey_000 = value_893\n"
                target = "value_893"
            if task == "order":
                records = "event_000 = value_321\nevent_001 = value_654\n"
                query = "\nThe value of the first event is"
                target = "value_321"
            prefix = enc(
                "Remember these records. Use the most recent assignment for each key.\n" + records
            )
            tail = enc(suffix + query)
            answer = enc(" " + target)
            remaining = length + 1 - len(prefix) - len(tail) - len(answer)
            if remaining < 0:
                continue
            ids = np.asarray(
                prefix + (filler * (remaining // len(filler) + 1))[:remaining] + tail + answer,
                dtype=np.int64,
            )
            try:
                row = score(ids, len(ids) - len(answer))
                # Teacher-forced token accuracy is explicitly not free-generation accuracy.
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model.hidden_states(torch.as_tensor(ids[:-1][None], device="cuda"))
                    logits = torch.nn.functional.linear(
                        hidden[:, -len(answer) :], model.output_weight
                    )
                    predictions = logits.argmax(-1).flatten().tolist()
                row["teacher_forced_exact"] = predictions == answer
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row = {"status": "oom"}
            result["retrieval"].append(
                {"task": task, "context": length, "facts": count, "answer": target, **row}
            )
    result["status"] = "complete"
    result["limitations"] = (
        "Four diagnostic prompts per length; not a statistically powered retrieval benchmark."
    )
    return result
