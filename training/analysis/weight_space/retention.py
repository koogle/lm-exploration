"""Teacher-forced retention on fixed spans from successful training sources."""

import gc
import hashlib
import json
import time
from pathlib import Path

from .investigate import file_sha


def run(config, deadline):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    root = Path("/runs/agent-mix/scale-500k")
    manifest = json.loads((root / "manifest.json").read_text())
    offsets = np.load(root / "offsets.npy")
    tokens = np.memmap(root / "tokens.u32", dtype=np.uint32, mode="r")
    masks = np.memmap(root / "mask.u8", dtype=np.uint8, mode="r")
    rng = np.random.default_rng(20261103)
    samples = []
    # These converters select resolved/reward=1 source trajectories. This is
    # explicitly a training-retention diagnostic, not held-out validation.
    for source in ("open-swe", "pr-glm", "terminal-lego"):
        begin, end = manifest["counts"][source]["stored"]
        selected = 0
        for index in rng.permutation(np.arange(begin, end)):
            start, stop = offsets[index : index + 2]
            ids = tokens[start:stop]
            mask = masks[start:stop]
            edges = np.diff(np.r_[0, mask, 0].astype(np.int16))
            spans = [
                (int(lo), int(hi))
                for lo, hi in zip(
                    np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
                )
                if lo > 0 and 32 <= hi - lo <= 1536 and hi <= 16384
            ]
            if not spans:
                continue
            lo, hi = spans[int(rng.integers(len(spans)))]
            sample_ids = ids[:hi].astype(np.int64)
            assert int(sample_ids[-1]) == 151645
            samples.append(
                {
                    "source": source,
                    "index": int(index),
                    "lo": lo,
                    "hi": hi,
                    "ids": sample_ids,
                    "tokens_sha256": hashlib.sha256(sample_ids.tobytes()).hexdigest(),
                }
            )
            selected += 1
            if selected == 64:
                break
        if selected != 64:
            raise ValueError(f"Insufficient eligible retention examples: {source}")
    out = Path("/runs/weight-space", config["experiment"], "retention")
    out.mkdir(parents=True, exist_ok=False)
    (out / "selection.json").write_text(
        json.dumps([{k: v for k, v in s.items() if k != "ids"} for s in samples], indent=2)
    )
    results = []
    for variant in config["variants"]:
        path = Path("/runs/agentic-reference", variant["model_run"], "model")
        assert file_sha(path / "model.safetensors") == variant["sha256"]
        model = (
            AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .to("cuda")
            .eval()
        )
        rows = []
        with torch.inference_mode():
            for sample in samples:
                if time.time() >= deadline - 120:
                    raise TimeoutError("Retention diagnostic deadline")
                ids = torch.tensor(sample["ids"], device="cuda")[None]
                # Evaluate exactly the supervised span with its full preceding
                # history. Materialize vocabulary logits only for target tokens.
                hidden = model.model(ids[:, :-1], use_cache=False).last_hidden_state
                target_hidden = hidden[0, sample["lo"] - 1 : sample["hi"] - 1]
                losses = []
                for begin in range(0, len(target_hidden), 128):
                    logits = model.lm_head(target_hidden[begin : begin + 128]).float()
                    target = ids[0, sample["lo"] + begin : sample["lo"] + begin + 128]
                    losses.append(
                        torch.nn.functional.cross_entropy(logits, target, reduction="none")
                    )
                nll = torch.cat(losses)
                rows.append(
                    {k: v for k, v in sample.items() if k != "ids"}
                    | {
                        "target_tokens": len(nll),
                        "nll_sum": nll.sum().item(),
                        "nll_mean": nll.mean().item(),
                        "end_token_nll": nll[-1].item(),
                    }
                )
                if len(rows) % 32 == 0:
                    print(json.dumps({"variant": variant["name"], "scored": len(rows)}), flush=True)
        assert file_sha(path / "model.safetensors") == variant["sha256"]
        result = {"variant": variant["name"], "rows": rows, "weights_unchanged": True}
        (out / f"{variant['name']}.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        del model, hidden, target_hidden, logits, target, ids, losses, nll
        gc.collect()
        torch.cuda.empty_cache()
    return results
