"""Compare four published checkpoints on fixed held-out text and generation."""

import json
from pathlib import Path

import modal

from distributed.image import project_image

image = project_image("huggingface-hub==1.33.0", "safetensors>=0.4")
app = modal.App("simple-llm-repetition-validation")
run_volume = modal.Volume.from_name("lm-exploration-training")
data_volume = modal.Volume.from_name("lm-exploration-mixture-data")
REPO = "jfrick/scholarly-code-modern-gpt-124m-step2"
CHECKPOINTS = [
    (1000, "cdf3f3636fa7226a13a300bb18290b069fe82a29"),
    (16000, "6da1f3d69a55883edbfd4bcf30b1c53f184bd515"),
    (32000, "3b338c7a595cea4530028fa5527457a22c034dd2"),
    (64227, "38df193c2cc2c5463e4bb20a5c6a4998f58b3af0"),
]


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=16384,
    timeout=2400,
    retries=0,
    max_containers=1,
    volumes={"/data": data_volume, "/runs": run_volume},
)
def compare():
    import hashlib
    import sqlite3
    import time
    import zlib

    import model_100m.model
    import torch
    from huggingface_hub import snapshot_download
    from model_100m.model import GPT, ModelConfig
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from torch.nn import functional as F

    torch.set_num_threads(4)
    started = time.monotonic()
    folder = Path(
        "/data/scholarly-code/scholarly-small-5b-raw-20261017/snapshots/scholarly-modern-10m-5b-20261017"
    )
    manifest = json.loads((folder / "manifest.json").read_text())
    db = sqlite3.connect(f"{(folder / 'raw.sqlite').as_uri()}?mode=ro&immutable=1", uri=True)
    finalroot = Path(
        snapshot_download(
            REPO,
            revision=CHECKPOINTS[-1][1],
            allow_patterns=["tokenizer/tokenizer.json", "config.json"],
        )
    )
    tokenizer = Tokenizer.from_file(str(finalroot / "tokenizer/tokenizer.json"))
    assert (
        hashlib.sha256((finalroot / "tokenizer/tokenizer.json").read_bytes()).hexdigest()
        == manifest["tokenizer_sha256"]
    )
    blocks = {}
    parents_by_source = {}
    for name in manifest["sources"]:
        seen = set()
        selected = []
        for key, parent, raw in db.execute(
            'SELECT hash,parent,text FROM documents WHERE source=? AND split="validation" AND conflict=0 ORDER BY hash',
            (name,),
        ):
            if parent in seen:
                continue
            seen.add(parent)
            ids = [
                1,
                *tokenizer.encode(zlib.decompress(raw).decode(), add_special_tokens=False).ids,
                2,
            ]
            for offset in range(0, min(len(ids) - 1, 4 * 2048), 2048):
                x = ids[offset : offset + 2048]
                y = ids[offset + 1 : offset + 2049]
                n = len(y)
                selected.append((x[:n] + [0] * (2048 - n), y + [-100] * (2048 - n), n))
            if len(seen) >= 32:
                break
        blocks[name] = selected
        parents_by_source[name] = len(seen)
        print(
            json.dumps({"validation_source": name, "parents": len(seen), "windows": len(selected)}),
            flush=True,
        )
    db.close()
    torch.manual_seed(42)
    model = GPT(ModelConfig(**json.loads((finalroot / "config.json").read_text()))).cuda().eval()

    @torch.inference_mode()
    def evaluate():
        scores = {}
        for name, items in blocks.items():
            loss_sum = 0.0
            tokens = 0
            for i in range(0, len(items), 4):
                chunk = items[i : i + 4]
                x = torch.tensor([b[0] for b in chunk], device="cuda")
                y = torch.tensor([b[1] for b in chunk], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(
                        model(x).flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum"
                    )
                loss_sum += float(loss)
                tokens += sum(b[2] for b in chunk)
            scores[name] = {
                "tokens": tokens,
                "loss": loss_sum / tokens,
                "parents": parents_by_source[name],
            }
        return {
            "per_source": scores,
            "mixture_weighted_loss": sum(
                s["loss"] * manifest["sources"][n]["weight_percent"] / 100
                for n, s in scores.items()
            ),
            "targets": sum(s["tokens"] for s in scores.values()),
        }

    def rep4(ids):
        grams = [tuple(ids[i : i + 4]) for i in range(max(0, len(ids) - 3))]
        return 1 - len(set(grams)) / len(grams) if grams else 0.0

    prompts = [
        "The solar system consists of",
        "In mathematics, a prime number is",
        "The purpose of an experiment is to",
        "Once upon a time,",
        "def add(a, b):\n    ",
        "def factorial(n):\n    ",
        "import json\n\ndef read_config(path):\n    ",
        "The capital of France is",
        "2 + 2 =",
        "A language model predicts",
    ]
    modes = [
        {"name": "greedy", "temperature": 0, "seed_offset": 1000},
        {"name": "sampled_08_seed1", "temperature": 0.8, "top_k": 50, "seed_offset": 1000},
        {"name": "sampled_08_seed2", "temperature": 0.8, "top_k": 50, "seed_offset": 2000},
        {"name": "sampled_10_topp95", "temperature": 1.0, "top_p": 0.95, "seed_offset": 1000},
        {"name": "greedy_penalty11", "temperature": 0, "penalty": 1.1, "seed_offset": 1000},
    ]
    answers = [
        (prompts[0], " eight planets"),
        (prompts[1], " an integer greater than 1 with exactly two positive divisors"),
        (prompts[7], " Paris"),
        (prompts[8], " 4"),
        (prompts[4], "return a + b"),
    ]
    random_eval = evaluate()
    print(
        json.dumps({"random_initialization_loss": random_eval["mixture_weighted_loss"]}), flush=True
    )
    reports = []
    for expected_step, revision in CHECKPOINTS:
        root = Path(
            snapshot_download(
                REPO, revision=revision, allow_patterns=["provenance.json", "model.safetensors"]
            )
        )
        p = json.loads((root / "provenance.json").read_text())
        assert p["step"] == expected_step
        assert (
            hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest()
            == p["weights_sha256"]
        )
        assert (
            hashlib.sha256(Path(model_100m.model.__file__).read_bytes()).hexdigest()
            == p["source_sha256"]["model"]
        )
        model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
        assert all(torch.isfinite(t).all() for t in model.parameters())
        heldout = evaluate()
        samples = []
        answer_scores = []
        with torch.inference_mode():
            for prompt, answer in answers:
                prefix = [1, *tokenizer.encode(prompt, add_special_tokens=False).ids]
                suffix = tokenizer.encode(answer, add_special_tokens=False).ids
                x = torch.tensor([prefix + suffix[:-1]], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)[0].float()
                relevant = logits[len(prefix) - 1 :]
                loss = F.cross_entropy(
                    relevant, torch.tensor(suffix, device="cuda"), reduction="mean"
                )
                first = logits[len(prefix) - 1]
                top = torch.topk(first, 5).indices.tolist()
                answer_scores.append(
                    {
                        "prompt": prompt,
                        "reference_continuation": answer,
                        "mean_reference_nll": float(loss),
                        "first_reference_token_rank": int((first > first[suffix[0]]).sum()) + 1,
                        "top5_next_tokens": [tokenizer.decode([t]) for t in top],
                    }
                )
            for pi, prompt in enumerate(prompts):
                prefix = [1, *tokenizer.encode(prompt, add_special_tokens=False).ids]
                for mode in modes:
                    torch.manual_seed(mode["seed_offset"] + pi)
                    tokens = torch.tensor([prefix], device="cuda")
                    generated = []
                    for _ in range(128):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(tokens)[0, -1].float()
                        if mode.get("penalty"):
                            ids = tokens.unique()
                            old = logits[ids]
                            logits[ids] = torch.where(
                                old < 0, old * mode["penalty"], old / mode["penalty"]
                            )
                        if not mode["temperature"]:
                            token = int(logits.argmax())
                        else:
                            scaled = logits / mode["temperature"]
                            if mode.get("top_k"):
                                values, indices = torch.topk(scaled, mode["top_k"])
                            else:
                                values, indices = torch.sort(scaled, descending=True)
                                prob = values.softmax(-1)
                                values[(prob.cumsum(-1) - prob) > mode["top_p"]] = -float("inf")
                            token = int(indices[torch.multinomial(values.softmax(-1), 1)])
                        generated.append(token)
                        tokens = torch.cat([tokens, torch.tensor([[token]], device="cuda")], dim=1)
                        if token == 2:
                            break
                    sample = {
                        "prompt": prompt,
                        "mode": mode["name"],
                        "seed": mode["seed_offset"] + pi,
                        "completion": tokenizer.decode(generated, skip_special_tokens=True),
                        "generated_tokens": len(generated),
                        "ended_with_eos": generated[-1] == 2,
                        "repeated_4gram_fraction": rep4(generated),
                        "generation_ids": generated,
                    }
                    samples.append(sample)
            aggregate = {
                mode["name"]: {
                    "mean_repeated_4gram_fraction": sum(
                        s["repeated_4gram_fraction"] for s in samples if s["mode"] == mode["name"]
                    )
                    / len(prompts),
                    "eos_completions": sum(
                        s["ended_with_eos"] for s in samples if s["mode"] == mode["name"]
                    ),
                }
                for mode in modes
            }
        report = {
            "step": expected_step,
            "trained_tokens": p["tokens"],
            "revision": revision,
            "heldout": heldout,
            "decoding_summary": aggregate,
            "reference_continuation_scores": answer_scores,
            "samples": samples,
        }
        reports.append(report)
        print(
            json.dumps(
                {
                    "step": expected_step,
                    "tokens": p["tokens"],
                    "heldout_loss": heldout["mixture_weighted_loss"],
                    "decoding_summary": aggregate,
                }
            ),
            flush=True,
        )
        interim = {
            "random_initialization": random_eval,
            "checkpoints": reports,
            "modes": modes,
            "prompts": prompts,
            "evaluation_context": 2048,
            "generation_max_new_tokens": 128,
            "note": "Bounded diagnostic evaluation, not a sealed benchmark. Reference prompts are manually chosen and not a standardized factual test.",
        }
        (Path("/runs") / "simple-llm-repetition-comparison-20261018.json").write_text(
            json.dumps(interim, indent=2) + "\n"
        )
        run_volume.commit()
    interim.update(
        elapsed_seconds=time.monotonic() - started,
        all_checkpoints_verified=True,
        all_checkpoints_finite=True,
        validation_parent_count=sum(parents_by_source.values()),
        torch_version=str(torch.__version__),
        device=torch.cuda.get_device_name(),
    )
    (Path("/runs") / "simple-llm-repetition-comparison-20261018.json").write_text(
        json.dumps(interim, indent=2) + "\n"
    )
    run_volume.commit()
    return interim


@app.local_entrypoint()
def main(output: str = "artifacts/pr-16/results/simple-llm-repetition-comparison-20261018.json"):
    r = compare.remote()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=2) + "\n")
    print(
        json.dumps(
            {
                "completed_checkpoints": len(r["checkpoints"]),
                "elapsed_seconds": r["elapsed_seconds"],
            }
        )
    )
