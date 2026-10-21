"""Compare four published checkpoints on fixed held-out text and reference continuations.

Generation from the same checkpoints is the recipe analysis/repetition/previous-comparison.json
for sampling.modal_sample; the archived result combines both parts.
"""

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

    answers = [
        ("The solar system consists of", " eight planets"),
        (
            "In mathematics, a prime number is",
            " an integer greater than 1 with exactly two positive divisors",
        ),
        ("The capital of France is", " Paris"),
        ("2 + 2 =", " 4"),
        ("def add(a, b):\n    ", "return a + b"),
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
        report = {
            "step": expected_step,
            "trained_tokens": p["tokens"],
            "revision": revision,
            "heldout": heldout,
            "reference_continuation_scores": answer_scores,
        }
        reports.append(report)
        print(
            json.dumps(
                {
                    "step": expected_step,
                    "tokens": p["tokens"],
                    "heldout_loss": heldout["mixture_weighted_loss"],
                }
            ),
            flush=True,
        )
        interim = {
            "random_initialization": random_eval,
            "checkpoints": reports,
            "evaluation_context": 2048,
            "note": "Bounded diagnostic evaluation, not a sealed benchmark. Reference prompts are manually chosen and not a standardized factual test.",
        }
        (Path("/runs") / "simple-llm-repetition-heldout.json").write_text(
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
    (Path("/runs") / "simple-llm-repetition-heldout.json").write_text(
        json.dumps(interim, indent=2) + "\n"
    )
    run_volume.commit()
    return interim


@app.local_entrypoint()
def main(output: str = "artifacts/results/simple-llm-repetition-heldout.json"):
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
