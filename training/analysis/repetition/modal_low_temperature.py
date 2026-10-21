"""Bounded top-k sampling study using exact code from pinned model revisions.

Run from this directory: modal run --detach modal_low_temperature.py
No training updates or model-repository writes occur in this program.
"""

import json
from pathlib import Path

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.6.0",
    "numpy",
    "pydantic>=2.11,<3",
    "tokenizers==0.22.2",
    "huggingface-hub==1.33.0",
    "safetensors>=0.4",
)
app = modal.App("simple-llm-low-temperature-study")
volume = modal.Volume.from_name("lm-exploration-training")
REPO = "jfrick/scholarly-code-modern-gpt-124m-step2"
CHECKPOINTS = [
    (1000, "cdf3f3636fa7226a13a300bb18290b069fe82a29"),
    (16000, "6da1f3d69a55883edbfd4bcf30b1c53f184bd515"),
    (32000, "3b338c7a595cea4530028fa5527457a22c034dd2"),
    (64227, "38df193c2cc2c5463e4bb20a5c6a4998f58b3af0"),
]
PROMPTS = [
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


def repeated_4grams(ids):
    grams = [tuple(ids[i : i + 4]) for i in range(max(0, len(ids) - 3))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=8192,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def compare():
    import hashlib
    import sys
    import time

    import httpx
    import torch
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    torch.set_num_threads(4)
    started = time.monotonic()

    def download(*args, **kwargs):
        for attempt in range(4):
            try:
                return snapshot_download(*args, **kwargs)
            except (httpx.TransportError, OSError, HfHubHTTPError) as exc:
                if isinstance(exc, HfHubHTTPError) and exc.response.status_code not in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    raise
                if attempt == 3:
                    raise
                print(f"Retrying checkpoint download after {type(exc).__name__}", flush=True)
                time.sleep(2**attempt)
        raise RuntimeError("Unreachable download state")

    source = Path(
        download(
            REPO,
            revision=CHECKPOINTS[-1][1],
            allow_patterns=[
                "config.json",
                "provenance.json",
                "tokenizer/tokenizer.json",
                "model_100m/*.py",
                "model_10m/*.py",
            ],
        )
    )
    identity = json.loads((source / "provenance.json").read_text())
    source_hash = hashlib.sha256((source / "model_100m/model.py").read_bytes()).hexdigest()
    assert source_hash == identity["source_sha256"]["model"]
    # This is our own previously published, hash-verified code, not the current main architecture.
    sys.path.insert(0, str(source))
    from model_100m.model import GPT, ModelConfig

    config = json.loads((source / "config.json").read_text())
    model = GPT(ModelConfig(**config)).cuda().eval()
    tokenizer = Tokenizer.from_file(str(source / "tokenizer/tokenizer.json"))
    tokenizer_hash = hashlib.sha256((source / "tokenizer/tokenizer.json").read_bytes()).hexdigest()
    modes = [{"name": "greedy", "temperature": 0.0, "seed_offset": 1000}]
    for temperature in (0.1, 0.2, 0.4):
        for seed_offset in (1000, 2000):
            modes.append(
                {
                    "name": f"topk50_t{temperature}_seed{seed_offset}",
                    "temperature": temperature,
                    "seed_offset": seed_offset,
                }
            )
    output = Path("/runs/simple-llm-low-temperature-20261018.json")
    checkpoints = []
    previous_elapsed = 0.0
    report = None
    if output.exists():
        report = json.loads(output.read_text())
        assert report["prompts"] == PROMPTS and report["modes"] == modes
        assert (
            report["model_source_sha256"] == source_hash
            and report["tokenizer_sha256"] == tokenizer_hash
        )
        assert report["max_new_tokens"] == 128 and report["torch_version"] == str(torch.__version__)
        checkpoints = report["checkpoints"]
        assert all(
            c["revision"] == dict(CHECKPOINTS)[c["step"]]
            and len(c["samples"]) == len(PROMPTS) * len(modes)
            for c in checkpoints
        )
        previous_elapsed = report["elapsed_seconds"]
        print(json.dumps({"resumed_completed_checkpoints": len(checkpoints)}), flush=True)
    with torch.inference_mode():
        for step, revision in CHECKPOINTS:
            if any(c["step"] == step for c in checkpoints):
                continue
            folder = Path(
                download(
                    REPO, revision=revision, allow_patterns=["model.safetensors", "provenance.json"]
                )
            )
            provenance = json.loads((folder / "provenance.json").read_text())
            assert (
                provenance["step"] == step and provenance["source_sha256"]["model"] == source_hash
            )
            assert (
                hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest()
                == provenance["weights_sha256"]
            )
            model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert sum(p.numel() for p in model.parameters()) == 124389632
            samples = []
            for pi, prompt in enumerate(PROMPTS):
                prefix = [1, *tokenizer.encode(prompt, add_special_tokens=False).ids]
                for mode in modes:
                    torch.manual_seed(mode["seed_offset"] + pi)
                    tokens = torch.tensor([prefix], device="cuda")
                    generated = []
                    for _ in range(128):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(tokens)[0, -1].float()
                        if mode["temperature"] == 0:
                            token = int(logits.argmax())
                        else:
                            values, indices = torch.topk(logits / mode["temperature"], 50)
                            token = int(indices[torch.multinomial(values.softmax(-1), 1)])
                        generated.append(token)
                        tokens = torch.cat([tokens, torch.tensor([[token]], device="cuda")], dim=1)
                        if token == 2:
                            break
                    samples.append(
                        {
                            "prompt": prompt,
                            "mode": mode["name"],
                            "temperature": mode["temperature"],
                            "top_k": 50 if mode["temperature"] else 0,
                            "seed": mode["seed_offset"] + pi,
                            "generated_tokens": len(generated),
                            "ended_with_eos": generated[-1] == 2,
                            "repeated_4gram_fraction": repeated_4grams(generated),
                            "generation_ids": generated,
                            "completion": tokenizer.decode(generated, skip_special_tokens=True),
                        }
                    )
            summary = {}
            for temperature in (0.0, 0.1, 0.2, 0.4):
                selected = [s for s in samples if s["temperature"] == temperature]
                summary[str(temperature)] = {
                    "completions": len(selected),
                    "mean_repeated_4gram_fraction": sum(
                        s["repeated_4gram_fraction"] for s in selected
                    )
                    / len(selected),
                    "eos_completions": sum(s["ended_with_eos"] for s in selected),
                }
            checkpoints.append(
                {
                    "step": step,
                    "trained_tokens": provenance["tokens"],
                    "revision": revision,
                    "decoding_summary": summary,
                    "samples": samples,
                }
            )
            report = {
                "repo_id": REPO,
                "checkpoints": checkpoints,
                "prompts": PROMPTS,
                "modes": modes,
                "model_source_sha256": source_hash,
                "tokenizer_sha256": tokenizer_hash,
                "max_new_tokens": 128,
                "torch_version": str(torch.__version__),
                "device": torch.cuda.get_device_name(),
                "dtype": "BF16 autocast / FP32 weights",
                "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                "note": "Bounded ten-prompt repetition diagnostic, not a quality benchmark; all samples retained.",
            }
            temporary = output.with_suffix(".partial")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(output)
            volume.commit()
            print(
                json.dumps(
                    {"step": step, "trained_tokens": provenance["tokens"], "summary": summary}
                ),
                flush=True,
            )
    return report


@app.local_entrypoint()
def main(output: str = "../../artifacts/repetition/low-temperature-20261018.json"):
    report = compare.remote()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "checkpoints": len(report["checkpoints"]),
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
    )
