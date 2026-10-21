"""Bounded inference of the published final Simple LLM 124M checkpoint."""

import json
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "numpy",
        "pydantic>=2.11,<3",
        "tokenizers==0.22.2",
        "huggingface-hub==1.33.0",
        "safetensors>=0.4",
    )
    .env({"PYTHONPATH": "/project"})
    .add_local_dir("100m", "/project/model_100m")
    .add_local_dir("10m", "/project/model_10m")
)
app = modal.App("simple-llm-124m-inference")
volume = modal.Volume.from_name("lm-exploration-training")
REPO = "jfrick/scholarly-code-modern-gpt-124m-step2"
REVISION = "2c8a078d3510349746abfe32b322898babb26e0a"


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=8192,
    timeout=900,
    retries=0,
    max_containers=1,
    volumes={"/runs": volume},
)
def infer():
    import hashlib
    import time

    import model_100m.model
    import torch
    from huggingface_hub import snapshot_download
    from model_100m.model import GPT, ModelConfig
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    torch.set_num_threads(4)
    root = Path(
        snapshot_download(
            REPO,
            revision=REVISION,
            allow_patterns=[
                "config.json",
                "provenance.json",
                "model.safetensors",
                "tokenizer/tokenizer.json",
            ],
        )
    )
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["tokens"] == 4999981176 and provenance["step"] == 64227
    assert digest(root / "model.safetensors") == provenance["weights_sha256"]
    assert digest(model_100m.model.__file__) == provenance["source_sha256"]["model"]
    model = GPT(ModelConfig(**json.loads((root / "config.json").read_text())))
    model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model = model.cuda().eval()
    tokenizer = Tokenizer.from_file(str(root / "tokenizer/tokenizer.json"))
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
    samples = []
    started = time.monotonic()
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            encoded = [1] + tokenizer.encode(prompt, add_special_tokens=False).ids
            for mode in ("greedy", "sampled"):
                torch.manual_seed(1000 + i)
                tokens = torch.tensor([encoded], device="cuda")
                generated = []
                for _ in range(128):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(tokens[:, -8192:])[0, -1].float()
                    if mode == "greedy":
                        next_token = int(logits.argmax())
                    else:
                        values, indices = torch.topk(logits / 0.8, 50)
                        next_token = int(indices[torch.multinomial(values.softmax(-1), 1)])
                    generated.append(next_token)
                    tokens = torch.cat([tokens, torch.tensor([[next_token]], device="cuda")], dim=1)
                    if next_token == 2:
                        break
                sample = {
                    "prompt": prompt,
                    "mode": mode,
                    "seed": 1000 + i,
                    "temperature": 0 if mode == "greedy" else 0.8,
                    "top_k": 0 if mode == "greedy" else 50,
                    "completion": tokenizer.decode(generated, skip_special_tokens=True),
                    "generated_tokens": len(generated),
                }
                samples.append(sample)
                print(json.dumps(sample), flush=True)
    report = {
        "repo_id": REPO,
        "revision": REVISION,
        "parameters": 124389632,
        "trained_tokens": 4999981176,
        "step": 64227,
        "context": 8192,
        "torch_version": str(torch.__version__),
        "device": torch.cuda.get_device_name(),
        "dtype": "BF16 autocast with FP32 weights",
        "max_new_tokens": 128,
        "elapsed_seconds": time.monotonic() - started,
        "note": "Unfiltered fixed-prompt samples, not a downstream benchmark; no instruction tuning.",
        "samples": samples,
    }
    (Path("/runs") / "simple-llm-124m-8k-inference-20261018.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    report = infer.remote()
    path = Path("results/simple-llm-124m-8k-inference-20261018.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(path), "samples": len(report["samples"])}))
