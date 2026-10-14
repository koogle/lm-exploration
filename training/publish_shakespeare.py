"""Export a trusted Shakespeare checkpoint without optimizer/RNG and publish it.

Run using a Python environment with torch, safetensors and huggingface-hub.
Authentication is read from HF_TOKEN, never from command flags or saved files.
"""

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi
from model_10m.model import GPT, ModelConfig
from safetensors.torch import save_file
from training_data.loading import CharacterTokenizer

from trainer.checkpoints import load_checkpoint
from trainer.sampling import sample

ROOT = Path(__file__).resolve().parent


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(checkpoint, report_path, output):
    state = load_checkpoint(checkpoint)
    report = json.loads(Path(report_path).read_text())
    if report["smoke"] or state["step"] != report["selected"]["step"]:
        raise ValueError("publication needs the validation-selected training checkpoint")
    model = GPT(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"], strict=True)
    count = sum(p.numel() for p in model.parameters())
    if count != 10_795_776:
        raise ValueError("unexpected model parameter count")
    if checksum(ROOT / "10m/model.py") != state["config"]["runtime"]["source_hashes"]["model"]:
        raise ValueError("architecture source differs from the executed training definition")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_file(
        {key: tensor.contiguous() for key, tensor in model.state_dict().items()},
        str(output / "model.safetensors"),
    )
    (output / "config.json").write_text(json.dumps(state["config"]["model"], indent=2) + "\n")
    (output / "tokenizer.json").write_text(json.dumps(state["tokenizer"], indent=2) + "\n")
    report["sample_checkpoint_step"] = report["final"]["step"]
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    provenance = {
        "source_checkpoint_sha256": checksum(checkpoint),
        "weights_sha256": checksum(output / "model.safetensors"),
        "architecture_sha256": checksum(ROOT / "10m/model.py"),
        "parameter_count": count,
        "step": state["step"],
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    torch.set_num_threads(4)
    selected_sample = sample(
        model, CharacterTokenizer(state["tokenizer"]["characters"]), "ROMEO:\n", 300, 1339
    )
    (output / "sample.txt").write_text(selected_sample + "\n")
    for filename in ("model.py", "generate.py"):
        shutil.copyfile(ROOT / "10m" / filename, output / filename)
    (output / "requirements.txt").write_text("torch==2.6.0\nsafetensors==0.8.0\n")
    val = report["selected"]["val_loss"]
    model_card = f"""---
language: en
tags:
- text-generation
- character-level
- vanilla-gpt
- pytorch
- from-scratch
---

# Shakespeare Vanilla GPT 10M

A **{count:,}-parameter** character-level vanilla GPT, trained from random
initialization on Tiny Shakespeare using one NVIDIA L4 on Modal. No pretrained
weights, instruction tuning or diffusion are used. Complete architecture definition
is in `model.py`, with its exact dimensions in `config.json`.

## Definition

- 6 decoder blocks; embedding width 384; 6 attention heads.
- Context: 256 characters; vocabulary: 65 characters.
- Learned token and position embeddings; causal dense multi-head attention.
- Pre-LayerNorm residual blocks; GELU MLP with 4x expansion; final LayerNorm.
- Separate input embeddings and output projection; FP32 parameters.

This is a small learning benchmark, not a general assistant. It generates
Shakespeare-like text and can repeat or produce nonsensical passages.

## Data and training

Tiny Shakespeare from the char-rnn repository, immutable source revision
`370cbcd448eb7daf32f21a6be560b70e0b33c4e3`. Dataset SHA256:
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
The first 90% of characters are training data and the last 10% validation;
windows never cross that boundary. Vocabulary comes from the full corpus.
Shakespeare's works are public domain; the compilation comes from the
[MIT-licensed char-rnn repository](https://github.com/karpathy/char-rnn).

AdamW, learning rate 0.0003, betas (0.9, 0.95), weight decay 0.1, clipping norm 1,
batch size 32, context 256, seed 1337. A bounded run targets at most 4,000 updates
or 1,200 seconds of training, with checkpoint/evaluation every 200 updates.
Validation uses 20 fixed-seed batches and chooses among saved checkpoints.

Selected update: **{state["step"]}**, validation cross-entropy:
**{val:.4f} nats/character** (character perplexity **{math.exp(val):.3f}**).
Selected token exposures: **{report["selected"]["tokens_seen"]:,}** characters,
including repeated sampling of the small corpus. Final executed update:
**{report["final"]["step"]}**. This is one seed, a contiguous split and no sealed
test; the validation score was used for selection and is not a test score.
See `training_report.json` for all settings, runtime and measurements.

## Usage

Download a pinned repository revision, install `requirements.txt`, then from
that downloaded folder run:

```bash
python generate.py --prompt 'ROMEO:' --tokens 300
```

```python
from generate import load_model, generate
model, characters = load_model(".")
print(generate(model, characters, "ROMEO:\\n", tokens=300, seed=1339))
```

`model.safetensors` contains only model weights, with no optimizer or RNG state.
`tokenizer.json` records the exact character ordering. CPU, CUDA and MPS are
supported by the standalone sampler. Prompt characters must belong to the
vocabulary. Sampling uses the full softmax at temperature 1.
This custom PyTorch definition does not use `AutoModelForCausalLM`.

## Provenance

[Source project](https://github.com/koogle/lm-exploration/tree/main/training/10m).
Weight, checkpoint and definition checksums are in `provenance.json`.
No project-wide license for these original weights/code has been specified.
"""
    (output / "README.md").write_text(model_card)
    return output


def publish(folder, repo_id):
    files = [
        "model.safetensors",
        "sample.txt",
        "model.py",
        "generate.py",
        "config.json",
        "tokenizer.json",
        "training_report.json",
        "provenance.json",
        "README.md",
        "requirements.txt",
    ]
    if any(not (Path(folder) / name).is_file() for name in files):
        raise ValueError("incomplete inference export")
    api = HfApi()
    api.whoami()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    if api.model_info(repo_id).private:
        raise ValueError("destination must already be public or be a new repository")
    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=folder,
        allow_patterns=files,
        commit_message="Publish trained Shakespeare vanilla 10M with definition",
    )
    return {"repo_id": repo_id, "revision": commit.oid, "url": f"https://huggingface.co/{repo_id}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    output = export(args.checkpoint, args.report, args.output)
    print(json.dumps(publish(output, args.repo_id) if args.repo_id else {"folder": str(output)}))
