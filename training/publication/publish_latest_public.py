"""Publish an explicitly authorized completed inference model; private archives stay private."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

SOURCE = "jfrick/124M-10.879B"
TARGET = "jfrick/simple-llm-124M-10.879B"
WEIGHTS_SHA256 = "f50314c30b42b2c871cdec6aab4796109ca5ac11e1f2e055e795ce97e1e469cc"
FILES = [
    "model.safetensors",
    "config.json",
    "generate.py",
    "requirements.txt",
    "model_100m/__init__.py",
    "model_100m/model.py",
    "model_10m/__init__.py",
    "model_10m/model.py",
    "tokenizer/tokenizer.json",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(folder, curves):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Use a fresh output directory for the reviewed public bundle")
    folder.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.model_info(SOURCE, files_metadata=True)
    if not info.private:
        raise ValueError("The full source checkpoint archive must remain private")
    source_hash = next(f.lfs.sha256 for f in info.siblings if f.rfilename == "model.safetensors")
    if source_hash != WEIGHTS_SHA256:
        raise ValueError("Latest completed checkpoint changed; review release metadata")
    try:
        api.model_info(TARGET)
    except RepositoryNotFoundError:
        pass
    else:
        raise ValueError("Public target already exists; do not overwrite an unreviewed repository")
    for name in FILES:
        downloaded = hf_hub_download(SOURCE, name, revision=info.sha)
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, destination)
    if sha(folder / "model.safetensors") != WEIGHTS_SHA256:
        raise ValueError("Downloaded weights do not match the completed model")
    record = {
        "model_name": "124M - 10.879B",
        "parameters": 124389632,
        "lifetime_training_tokens": 10878562619,
        "latest_phase_training_tokens": 4999971081,
        "latest_phase_optimizer_updates": 66065,
        "context": 8192,
        "weights_sha256": WEIGHTS_SHA256,
        "tokenizer_sha256": sha(folder / "tokenizer/tokenizer.json"),
        "source_code": "https://github.com/koogle/lm-exploration",
        "architecture": "14 layers, width 768, 12 heads, RMSNorm, RoPE, SwiGLU, tied embeddings",
        "private_training_documents_included": False,
        "historical_checkpoints_included": False,
    }
    (folder / "release.json").write_text(json.dumps(record, indent=2) + "\n")
    (folder / "README.md").write_text(
        """---
language: en
library_name: pytorch
model_name: 124M - 10.879B
tags:
- causal-lm
- from-scratch
- custom-code
- experimental
---
# Simple LLM — 124M - 10.879B

Public inference release of the latest completed **124,389,632-parameter** Simple
LLM checkpoint, trained on **10,878,562,619 next-token targets**. This is a base
language model, not an instruction-tuned assistant. Generations can be inaccurate,
incorrect as code, and severely repetitive, especially with greedy/low-temperature
decoding. The new multilingual continuation is not included in this release.

## Architecture and training

14 layers, width 768, 12 attention heads; RMSNorm, RoPE (theta 10,000), SwiGLU and tied
32,768-token embeddings. Configured context 8,192; this is not a claim of validated
8K task performance. Frozen byte-level BPE, PAD 0 / BOS 1 / EOS 2 / UNK 3.

Training history: 4,999,981,176 targets on the original web/code/math/research/books
mix, 878,610,362 on ECB/EMA material, then 4,999,971,081 in a 60%-Python-code
continuation, ending at step 66,065 of that phase. The final phase used eight H100s,
BF16 computation with FP32 weights, cropped padding, cached RoPE, torch.compile,
microbatch 2 / global batch 32 and fused AdamW. Adam moments and RNG were preserved;
final-phase LR was constant 3e-5. The tokenizer remained frozen.

The last phase collected 2,629,230,692 fresh unique targets and allowed at most two
passes per source within that phase. Lifetime exposure can exceed two passes.
Split, near-duplicate and contamination audits for this historical training remain
incomplete. Model weights and inference code are released; training documents,
optimizer/RNG state and earlier checkpoint archives are not included here.

## Evaluation and repetition

Matched expanded evaluation used 7,620,187 targets across 12 original sources plus
ECB/EMA. Final coding loss 1.421 versus 1.559 for the original 5B model; other original
sources mostly recovered from EU continuation but remain slightly worse than the
original baseline. ECB/EMA losses 2.494 / 2.379 show substantial loss of specialization.
These are 2K window loss measurements, not capability or factuality benchmarks.

In a small matched set of ten prompts, repeated generated token four-gram fraction was
81.6% for greedy, 77.0% at T=0.2/k=10 and 23.5% at T=0.8/k=50. Lower exact repetition does
not establish correct facts or code; semantic repetition may remain.

[Training/validation loss curves](evaluation/loss-curves.png) ·
[Gradient norm, learning rate and throughput](evaluation/optimizer-curves.png).
Historical validation curves use one fixed window per source, separate from the
expanded checkpoint evaluation; different phase mixtures are not directly comparable.

## Download and generate

Custom PyTorch model; it does not implement Transformers AutoModel.

```bash
pip install huggingface_hub torch==2.6.0 safetensors tokenizers==0.22.2
python -c "from huggingface_hub import snapshot_download; snapshot_download('jfrick/simple-llm-124M-10.879B', local_dir='simple-llm')"
python simple-llm/generate.py --prompt 'def add(a, b):' --tokens64 --temperature0.8 --top-k50
```

Use spaces in CLI flags: `--tokens 64 --temperature 0.8 --top-k 50`.
Generation suppresses PAD/BOS/UNK and stops on EOS; no repetition penalty is applied.
Exact token counts and artifact hashes are recorded in release.json.
Source: https://github.com/koogle/lm-exploration .
""".replace(
            "--tokens64 --temperature0.8 --top-k50", "--tokens 64 --temperature 0.8 --top-k 50"
        ).replace("Use spaces in CLI flags: `--tokens 64 --temperature 0.8 --top-k 50`.\n", "")
    )
    (folder / "evaluation").mkdir()
    for name in [
        "loss-curves.png",
        "optimizer-curves.png",
        "loss-curves.svg",
        "optimizer-curves.svg",
    ]:
        shutil.copyfile(Path(curves) / name, folder / "evaluation" / name)
    # Clean subprocess exercises only the exported bundle, never project imports.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    smoke = subprocess.run(
        [
            sys.executable,
            str(folder / "generate.py"),
            "--prompt",
            "def add(a, b):",
            "--tokens",
            "16",
            "--temperature",
            "0",
        ],
        cwd=folder,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    if not smoke.stdout.startswith("def add(a, b):"):
        raise ValueError("Standalone inference smoke check failed")
    print(
        json.dumps({"standalone_inference_verified": True, "weights_hash_verified": True}),
        flush=True,
    )
    # Upload privately first, then make only this allowlisted inference bundle public.
    api.create_repo(TARGET, private=True)
    revision = api.upload_folder(
        repo_id=TARGET,
        folder_path=str(folder),
        allow_patterns=[
            *FILES,
            "README.md",
            "release.json",
            "evaluation/*.png",
            "evaluation/*.svg",
        ],
        commit_message="Release completed Simple LLM124M at10.879B training tokens",
    )
    remote = api.model_info(TARGET, revision=revision.oid, files_metadata=True)
    actual = next(f.lfs.sha256 for f in remote.siblings if f.rfilename == "model.safetensors")
    if actual != WEIGHTS_SHA256:
        raise ValueError("Remote release weights mismatch; keep staged repo private")
    api.update_repo_settings(TARGET, private=False)
    public = HfApi(token=False).model_info(TARGET, files_metadata=True)
    if (
        public.private
        or next(f.lfs.sha256 for f in public.siblings if f.rfilename == "model.safetensors")
        != WEIGHTS_SHA256
    ):
        raise ValueError("Anonymous public release verification failed")
    # Source archive privacy must not change during public publication.
    if not api.model_info(SOURCE).private:
        raise ValueError("Source archive unexpectedly changed visibility")
    result = {
        "repo_id": TARGET,
        "public": True,
        "revision": public.sha,
        "weights_hash_verified": True,
        "standalone_inference_verified": True,
        "anonymous_access_verified": True,
        "private_archive_preserved": True,
    }
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--curves", required=True, type=Path)
    args = parser.parse_args()
    publish(args.output.resolve(), args.curves.resolve())
