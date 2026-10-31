"""Run one sampling recipe on one Modal H100. No training or model-repository writes.

Run from training/:
    modal run --detach -m sampling.modal_sample --recipe <recipe.json> [--output <result.json>]

One function call per launch, a 30-minute hard timeout and no retries. The call refuses new
work once 24 hours have passed since launch, including time spent queued. Completed runs are
saved to the output volume after each model and reused by an identical rerun.
"""

import json
import time
from pathlib import Path

import modal

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
TIMEOUT_SECONDS = 30 * 60


def image(*packages):
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.14.1", "numpy", *packages)
        .env({"PYTHONPATH": "/project"})
        .add_local_dir("analysis/sampling", "/project/sampling")
    )


# Own checkpoints load their model code from the pinned revision, never from this checkout.
checkpoint_image = image(
    "pydantic>=2.11,<3", "tokenizers==0.22.2", "huggingface-hub==1.33.0", "safetensors>=0.4"
)
transformers_image = image("transformers==4.51.3", "huggingface-hub==0.30.2", "safetensors")
# Unpublished training checkpoints load with this checkout's model code, verified against
# the source hashes the checkpoint recorded.
volume_image = (
    image("pydantic>=2.11,<3", "tokenizers==0.22.2")
    .add_local_dir("100m", "/project/model_100m", ignore=["__pycache__", "images"])
    .add_local_dir("10m", "/project/model_10m", ignore=["__pycache__"])
    .add_local_dir("tokenizer", "/project/tokenizer", ignore=["__pycache__", "research"])
)


def hybrid_image():
    """The hybrid family's CUDA image (fla, FlexAttention) with this checkout's model code.

    Containers of the other images do not ship model_hybrid; they never run sample_hybrid.
    """
    try:
        from model_hybrid.image import image as base
    except ImportError:
        return None
    return base.add_local_dir("analysis/sampling", "/project/sampling")


app = modal.App("sampling-recipe")
volume = modal.Volume.from_name("lm-exploration-training")
limits = {
    "gpu": "H100",
    "cpu": 4,
    "timeout": TIMEOUT_SECONDS,
    "retries": 0,
    "max_containers": 1,
    "volumes": {"/runs": volume},
}


def sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(*args, **kwargs):
    """snapshot_download with four bounded attempts for transient failures."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    transient = (OSError, HfHubHTTPError)
    try:
        import httpx

        transient += (httpx.TransportError,)
    except ImportError:
        pass
    for attempt in range(4):
        try:
            return Path(snapshot_download(*args, **kwargs))
        except transient as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(exc, HfHubHTTPError) and status not in (429, 500, 502, 503, 504):
                raise
            if attempt == 3:
                raise
            print(f"Retrying download after {type(exc).__name__}", flush=True)
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable download state")


def open_checkpoints(spec, done):
    """Yield hash-verified checkpoints of our own published model."""
    import sys
    from types import SimpleNamespace

    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    source = download(
        spec["repo"],
        revision=spec["source_revision"],
        allow_patterns=[
            "config.json",
            "provenance.json",
            "tokenizer/tokenizer.json",
            "model_100m/*.py",
            "model_10m/*.py",
        ],
    )
    source_hash = sha256(source / "model_100m/model.py")
    tokenizer_hash = sha256(source / "tokenizer/tokenizer.json")
    identity = json.loads((source / "provenance.json").read_text())
    assert source_hash == identity["source_sha256"]["model"] == spec["model_source_sha256"]
    assert tokenizer_hash == spec["tokenizer_sha256"]
    # Our own previously published, hash-verified code, not the current repository architecture.
    sys.path.insert(0, str(source))
    from model_100m.model import GPT, ModelConfig

    model = GPT(ModelConfig(**json.loads((source / "config.json").read_text()))).cuda().eval()
    tokenizer = Tokenizer.from_file(str(source / "tokenizer/tokenizer.json"))
    wrapped = SimpleNamespace(
        forward=model,
        encode=lambda p: [
            *spec["prefix_token_ids"],
            *tokenizer.encode(p, add_special_tokens=False).ids,
        ],
        decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        eos_token_id=spec["eos_token_id"],
        device="cuda",
    )
    for checkpoint in spec["checkpoints"]:
        name = f"{spec['name']}-step{checkpoint['step']}"
        if name in done:
            continue
        folder = download(
            spec["repo"],
            revision=checkpoint["revision"],
            allow_patterns=["model.safetensors", "provenance.json"],
        )
        provenance = json.loads((folder / "provenance.json").read_text())
        assert provenance["step"] == checkpoint["step"]
        assert provenance["source_sha256"]["model"] == source_hash
        assert sha256(folder / "model.safetensors") == provenance["weights_sha256"]
        model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert sum(p.numel() for p in model.parameters()) == spec["parameters"]
        info = {
            "name": name,
            "kind": "checkpoint",
            "repo": spec["repo"],
            "revision": checkpoint["revision"],
            "step": checkpoint["step"],
            "trained_tokens": provenance["tokens"],
            "parameters": spec["parameters"],
            "weights_sha256": provenance["weights_sha256"],
            "model_source_sha256": source_hash,
            "tokenizer_sha256": tokenizer_hash,
        }
        print(json.dumps(info), flush=True)
        yield info, wrapped


def open_volume(spec, done):
    """Yield one training checkpoint from the run volume, verified like publication export."""
    import importlib
    from types import SimpleNamespace

    import torch
    from tokenizers import Tokenizer

    name = spec["name"]
    if name in done:
        return
    path = Path("/runs") / spec["checkpoint"]
    # Only trusted checkpoints written by our own trainer on our own volume.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    architecture = saved["architecture"]
    module = importlib.import_module(architecture["module"])
    definitions = {"model": module.__file__, **getattr(module, "MODEL_DEPENDENCIES", {})}
    for key, source in definitions.items():
        assert sha256(source) == saved["source_sha256"][key], f"model source differs: {key}"
    tokenizer_path = Path("/project/tokenizer") / spec["tokenizer"] / "tokenizer.json"
    tokenizer_hash = sha256(tokenizer_path)
    assert tokenizer_hash == saved["data_identity"]["tokenizer_sha256"] == spec["tokenizer_sha256"]
    model = module.GPT(module.ModelConfig(**architecture["config"]))
    model.load_state_dict(saved["model"], strict=True)
    model = model.cuda().eval()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == spec["parameters"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = saved["special_tokens"]
    info = {
        "name": name,
        "kind": "volume",
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": sha256(path),
        "step": saved.get("step"),
        "trained_tokens": saved.get("tokens"),
        "parameters": spec["parameters"],
        "tokenizer_sha256": tokenizer_hash,
    }
    print(json.dumps(info), flush=True)
    yield (
        info,
        SimpleNamespace(
            forward=model,
            encode=lambda p: [ids["bos"], *tokenizer.encode(p, add_special_tokens=False).ids],
            decode=lambda generated: tokenizer.decode(generated, skip_special_tokens=True),
            eos_token_id=ids["eos"],
            device="cuda",
        ),
    )


def open_transformers(spec, done):
    """Yield one Hugging Face Transformers model at a pinned commit, without remote code."""
    from types import SimpleNamespace

    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if spec["name"] in done:
        return
    repo, revision = spec["repo"], spec["revision"]
    assert HfApi().model_info(repo, revision=revision).sha == revision
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        revision=revision,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    model = model.cuda().eval()
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == spec.get("parameters", parameters)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    prefix = spec.get("prefix_token_ids", [])
    info = {
        "name": spec["name"],
        "kind": "transformers",
        "repo": repo,
        "revision": revision,
        "trained_tokens": spec.get("trained_tokens"),
        "parameters": parameters,
        "transformers_version": transformers.__version__,
    }
    print(json.dumps(info), flush=True)
    yield (
        info,
        SimpleNamespace(
            forward=lambda tokens: model(tokens).logits,
            encode=lambda p: [*prefix, *tokenizer.encode(p, add_special_tokens=True)],
            decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            eos_token_id=tokenizer.eos_token_id,
            device="cuda",
        ),
    )


def open_hybrid(spec, done):
    """Yield one hybrid-family export (GLM-token hybrid or byte-level H-Net) at a pinned commit.

    The repository folder holds trainer exports: bf16 trunk weights, config.json and
    progress.json. Model code is this checkout's model_hybrid package; the recipe names
    the architecture, and config fields outside its dataclass (tokenizer, provenance) are
    not model settings.
    """
    import dataclasses
    from types import SimpleNamespace

    import torch
    from huggingface_hub import HfApi
    from model_hybrid import hnet, model
    from safetensors.torch import load_file

    from tokenizer import glm as tokenizer

    if spec["name"] in done:
        return
    repo, revision = spec["repo"], spec["revision"]
    assert HfApi().model_info(repo, revision=revision).sha == revision
    folder = download(repo, revision=revision, allow_patterns=[f"{spec['folder']}/*"])
    folder = folder / spec["folder"]
    saved = json.loads((folder / "config.json").read_text())
    progress = json.loads((folder / "progress.json").read_text())
    if spec["architecture"] == "hnet":
        config_class, build = hnet.HNetConfig, hnet.HNetLM
        text = hnet.ByteTokenizer()
        encode = lambda p: text.encode(p).ids
        decode, eos = text.decode, hnet.BYTE_EOS
    else:
        config_class, build = model.HybridConfig, model.HybridLM
        glm = tokenizer.load()
        encode = lambda p: glm.encode(p, add_special_tokens=False).ids
        decode = lambda ids: glm.decode(ids, skip_special_tokens=True)
        eos = tokenizer.EOS_ID
        # The family's token models use the GLM EOS; exports hold the inference trunk only.
        saved |= {"eos_id": eos, "mtp": False}
    fields = {f.name for f in dataclasses.fields(config_class)}
    settings = {k: tuple(v) if isinstance(v, list) else v for k, v in saved.items() if k in fields}
    lm = build(config_class(**settings))
    weights = load_file(str(folder / "model.safetensors"))
    lm.load_state_dict({k: v.float() for k, v in weights.items()}, strict=True)
    lm = lm.cuda().eval()
    parameters = sum(p.numel() for p in lm.parameters())
    assert parameters == spec["parameters"]
    assert all(torch.isfinite(p).all() for p in lm.parameters())
    info = {
        "name": spec["name"],
        "kind": "hybrid",
        "architecture": spec["architecture"],
        "repo": repo,
        "revision": revision,
        "folder": spec["folder"],
        "step": progress["step"],
        "trained_tokens": progress["tokens"],
        "parameters": parameters,
        "weights_sha256": sha256(folder / "model.safetensors"),
    }
    print(json.dumps(info), flush=True)

    def forward(tokens):
        """Left-pad to a multiple of 128 with EOS so kernels see few shapes.

        Each pad EOS ends a document, so document masking keeps the padding out of
        the prompt's attention, KDA state, routing and positions.
        """
        pad = -tokens.size(1) % 128
        padding = torch.full((tokens.size(0), pad), eos, dtype=tokens.dtype, device=tokens.device)
        return lm(torch.cat((padding, tokens), 1))[:, pad:]

    yield (
        info,
        SimpleNamespace(
            forward=forward,
            encode=encode,
            decode=decode,
            eos_token_id=eos,
            device="cuda",
            max_new_tokens=spec.get("max_new_tokens"),
        ),
    )


def run(recipe, deadline, open_runs):
    import torch

    from sampling.sampler import check_deadline, run_recipe

    check_deadline(deadline)
    torch.set_num_threads(4)
    path = Path("/runs/sampling") / f"{recipe['name']}.json"
    existing = json.loads(path.read_text()) if path.exists() else None

    def save(report):
        report["device"] = torch.cuda.get_device_name()
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(report, indent=2) + "\n")
        partial.replace(path)
        volume.commit()

    report = run_recipe(recipe, open_runs, deadline, existing, save)
    report["device"] = torch.cuda.get_device_name()
    return report


@app.function(image=checkpoint_image, memory=8192, **limits)
def sample_checkpoints(recipe, deadline):
    return run(recipe, deadline, open_checkpoints)


@app.function(image=transformers_image, memory=12288, **limits)
def sample_transformers(recipe, deadline):
    return run(recipe, deadline, open_transformers)


@app.function(image=volume_image, memory=16384, **limits)
def sample_volume(recipe, deadline):
    return run(recipe, deadline, open_volume)


# Uncached decoding through the byte-level H-Net runs one full forward pass per byte.
HYBRID_TIMEOUT_SECONDS = 2 * 60 * 60


@app.function(
    image=hybrid_image(),
    memory=32768,
    secrets=[modal.Secret.from_name("simple-llm-hf")],
    **{**limits, "timeout": HYBRID_TIMEOUT_SECONDS},
)
def sample_hybrid(recipe, deadline):
    return run(recipe, deadline, open_hybrid)


@app.local_entrypoint()
def main(recipe: str, output: str = ""):
    from sampling.recipe import load_recipe

    spec = load_recipe(recipe)
    deadline = time.time() + LAUNCH_LIMIT_SECONDS
    kind = spec["models"][0]["kind"]
    function = {
        "checkpoint": sample_checkpoints,
        "transformers": sample_transformers,
        "volume": sample_volume,
        "hybrid": sample_hybrid,
    }[kind]
    report = function.remote(spec, deadline)
    path = Path(output or f"artifacts/sampling/{spec['name']}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(path), "models": len(report["models"])}))
