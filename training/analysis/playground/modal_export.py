"""Export checkpoints to a private HF model repository and serve them in a private Space.

Run from training/ (needs an HF token with write access, e.g. `hf auth login`):
    modal run -m playground.modal_export --config analysis/playground/configs/base-vs-math-sft.json \
        --name NAME

The config names the model repository, the Space, the checkpoints (label, description and
volume path), their display layout, the models selected by default, the decoder file they share and the evaluation results to quote.
One CPU call (30-minute timeout, no retries; refuses work 24 hours after launch) loads each
checkpoint with its run's own model code (`trainer.checkpoints`), checks that the shared
`model.py` reproduces its logits and writes float32 safetensors, the tokenizer and
`models.json` to `/runs/playground/NAME`. The local entrypoint downloads that folder to
`artifacts/playground/NAME`, uploads it to the model repository, then uploads this folder's
app files with `models.json` (pinned to that upload's commit) to the Space and gives the
Space your token as its `HF_TOKEN` secret, so it can download the private weights.
"""

import json
import shutil
import time
from pathlib import Path

import modal

LAUNCH_LIMIT_SECONDS = 24 * 60 * 60
APP_FILES = ["app.py", "intro.md", "requirements.txt"]
# The shared cached sampler, served next to the app.
SAMPLER = Path(__file__).resolve().parents[2] / "trainer" / "rollout.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0", "numpy", "pydantic>=2.11,<3", "safetensors==0.5.3", "tokenizers==0.22.2"
    )
    .env({"PYTHONPATH": "/project"})
    .add_local_dir("data", "/project/training_data", ignore=["raw", "cache", "__pycache__"])
    .add_local_dir("analysis/evaluation", "/project/evaluation", ignore=["__pycache__"])
    .add_local_file("trainer/__init__.py", "/project/trainer/__init__.py")
    .add_local_file("trainer/checkpoints.py", "/project/trainer/checkpoints.py")
)
app = modal.App("playground-export")
volume = modal.Volume.from_name("lm-exploration-training")


def quoted_scores(names):
    """Headline scores per checkpoint from evaluation results; later files win."""
    found = {}
    for name in names:
        result = json.loads((Path("/runs/evaluation") / f"{name}.json").read_text())
        for model in result["models"]:
            entry = found.setdefault(model["checkpoint"], {})
            values = model["scores"]
            for task in ("gsm8k", "svamp"):
                if task in values:
                    entry[task] = values[task]["acc"]
                    entry[f"{task}_shots"] = result["settings"]["shots"]
            for task in ("mbpp_plus", "humaneval_plus"):
                if task in values:
                    entry[task] = values[task]["pass@1"]
            if "hellaswag" in values:
                entry["hellaswag_acc_norm"] = values["hellaswag"]["acc_norm"]
                entry["arc_easy_acc"] = values["arc_easy"]["acc"]
    return found


def weights_file(label):
    return f"weights/{label.lower().replace(' ', '-')}.safetensors"


@app.function(image=image, cpu=8, memory=16384, timeout=1800, retries=0, volumes={"/runs": volume})
def export(name, config, deadline):
    import importlib.util

    import torch
    from safetensors.torch import save_file
    from training_data.subword import digest

    from trainer.checkpoints import load

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing export")
    output = Path("/runs/playground") / name
    if output.exists():
        raise ValueError("Use a fresh export name")
    (output / "weights").mkdir(parents=True)
    shutil.copyfile(Path("/runs") / config["model_code"], output / "model.py")
    spec = importlib.util.spec_from_file_location("served_model", output / "model.py")
    served_code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(served_code)
    scores = quoted_scores(config["evaluations"])
    probe = torch.randint(4, 32768, (1, 64), generator=torch.Generator().manual_seed(0))
    entries, shared = [], None
    for item in config["models"]:
        loaded = load(Path("/runs") / item["checkpoint"])
        keys = ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")
        dimensions = {key: loaded.config[key] for key in keys}
        served = served_code.GPT(served_code.ModelConfig(**dimensions)).eval()
        served.load_state_dict(loaded.state, strict=True)
        with torch.no_grad():
            if not torch.equal(loaded.model(probe), served(probe)):
                raise ValueError(f"model.py does not reproduce {item['checkpoint']}")
        common = {
            "config": dimensions,
            "tokenizer_sha256": loaded.info["tokenizer_sha256"],
            "special_tokens": {"bos": loaded.special["bos"], "eos": loaded.special["eos"]},
        }
        if shared is None:
            shared = common
            shutil.copyfile(loaded.tokenizer_path, output / "tokenizer.json")
        elif common != shared:
            raise ValueError("Every model must share dimensions, tokenizer and special tokens")
        state = loaded.state
        if not torch.equal(state["lm_head.weight"], state["token_embedding.weight"]):
            raise ValueError("Expected tied output and embedding weights")
        # The output head shares the embedding matrix; store it once.
        weights = {k: v.contiguous() for k, v in state.items() if k != "lm_head.weight"}
        file = weights_file(item["label"])
        save_file(weights, output / file)
        entries.append(
            {
                **item,
                "weights": file,
                "weights_sha256": digest(output / file),
                **loaded.info,
                "scores": scores.get(item["checkpoint"], {}),
            }
        )
        print(json.dumps(entries[-1]), flush=True)
    manifest = {
        **shared,
        "model_code": {"source": config["model_code"], "sha256": digest(output / "model.py")},
        "layout": config["layout"],
        "default_models": config["default_models"],
        "models": entries,
    }
    (output / "models.json").write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()
    return manifest


@app.local_entrypoint()
def main(config: str, name: str):
    from huggingface_hub import HfApi, get_token

    settings = json.loads(Path(config).read_text())
    labels = [item["label"] for item in settings["models"]]
    if sorted(labels) != sorted(label for row in settings["layout"] for label in row):
        raise ValueError("layout must name every model once")
    if not set(settings["default_models"]) <= set(labels):
        raise ValueError("default_models must name configured models")
    local = Path("artifacts/playground") / name
    if local.exists():
        raise ValueError(f"{local} already exists")
    manifest = export.remote(name, settings, time.time() + LAUNCH_LIMIT_SECONDS)
    files = ["model.py", "tokenizer.json", "models.json"]
    files += [entry["weights"] for entry in manifest["models"]]
    for file in files:
        target = local / "model" / file
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            for chunk in volume.read_file(f"playground/{name}/{file}"):
                handle.write(chunk)
    api = HfApi()
    repo = settings["model_repo"]
    api.create_repo(repo, private=True, exist_ok=True)
    weights = api.upload_folder(
        repo_id=repo, folder_path=local / "model", commit_message=f"Export {name}"
    )
    space = local / "space"
    space.mkdir()
    here = Path(__file__).parent
    for file in APP_FILES:
        shutil.copyfile(here / file, space / file)
    shutil.copyfile(SAMPLER, space / "rollout.py")
    shutil.copyfile(here / "space-card.md", space / "README.md")
    for file in ("model.py", "tokenizer.json"):
        shutil.copyfile(local / "model" / file, space / file)
    served = {**manifest, "weights_repo": repo, "weights_revision": weights.oid}
    (space / "models.json").write_text(json.dumps(served, indent=2) + "\n")
    api.create_repo(
        settings["space"], repo_type="space", space_sdk="gradio", private=True, exist_ok=True
    )
    api.add_space_secret(settings["space"], "HF_TOKEN", get_token())
    commit = api.upload_folder(
        repo_id=settings["space"],
        repo_type="space",
        folder_path=space,
        commit_message=f"Serve {name} (weights {weights.oid[:12]})",
    )
    print(json.dumps({"model_repo": repo, "weights": weights.oid, "space": commit.oid}))
