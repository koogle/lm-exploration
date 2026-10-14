"""Export inference-only drafting checkpoints and publish them to Hugging Face."""

import hashlib
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

from .dflash import DraftConfig

QUALIFIED_SHA256 = "44865d84016898f575cf009c5cd2c2acb919564f009e912170cf4ba131b4e768"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(checkpoint, output, expected_sha256=None):
    """Never export optimizer, RNG, datasets, or other training checkpoint fields."""
    checksum = sha256(checkpoint)
    if expected_sha256 and checksum != expected_sha256:
        raise ValueError("checkpoint checksum does not match the selected artifact")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = DraftConfig(**state["config"]).validate()
    inference = {key: state[key] for key in ("drafter", "config", "mask_token_id", "step")}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(inference, output / "drafter.pt")
    (output / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "source_checkpoint_sha256": checksum,
                "inference_checkpoint_sha256": sha256(output / "drafter.pt"),
                "step": state["step"],
                "target_model": cfg.model,
                "target_revision": cfg.model_revision,
                "draft_parameters": sum(t.numel() for t in state["drafter"].values()),
            },
            indent=2,
        )
        + "\n"
    )
    project = Path(__file__).resolve().parents[2]
    for filename in ("dflash.py", "dflash_cli.py"):
        source = Path(__file__).with_name(filename).read_text()
        (output / filename).write_text(source.replace("from .dflash import", "from dflash import"))
    shutil.copyfile(project / "references/dflash-LICENSE", output / "dflash-LICENSE")
    if checksum == QUALIFIED_SHA256:
        shutil.copyfile(project / "docs/huggingface-model-card.md", output / "README.md")
    else:
        (output / "README.md").write_text(
            f"# Research drafting checkpoint\n\nTarget: {cfg.model} at {cfg.model_revision}. "
            f"Step: {state['step']}. Custom DFlash-style head; requires the frozen target.\n"
            "No qualification results are asserted for this artifact. "
            "Load drafter.pt with the bundled dflash_cli.generate_checkpoint function.\n"
        )
    (output / "requirements.txt").write_text(
        "torch==2.6.0\ntransformers==5.5.0\nhuggingface-hub>=1.5,<2\n"
    )
    return output


def publish_checkpoint(folder, repo_id):
    """Publish only the explicitly exported files; authentication stays in HF_TOKEN."""
    folder = Path(folder)
    names = (
        "README.md",
        "drafter.pt",
        "config.json",
        "provenance.json",
        "dflash.py",
        "dflash_cli.py",
        "dflash-LICENSE",
        "requirements.txt",
    )
    for name in names:
        if not (folder / name).is_file():
            raise ValueError(f"missing export file: {name}")
    api = HfApi()
    api.whoami()  # Fail before creating anything when write credentials are absent.
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    if api.model_info(repo_id).private:
        raise ValueError("destination is private; choose a new public model repository")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=folder,
        allow_patterns=list(names),
        commit_message="Publish qualified frozen-Qwen drafter and inference code",
    )
    return {"url": f"https://huggingface.co/{repo_id}", "revision": commit.oid}


def download_checkpoint(repo_id, revision):
    if not revision:
        raise ValueError("provide a Hugging Face revision to pin the inference artifact")
    return hf_hub_download(repo_id, "drafter.pt", revision=revision)
