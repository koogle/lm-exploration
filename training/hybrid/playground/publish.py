"""Publish a hybrid-family export to a private Gradio Space on Hugging Face.

Run from training/ (needs HF_TOKEN with write access):

    uv run --all-extras python -m model_hybrid.playground.publish \\
        --space jfrick/hybrid-720m-playground --export tokens-039998980096

Uploads this app as a Docker Space (Dockerfile: CUDA 13.0 devel, the family's
kernels prebuilt) with the model code (the inference modules and kernels of hybrid/
as model_hybrid/, 10m/model.py as model_10m/, trainer/roles.py, tokenizer/glm.py and
data/subword.py, chat.py, chat_sft.py), on GPU hardware (`--hardware`, L4 by
default; it sleeps after `--sleep-seconds` without visitors). GPU hardware is billed
to the Space owner while it runs. Spaces hold at most 1 GB, so the app downloads the
export from the private model repo at startup; add a read token as the Space secret
HF_TOKEN in the Space settings. `--stage-only` assembles locally.
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1]
# What model.py and generate.py import on CUDA, and the kernels they load.
MODULES = (
    "__init__",
    "model",
    "kda",
    "documents",
    "generate",
    "fused_loss",
    "softmax_kernel",
    "cuda_extension",
    "kda_convolution",
)
# Beside them: the GLM tokenizer pin and chat format, and what they import.
SHARED = {
    "tokenizer": ("tokenizer/glm.py",),
    "training_data": ("data/subword.py", "data/chat.py", "data/chat_sft.py"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True)
    parser.add_argument("--repo", default="jfrick/hybrid-720m-40b")
    parser.add_argument("--export", required=True, help="export folder in the model repo")
    parser.add_argument("--name", default="Hybrid 720M, 40B tokens (base model)")
    parser.add_argument("--description", default="")
    parser.add_argument("--chat", action="store_true", help="a chat SFT export: add the chat tab")
    parser.add_argument("--hardware", default="l4x1", help="Space GPU hardware")
    parser.add_argument("--sleep-seconds", type=int, default=3600)
    parser.add_argument("--stage-only", action="store_true", help="assemble locally, no upload")
    args = parser.parse_args()
    local = TRAINING / "artifacts" / "playground" / args.export
    progress_file = hf_hub_download(args.repo, f"{args.export}/progress.json", local_dir=local)
    progress = json.loads(Path(progress_file).read_text())
    settings = {
        "name": args.name,
        "description": args.description
        or (
            f"`{args.repo}/{args.export}`: trained on {progress['tokens'] / 1e9:.1f}B tokens of "
            "ClimbMix, permissive code and FineMath. A base model: it continues text."
        ),
        "repo": args.repo,
        "export": args.export,
        "chat": args.chat,
    }
    stage = local / "space"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "model_hybrid").mkdir(parents=True)
    for module in MODULES:
        shutil.copy(TRAINING / "hybrid" / f"{module}.py", stage / "model_hybrid" / f"{module}.py")
    shutil.copytree(TRAINING / "hybrid" / "kernels", stage / "model_hybrid" / "kernels")
    (stage / "model_10m").mkdir()
    shutil.copy(TRAINING / "10m" / "model.py", stage / "model_10m" / "model.py")
    (stage / "model_10m" / "__init__.py").write_text("")
    # The modules declare their optimizer roles through trainer.roles.
    (stage / "trainer").mkdir()
    shutil.copy(TRAINING / "trainer" / "roles.py", stage / "trainer" / "roles.py")
    (stage / "trainer" / "__init__.py").write_text("")
    for package, files in SHARED.items():
        (stage / package).mkdir()
        (stage / package / "__init__.py").write_text("")
        for file in files:
            shutil.copy(TRAINING / file, stage / package / Path(file).name)
    shutil.copy(HERE / "app.py", stage / "app.py")
    shutil.copy(HERE / "requirements.txt", stage / "requirements.txt")
    shutil.copy(HERE / "Dockerfile", stage / "Dockerfile")
    shutil.copy(HERE / "space-card.md", stage / "README.md")
    (stage / "model-settings.json").write_text(json.dumps(settings, indent=1))
    if args.stage_only:
        print(stage)
        return
    api = HfApi()
    api.create_repo(
        args.space,
        repo_type="space",
        space_sdk="docker",
        space_hardware=args.hardware,
        space_sleep_time=args.sleep_seconds,
        private=True,
        exist_ok=True,
    )
    api.request_space_hardware(args.space, args.hardware, sleep_time=args.sleep_seconds)
    commit = api.upload_folder(
        repo_id=args.space,
        repo_type="space",
        folder_path=str(stage),
        commit_message=f"Publish {args.repo}/{args.export}",
    )
    print(json.dumps({"space": args.space, "revision": commit.oid, **settings}, indent=1))


if __name__ == "__main__":
    main()
