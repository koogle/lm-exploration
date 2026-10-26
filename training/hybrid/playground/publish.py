"""Publish a hybrid-family export to a private Gradio Space on Hugging Face.

Run from training/ (needs HF_TOKEN with write access):

    uv run --all-extras python -m model_hybrid.playground.publish \\
        --space jfrick/hybrid-720m-playground --export tokens-039998980096

Uploads this app with the model code (the CPU inference modules of hybrid/ as
model_hybrid/, 10m/model.py as model_10m/). Spaces hold at most 1 GB, so the app
downloads the export from the private model repo at startup; add a read token as
the Space secret HF_TOKEN in the Space settings. `--stage-only` assembles locally.
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).parent
TRAINING = HERE.parents[1]
# What model.py and generate.py import on CPU.
MODULES = (
    "__init__",
    "model",
    "kda",
    "documents",
    "generate",
    "tokenizer",
    "fused_loss",
    "softmax_kernel",
    "cuda_extension",
    "kda_convolution",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True)
    parser.add_argument("--repo", default="jfrick/hybrid-720m-40b")
    parser.add_argument("--export", required=True, help="export folder in the model repo")
    parser.add_argument("--name", default="Hybrid 720M, 40B tokens (base model)")
    parser.add_argument("--description", default="")
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
    }
    stage = local / "space"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "model_hybrid").mkdir(parents=True)
    for module in MODULES:
        shutil.copy(TRAINING / "hybrid" / f"{module}.py", stage / "model_hybrid" / f"{module}.py")
    (stage / "model_10m").mkdir()
    shutil.copy(TRAINING / "10m" / "model.py", stage / "model_10m" / "model.py")
    (stage / "model_10m" / "__init__.py").write_text("")
    shutil.copy(HERE / "app.py", stage / "app.py")
    shutil.copy(HERE / "requirements.txt", stage / "requirements.txt")
    shutil.copy(HERE / "space-card.md", stage / "README.md")
    (stage / "model-settings.json").write_text(json.dumps(settings, indent=1))
    if args.stage_only:
        print(stage)
        return
    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.space,
        repo_type="space",
        folder_path=str(stage),
        commit_message=f"Publish {args.repo}/{args.export}",
    )
    print(json.dumps({"space": args.space, "revision": commit.oid, **settings}, indent=1))


if __name__ == "__main__":
    main()
