"""Publish a distillation student to a private Gradio Space on Hugging Face.

Run from the repository root (needs ``HF_TOKEN`` with write access):

    python distillation/playground/publish.py --space jfrick/distill-student-150m-playground \
        --campaign distill-validation-20261023-01 --student trim-large-500m-kd

Downloads the student checkpoint and its tokenizer/vocabulary into the ignored
``distillation/playground/artifacts/`` directory, then uploads them with the app code.
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CHECKPOINTS = "jfrick/distillation-validation-checkpoints-20261023"
SHARDS = "jfrick/qwen35-distillation-shards-20261023"
CODE = {
    "app.py": HERE / "app.py",
    "requirements.txt": HERE / "requirements.txt",
    "README.md": HERE / "space-card.md",
    "model.py": ROOT / "training" / "10m" / "model.py",
    "evalsets.py": ROOT / "distillation" / "validation" / "evalsets.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--stage-only", action="store_true", help="assemble locally, no upload"
    )
    args = parser.parse_args()
    local = HERE / "artifacts" / args.student
    local.mkdir(parents=True, exist_ok=True)
    checkpoint = hf_hub_download(
        CHECKPOINTS,
        f"{args.campaign}/students/{args.student}/model.pt",
        repo_type="dataset",
        local_dir=local,
    )
    summary = json.loads(
        Path(
            hf_hub_download(
                CHECKPOINTS,
                f"{args.campaign}/students/{args.student}/summary.json",
                repo_type="dataset",
                local_dir=local,
            )
        ).read_text()
    )
    data = {
        name: hf_hub_download(SHARDS, name, repo_type="dataset", local_dir=local)
        for name in ("tokenizer.json", "vocab-table.npy", "vocab-kept.npy")
    }
    settings = {
        "name": args.student,
        "description": args.description
        or (
            f"From-scratch student from campaign `{args.campaign}`: "
            f"{summary['backbone_parameters'] / 1e6:.1f}M backbone parameters, "
            f"{summary['labels'] / 1e6:.0f}M training tokens."
        ),
        "campaign": args.campaign,
        "checkpoint": f"{CHECKPOINTS}/{args.campaign}/students/{args.student}/model.pt",
    }
    stage = local / "space"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    for name, source in {**CODE, **data, "model.pt": checkpoint}.items():
        shutil.copy(source, stage / name)
    (stage / "model-settings.json").write_text(json.dumps(settings, indent=1))
    if args.stage_only:
        print(stage)
        return
    api = HfApi()
    api.create_repo(
        args.space, repo_type="space", space_sdk="gradio", private=True, exist_ok=True
    )
    commit = api.upload_folder(
        repo_id=args.space,
        repo_type="space",
        folder_path=str(stage),
        commit_message=f"Publish {args.student} from {args.campaign}",
    )
    print(
        json.dumps({"space": args.space, "revision": commit.oid, **settings}, indent=1)
    )


if __name__ == "__main__":
    main()
