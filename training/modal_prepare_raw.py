"""Durable CPU preparation for the authorized 5B-token small-model experiment."""

import json
from pathlib import Path

import modal

from modal_scripts.modal_prepare_simple_llm import image

image = image.add_local_dir("modal_scripts", "/project/modal_scripts")

app = modal.App("lm-exploration-small-5b-preparation")
volume = modal.Volume.from_name("lm-exploration-mixture-data")


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=86400,
    retries=0,
    max_containers=1,
    volumes={"/data": volume},
)
def prepare(version, target_tokens=5_000_000_000):
    from training_data.raw_corpus import RawCorpus
    from training_data.simple_llm import load_recipe

    recipe = load_recipe("/project/tokenizer/recipe-v1.json")
    folder = Path("/data/scholarly-code") / version
    corpus = RawCorpus(folder, recipe)
    try:
        # Check potentially constrained book collections first, then the larger sources.
        names = sorted(recipe["sources"], key=lambda s: len(s["files"]))
        for source in names:
            name = source["name"]
            while True:
                report = corpus.report(target_tokens)
                status = report["sources"][name]
                # Five percent reserve for later removals and whole-batch rounding.
                if (
                    status["train_loss_tokens"] >= status["minimum_unique_train_tokens"] * 1.05
                    and status["validation_loss_tokens"] > 0
                ):
                    break
                if status["upstream_exhausted"]:
                    break
                scanned = corpus.prepare(name)
                report = corpus.report(target_tokens)
                (folder / "capacity.json").write_text(json.dumps(report, indent=2) + "\n")
                volume.commit()
                print(
                    json.dumps({"source": name, "scanned": scanned, **report["sources"][name]}),
                    flush=True,
                )
        report = corpus.report(target_tokens)
        (folder / "capacity.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        corpus.close()
        volume.commit()


@app.local_entrypoint()
def main(version: str = "scholarly-small-5b-raw-20261017"):
    import re

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", version):
        raise ValueError("Invalid version")
    report = prepare.remote(version)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{version}-capacity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
