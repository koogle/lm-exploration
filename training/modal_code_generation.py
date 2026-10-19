"""Private matched generation of original, EU, and final code-heavy checkpoints."""

from pathlib import Path

import modal

from modal_eu_continuation import image, run_volume

image = image.add_local_file("modal_eu_continuation.py", "/project/modal_eu_continuation.py")
app = modal.App("simple-llm-code-heavy-generation")
CHECKPOINTS = {
    "original_5b": "simple-llm-124m-8k-5b-20261018-s03/step-000064227.pt",
    "eu_parent": "simple-llm-124m-eu-onepass-20261018-s02/step-000003838.pt",
    "final_code60": "simple-llm-code60-fresh-5b-20261019-r02-s02/step-000066065.pt",
}
RUN = "simple-llm-code-heavy-generation-20261019"


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=3600,
    retries=0,
    max_containers=2,
    volumes={"/runs": run_volume},
)
def generate(label):
    from trainer.continuation_generation import generate_checkpoint

    if label not in CHECKPOINTS:
        raise ValueError("Unknown checkpoint label")
    return generate_checkpoint(
        Path("/runs") / CHECKPOINTS[label],
        Path("/runs") / RUN / (label + ".json"),
        label,
        run_volume.commit,
    )
