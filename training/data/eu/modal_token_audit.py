"""Bounded token measurement, isolated from document collection workers."""

import json
from pathlib import Path

import modal

app = modal.App("lm-exploration-eu-token-audit")
image = None
if modal.is_local():
    from modal_pipeline import image as base_image

    image = base_image
    for name in ("token_audit.py", "goal_pipeline_sources.json"):
        image = image.add_local_file(Path(__file__).with_name(name), f"/root/eu/{name}")


@app.function(image=image, cpu=1, memory=512, timeout=120, retries=0)
def verify_tokenizer():
    import tiktoken

    encoder = tiktoken.get_encoding("o200k_base")
    return {
        "tokenizer": encoder.name,
        "sample_tokens": len(encoder.encode("EU corpus verification")),
    }


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("eu-corpus-huggingface", required_keys=["HF_TOKEN"])],
    cpu=1,
    memory=2048,
    timeout=1800,
    max_containers=1,
    retries=0,
)
def audit(source_id: str, max_shards: int = 1000, work_seconds: int = 1500):
    from token_audit import audit_source

    return audit_source(source_id, max_shards=max_shards, work_seconds=work_seconds)


@app.local_entrypoint()
def main(source: str = "ecb", max_shards: int = 1000, work_seconds: int = 1500):
    print(json.dumps(audit.remote(source, max_shards, work_seconds), indent=2))
