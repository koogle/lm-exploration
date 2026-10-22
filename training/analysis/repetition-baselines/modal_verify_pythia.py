"""Independent greedy API / precision / prefix check on Pythia160M final.

Run from training/: modal run analysis/repetition-baselines/modal_verify_pythia.py
One H100, 10-minute hard timeout, no retries; refuses to start 24 hours after launch.
"""

import json
import time
from pathlib import Path

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.6.0", "transformers==4.51.3", "huggingface-hub==0.30.2", "safetensors", "numpy"
)

app = modal.App("simple-llm-pythia-verification")
LAUNCH_LIMIT_SECONDS = 24 * 60 * 60


@app.function(image=image, gpu="H100", cpu=4, memory=8192, timeout=600, retries=0)
def verify(deadline):
    from contextlib import nullcontext

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if time.time() >= deadline:
        raise TimeoutError("Launch deadline passed; refusing the check")

    repo = "EleutherAI/pythia-160m"
    revision = "50f5173d932e8e61f858120bcb800b97af589f46"
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, torch_dtype=torch.float32, attn_implementation="eager"
        )
        .cuda()
        .eval()
    )
    samples = []
    with torch.inference_mode():
        for prompt in ["The solar system consists of", "The capital of France is"]:
            for prefix_bos in [False, True]:
                ids = tok.encode(prompt, add_special_tokens=True)
                if prefix_bos:
                    ids = [tok.eos_token_id, *ids]
                for precision in ["fp32", "bf16"]:
                    ctx = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if precision == "bf16"
                        else nullcontext()
                    )
                    with ctx:
                        output = model.generate(
                            torch.tensor([ids], device="cuda"),
                            max_new_tokens=128,
                            do_sample=False,
                            pad_token_id=tok.eos_token_id,
                            use_cache=False,
                        )[0, len(ids) :].tolist()
                    samples.append(
                        {
                            "prompt": prompt,
                            "prefix_bos": prefix_bos,
                            "precision": precision,
                            "generation_ids": output,
                            "completion": tok.decode(output, skip_special_tokens=True),
                        }
                    )
    return {
        "repo": repo,
        "revision": revision,
        "api": "transformers model.generate do_sample=False use_cache=False",
        "samples": samples,
    }


@app.local_entrypoint()
def main(output: str = "artifacts/repetition-baselines/pythia-verification.json"):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verify.remote(time.time() + LAUNCH_LIMIT_SECONDS), indent=2))
