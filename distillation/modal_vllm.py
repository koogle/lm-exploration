"""Export and qualify the student's standard vLLM backend on a bounded L4."""

import re

import modal

app = modal.App("lm-distillation-vllm", include_source=False)
volume = modal.Volume.from_name("lm-exploration-distillation")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "tokenizers==0.22.2")
    .env({"PYTHONPATH": "/project", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
    .add_local_dir(
        "distillation", "/project/distillation", ignore=["artifacts", "__pycache__"]
    )
    .add_local_dir("training/100m", "/project/model_100m")
    .add_local_dir("training/10m", "/project/model_10m")
    .add_local_dir(
        "training/data", "/project/training_data", ignore=["eu", "__pycache__"]
    )
    .add_local_dir(
        "training/tokenizer/structured-code-v1", "/project/tokenizer/structured-code-v1"
    )
)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=900,
    retries=0,
    volumes={"/distill": volume},
)
def qualify(checkpoint):
    import json
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    from distillation.export import export_checkpoint

    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}/step-[0-9]{9}\.pt", checkpoint):
        raise ValueError("Expected a distillation checkpoint")
    export = Path("/distill") / (checkpoint.removesuffix(".pt") + "-hf")
    report = export_checkpoint("/distill/" + checkpoint, export)
    tokenizer = AutoTokenizer.from_pretrained(export)
    prompts = [
        "def add(a, b):\n    return",
        "class Stack:\n    def __init__(self):",
        "for i in range(10):\n    print(",
        "# café\nx = ",
        "function sum(a, b) {\n  return",
    ]
    prefix = tokenizer.bos_token_id
    if prefix is None:
        prefix = tokenizer.eos_token_id
    token_ids = [
        [prefix] + tokenizer.encode(p, add_special_tokens=False) for p in prompts
    ]
    reference = AutoModelForCausalLM.from_pretrained(
        export, torch_dtype=torch.float32
    ).eval()
    expected = []
    with torch.no_grad():
        for ids in token_ids:
            logits = reference(torch.tensor([ids])).logits[0, -1].float()
            token = logits.argmax().item()
            expected.append((token, logits.log_softmax(-1)[token].item()))
    del reference
    llm = LLM(
        model=str(export),
        dtype="float32",
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=False,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in token_ids],
        SamplingParams(temperature=0, max_tokens=1, logprobs=1),
    )
    errors = []
    for result, (token, logprob) in zip(outputs, expected, strict=True):
        actual = result.outputs[0]
        if actual.token_ids != [token]:
            raise ValueError("vLLM greedy token mismatch")
        error = abs(actual.logprobs[0][token].logprob - logprob)
        if error > 1e-3:
            raise ValueError(f"vLLM log probability mismatch: {error}")
        errors.append(error)
    report.update(
        vllm_version="0.11.0",
        prompts=len(prompts),
        max_logprob_error=max(errors),
        export=str(export),
        dtype="float32",
    )
    (export / "vllm-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main(checkpoint: str):
    print(qualify.remote(checkpoint))
