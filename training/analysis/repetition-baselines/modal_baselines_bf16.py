"""Bounded external base-model comparison. Run: modal run modal_baselines.py."""

import json
from pathlib import Path

import modal

app = modal.App("simple-llm-repetition-baselines")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.6.0", "transformers==4.51.3", "huggingface-hub==0.30.2", "safetensors", "numpy"
)
MODELS = [
    (
        "pythia-160m-4b",
        "EleutherAI/pythia-160m",
        "a4817f95aaebbfddd0763a6a027f5be6bd2c9246",
        4194304000,
    ),
    (
        "pythia-160m-300b",
        "EleutherAI/pythia-160m",
        "50f5173d932e8e61f858120bcb800b97af589f46",
        299892736000,
    ),
    (
        "smollm-135m-600b",
        "HuggingFaceTB/SmolLM-135M",
        "1d461723eec654e65efdc40cf49301c89c0c92f4",
        600000000000,
    ),
]
PROMPTS = [
    "The solar system consists of",
    "In mathematics, a prime number is",
    "The purpose of an experiment is to",
    "Once upon a time,",
    "def add(a, b):\n    ",
    "def factorial(n):\n    ",
    "import json\n\ndef read_config(path):\n    ",
    "The capital of France is",
    "2 + 2 =",
    "A language model predicts",
]
MODES = [
    {"name": "greedy", "temperature": 0, "seed_offset": 1000},
    {"name": "topk50_t0.1_seed1000", "temperature": 0.1, "top_k": 50, "seed_offset": 1000},
    {"name": "topk50_t0.1_seed2000", "temperature": 0.1, "top_k": 50, "seed_offset": 2000},
    {"name": "sampled_08_seed1", "temperature": 0.8, "top_k": 50, "seed_offset": 1000},
    {"name": "sampled_08_seed2", "temperature": 0.8, "top_k": 50, "seed_offset": 2000},
    {"name": "sampled_10_topp95", "temperature": 1.0, "top_p": 0.95, "seed_offset": 1000},
]


@app.function(
    image=image, gpu="H100", cpu=4, memory=12288, timeout=1800, retries=0, max_containers=3
)
def evaluate(spec):
    import time

    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    name, repo, branch, budget = spec
    revision = HfApi().model_info(repo, revision=branch).sha
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, torch_dtype=torch.float32, attn_implementation="eager"
        )
        .cuda()
        .eval()
    )
    torch.set_num_threads(4)
    rows = []
    with torch.inference_mode():
        for pi, prompt in enumerate(PROMPTS):
            # Respect each model's native prefix convention: Pythia EOS-as-BOS; SmolLM has no BOS.
            prefix = tokenizer.encode(prompt, add_special_tokens=True)
            if model.config.model_type == "gpt_neox":
                prefix = [tokenizer.bos_token_id or tokenizer.eos_token_id, *prefix]
            for mode in MODES:
                seed = mode["seed_offset"] + pi
                torch.manual_seed(seed)
                tokens = torch.tensor([prefix], device="cuda")
                generated = []
                for _ in range(128):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(tokens).logits[0, -1].float()
                    if mode["temperature"] == 0:
                        token = int(logits.argmax())
                    elif "top_k" in mode:
                        values, indices = torch.topk(logits / mode["temperature"], mode["top_k"])
                        token = int(indices[torch.multinomial(values.softmax(-1), 1)])
                    else:
                        values, indices = torch.sort(logits / mode["temperature"], descending=True)
                        probs = values.softmax(-1)
                        remove = probs.cumsum(-1) > 0.95
                        remove[1:] = remove[:-1].clone()
                        remove[0] = False
                        values[remove] = -float("inf")
                        token = int(indices[torch.multinomial(values.softmax(-1), 1)])
                    generated.append(token)
                    tokens = torch.cat([tokens, torch.tensor([[token]], device="cuda")], dim=1)
                    if token == tokenizer.eos_token_id:
                        break
                rows.append(
                    {
                        "prompt": prompt,
                        "mode": mode["name"],
                        "seed": seed,
                        "completion": tokenizer.decode(generated, skip_special_tokens=True),
                        "generation_ids": generated,
                        "generated_tokens": len(generated),
                        "ended_with_eos": generated[-1] == tokenizer.eos_token_id,
                    }
                )
    return {
        "name": name,
        "repo": repo,
        "revision": revision,
        "requested_revision": branch,
        "trained_tokens": budget,
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": torch.cuda.get_device_name(),
        "prefix_policy": "native tokenizer special tokens; Pythia EOS-as-BOS prepended",
        "elapsed_seconds": time.monotonic() - started,
        "samples": rows,
    }


@app.local_entrypoint()
def main():
    output = Path(__file__).with_name("external-results-bf16.json")
    results = []
    for result in evaluate.map(MODELS):
        results.append(result)
        output.write_text(
            json.dumps(
                {"prompts": PROMPTS, "modes": MODES, "max_new_tokens": 128, "models": results},
                indent=2,
            )
        )
        print(result["name"], result["revision"], len(result["samples"]), flush=True)
