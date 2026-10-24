"""vLLM generation for the teacher and the students (one GPU, greedy).

Modes:
- ``tests``: teacher writes hidden asserts for held-out curriculum problems.
- ``solve``: teacher solves eval items through its chat template (the ceiling).
- ``student``: a student completes the training-format prompt. vLLM runs our decoder
  (``vllm_model.py``) on the training checkpoint itself, without a tokenizer; prompts
  and outputs are student IDs mapped through the student's vocabulary, and a parity
  check compares vLLM's prompt log probabilities with the PyTorch checkpoint.

Usage:
    python vllm_eval.py tests|solve ITEMS.json OUT.json
    python vllm_eval.py student ITEMS.json OUT.json --model-dir DIR --tokenizer tokenizer.json
"""

import argparse
import json
import os
import time
from pathlib import Path

from common import TEACHER, TEACHER_REVISION
from evalsets import (
    SOLVE_PROMPT,
    TEST_PROMPT,
    parse_code_string,
    parse_json_answer,
    student_prompt,
    valid_tests,
)

IM_END = 248046  # teacher stop token at the end of every accepted document
PARITY_PROMPTS = 8


def teacher(mode, items):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=TEACHER,
        revision=TEACHER_REVISION,
        tokenizer_revision=TEACHER_REVISION,
        dtype="bfloat16",
        download_dir="/runs/hf-cache",
        language_model_only=True,
        max_model_len=4096,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    template = TEST_PROMPT if mode == "tests" else SOLVE_PROMPT
    tokenizer = llm.get_tokenizer()
    prompts = [
        {
            "prompt_token_ids": tokenizer.apply_chat_template(
                [{"role": "user", "content": template.format(problem=item["problem"])}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        }
        for item in items
    ]
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=2048))
    results = {}
    for item, output in zip(items, outputs):
        text = output.outputs[0].text
        entry = {"text": text, "finish_reason": output.outputs[0].finish_reason}
        if mode == "tests":
            entry["tests"] = valid_tests(parse_json_answer(text, "tests"))
        else:
            code = parse_json_answer(text, "code")
            entry["code"] = code if isinstance(code, str) else None
        results[item["id"]] = entry
    return {"results": results}


def load_vocab(model_dir):
    import numpy as np
    from student import Vocabulary

    arrays = np.load(Path(model_dir) / "vocab.npz")
    pieces = (
        tuple(arrays[k] for k in ("pieces", "offsets", "lengths"))
        if "pieces" in arrays
        else (None,) * 3
    )
    return Vocabulary(arrays["table"], arrays["native"], int(arrays["bos"]), *pieces)


def parity(checkpoint, prompts, outputs):
    """Max/mean |log p| difference on prompt tokens between vLLM and PyTorch (FP32)."""
    import torch
    from student import GPT, ModelConfig

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = GPT(ModelConfig(**saved["config"])).eval()
    model.load_state_dict(saved["model"])
    differences = []
    with torch.no_grad():
        for prompt, output in zip(prompts, outputs):
            ids = torch.tensor(prompt["prompt_token_ids"])
            reference = model(ids[None])[0].float().log_softmax(-1)
            expected = reference[:-1].gather(1, ids[1:, None])[:, 0]
            rows = output.prompt_logprobs[1:]
            got = torch.tensor([row[int(i)].logprob for row, i in zip(rows, ids[1:])])
            differences.append((got - expected).abs())
    flat = torch.cat(differences)
    return {
        "prompts": len(prompts),
        "tokens": int(flat.numel()),
        "max_abs": float(flat.max()),
        "mean_abs": float(flat.mean()),
    }


def students(items, model_dir, tokenizer_path, max_tokens):
    # Run the engine in this process so the model registered below is visible to it.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from tokenizers import Tokenizer
    from vllm import LLM, ModelRegistry, SamplingParams
    from vllm_model import ARCHITECTURE, SimpleLLMForCausalLM

    ModelRegistry.register_model(ARCHITECTURE, SimpleLLMForCausalLM)
    vocab = load_vocab(model_dir)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos = int(vocab.encode([IM_END])[0])
    trimmed = vocab.size < len(vocab.table)
    prompts = [
        {
            "prompt_token_ids": [vocab.bos]
            + vocab.expand(
                tokenizer.encode(
                    student_prompt(item["problem"]), add_special_tokens=False
                ).ids
            )[0].tolist()
        }
        for item in items
    ]
    llm = LLM(
        model=str(model_dir),
        load_format="pt",  # the training checkpoint, model.pt
        skip_tokenizer_init=True,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=json.loads((Path(model_dir) / "config.json").read_text())[
            "max_position_embeddings"
        ],
        enable_prefix_caching=False,
        gpu_memory_utilization=0.7,
        seed=0,
    )
    check = llm.generate(
        prompts[:PARITY_PROMPTS],
        SamplingParams(
            temperature=0, max_tokens=1, prompt_logprobs=0, detokenize=False
        ),
    )
    params = SamplingParams(
        temperature=0, max_tokens=max_tokens, stop_token_ids=[eos], detokenize=False
    )
    outputs = llm.generate(prompts, params)
    results = {}
    for item, prompt, output in zip(items, prompts, outputs):
        ids = list(output.outputs[0].token_ids)
        text = tokenizer.decode(vocab.decode_native(ids), skip_special_tokens=False)
        results[item["id"]] = {
            "text": text,
            "code": parse_code_string(text),
            "finish_reason": output.outputs[0].finish_reason,
            "generated_tokens": len(ids),
            "unknown_tokens": ids.count(2) if trimmed else 0,
            "prompt_unknown_tokens": prompt["prompt_token_ids"].count(2)
            if trimmed
            else 0,
        }
    return {
        "results": results,
        "parity": parity(Path(model_dir) / "model.pt", prompts[:PARITY_PROMPTS], check),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tests", "solve", "student"))
    parser.add_argument("items")
    parser.add_argument("out")
    parser.add_argument("--model-dir")
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()
    items = json.loads(Path(args.items).read_text())
    started = time.time()
    if args.mode == "student":
        report = students(items, args.model_dir, args.tokenizer, args.max_tokens)
    else:
        report = teacher(args.mode, items)
    report.update(mode=args.mode, seconds=time.time() - started)
    Path(args.out).write_text(json.dumps(report))


if __name__ == "__main__":
    main()
