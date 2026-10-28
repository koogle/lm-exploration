"""Compare 124M checkpoints on one prompt, served on a CPU Hugging Face Space.

`models.json` lists each model's label, description, weights file, provenance and quoted
benchmark scores, the display layout, and the private model
repository and commit holding the weights (downloaded with the Space's `HF_TOKEN` secret).
Weights load into `model.py`, the decoder code the checkpoints were trained with (copied by
`modal_export.py`, which also checked that it reproduces each checkpoint's logits).
Generation uses the repository's shared cached sampler (`rollout.py`, copied from
`trainer/rollout.py` by `modal_export.py`), one selected model after
another. A prompt format wraps the text the way the SFT and RL models were trained.
"""

import json
import os
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from model import GPT, ModelConfig
from rollout import sample
from safetensors.torch import load_file
from tokenizers import Tokenizer

torch.set_num_threads(os.cpu_count() or 1)
BUNDLE = json.loads(Path("models.json").read_text())
BOS, EOS = BUNDLE["special_tokens"]["bos"], BUNDLE["special_tokens"]["eos"]
tokenizer = Tokenizer.from_file("tokenizer.json")


def load(entry):
    path = hf_hub_download(
        BUNDLE["weights_repo"],
        entry["weights"],
        revision=BUNDLE["weights_revision"],
        token=os.environ.get("HF_TOKEN"),
    )
    model = GPT(ModelConfig(**BUNDLE["config"])).eval()
    missing, unexpected = model.load_state_dict(load_file(path), strict=False)
    # The output head shares the embedding matrix, so the file stores it once.
    if missing != ["lm_head.weight"] or unexpected:
        raise ValueError(f"Unexpected weights in {entry['weights']}")
    return model


MODELS = {entry["label"]: load(entry) for entry in BUNDLE["models"]}
# How each format wraps the prompt: the math and code SFT prompts (`training_data`).
FORMATS = {
    "Plain text": "{}",
    "Math question": "Question: {}\nAnswer:",
    "Code task": "Question: {}\nAnswer:\n",
}
SCORES = {
    "gsm8k": "GSM8K",
    "svamp": "SVAMP",
    "mbpp_plus": "MBPP+",
    "humaneval_plus": "HumanEval+",
    "hellaswag_acc_norm": "HellaSwag",
    "arc_easy_acc": "ARC-E",
}


def score_table():
    rows = []
    for entry in BUNDLE["models"]:
        scores = entry["scores"]
        values = [f"{100 * scores[key]:.1f}" if key in scores else "" for key in SCORES]
        rows.append([entry["label"], entry.get("description", ""), *values])
    return rows


def generate(
    prompt, prompt_format, selected, tokens, temperature, top_k, top_p, penalty, seed, blank
):
    # Plain text keeps trailing whitespace, such as the indentation after `def f():`.
    text = prompt if prompt_format == "Plain text" else prompt.strip()
    prompt = FORMATS[prompt_format].format(text)
    outputs = ["(not selected)" if label not in selected else "…" for label in MODELS]
    yield outputs
    for index, (label, model) in enumerate(MODELS.items()):
        if label not in selected:
            continue
        ids = [BOS, *tokenizer.encode(prompt, add_special_tokens=False).ids]
        # Keep the end of a prompt too long to leave room for the new tokens.
        ids = ids[-(BUNDLE["config"]["block_size"] - int(tokens)) :]
        samples = sample(
            model,
            [ids],
            int(tokens),
            EOS,
            temperature,
            torch.Generator().manual_seed(int(seed)),
            top_k=int(top_k),
            top_p=top_p,
            penalty=penalty,
            stop=(lambda g: "\n\n" in tokenizer.decode(g)) if blank else None,
        )
        text = tokenizer.decode(samples.completions[0])
        outputs[index] = text.split("\n\n")[0] if blank else text
        yield outputs


with gr.Blocks(title="ClimbMix 124M playground") as demo:
    gr.Markdown(Path("intro.md").read_text())
    gr.Dataframe(
        score_table(),
        headers=["Model", "What it is", *SCORES.values()],
        label="Models and benchmark scores (%)",
        interactive=False,
        wrap=True,
    )
    prompt = gr.Textbox(
        label="Prompt",
        lines=4,
        value="Tom has 3 apples and buys 5 more. How many apples does he have?",
    )
    with gr.Row():
        prompt_format = gr.Radio(list(FORMATS), value="Math question", label="Prompt format")
        blank = gr.Checkbox(value=True, label="Stop at a blank line")
    selected = gr.CheckboxGroup(
        list(MODELS), value=BUNDLE["default_models"], label="Models to run (one after another)"
    )
    with gr.Row():
        tokens = gr.Slider(8, 384, value=192, step=8, label="New tokens")
        temperature = gr.Slider(0, 1.5, value=0, step=0.05, label="Temperature (0 = greedy)")
        top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 = off)")
        top_p = gr.Slider(0.05, 1, value=0.95, step=0.05, label="Top-p")
        penalty = gr.Slider(1, 1.5, value=1.0, step=0.05, label="Repetition penalty")
        seed = gr.Number(value=42, precision=0, label="Seed")
    run = gr.Button("Generate with the selected models", variant="primary")
    boxes = []
    for pair in BUNDLE["layout"]:
        with gr.Row():
            for label in pair:
                boxes.append(gr.Textbox(label=label, lines=10, interactive=False))
    order = [label for pair in BUNDLE["layout"] for label in pair]
    if sorted(order) != sorted(MODELS):
        raise ValueError("layout must name every model once")

    def ordered(*args):
        for outputs in generate(*args):
            by_label = dict(zip(MODELS, outputs, strict=True))
            yield [by_label[label] for label in order]

    inputs = [prompt, prompt_format, selected, tokens, temperature, top_k, top_p, penalty]
    inputs += [seed, blank]
    run.click(ordered, inputs, boxes)
    gr.Examples(
        [
            [
                "A train travels 60 miles in 2 hours. What is its average speed in miles per hour?",
                "Math question",
                True,
            ],
            [
                "Sara has 24 stickers. She gives a third of them to her brother. How many are left?",
                "Math question",
                True,
            ],
            [
                (
                    "Return the n-th Fibonacci number, with fibonacci(0) == 0 and fibonacci(1) == 1."
                    "\nWrite a Python function `def fibonacci(n):`."
                ),
                "Code task",
                False,
            ],
            [
                (
                    "Given a list of words, return the longest one; return None for an empty list."
                    "\nWrite a Python function `def longest_word(words):`."
                ),
                "Code task",
                False,
            ],
            ["The solar system consists of", "Plain text", True],
        ],
        [prompt, prompt_format, blank],
    )

demo.queue(default_concurrency_limit=1).launch()
