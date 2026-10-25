"""Compare base and math-SFT 124M models on one prompt, served on a CPU Hugging Face Space.

`models.json` lists each model's label, weights file and provenance, and the private model
repository and commit holding the weights (downloaded with the Space's `HF_TOKEN` secret).
Weights load into `model.py`, the decoder code the checkpoints were trained with (copied by
`modal_export.py`, which also checked that it reproduces each checkpoint's logits).
Generation runs token by token with a per-layer key/value cache.
"""

import json
import os
from pathlib import Path

import gradio as gr
import torch
from generation import complete
from huggingface_hub import hf_hub_download
from model import GPT, ModelConfig
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


def generate(prompt, question_format, tokens, temperature, top_k, top_p, penalty, seed, blank):
    if question_format:
        prompt = f"Question: {prompt.strip()}\nAnswer:"
    outputs = ["…"] * len(MODELS)
    for index, model in enumerate(MODELS.values()):
        ids = [BOS, *tokenizer.encode(prompt, add_special_tokens=False).ids]
        generated = complete(
            model,
            ids,
            int(tokens),
            temperature,
            int(top_k),
            top_p,
            penalty,
            int(seed),
            EOS,
            BUNDLE["config"]["block_size"],
            stop=(lambda g: "\n\n" in tokenizer.decode(g)) if blank else None,
        )
        text = tokenizer.decode(generated)
        outputs[index] = text.split("\n\n")[0] if blank else text
        yield outputs


with gr.Blocks(title="ClimbMix 124M: base vs math SFT") as demo:
    gr.Markdown(Path("intro.md").read_text())
    prompt = gr.Textbox(
        label="Prompt",
        lines=4,
        value="Tom has 3 apples and buys 5 more. How many apples does he have?",
    )
    with gr.Row():
        question_format = gr.Checkbox(
            value=True, label="Wrap as `Question: …\\nAnswer:` (the math SFT format)"
        )
        blank = gr.Checkbox(value=True, label="Stop at a blank line")
    with gr.Row():
        tokens = gr.Slider(8, 384, value=192, step=8, label="New tokens")
        temperature = gr.Slider(0, 1.5, value=0, step=0.05, label="Temperature (0 = greedy)")
        top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 = off)")
        top_p = gr.Slider(0.05, 1, value=0.95, step=0.05, label="Top-p")
        penalty = gr.Slider(1, 1.5, value=1.0, step=0.05, label="Repetition penalty")
        seed = gr.Number(value=42, precision=0, label="Seed")
    run = gr.Button("Generate with every model", variant="primary")
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

    inputs = [prompt, question_format, tokens, temperature, top_k, top_p, penalty, seed, blank]
    run.click(ordered, inputs, boxes)
    gr.Examples(
        [
            [
                "A train travels 60 miles in 2 hours. What is its average speed in miles per hour?",
                True,
            ],
            [
                "Sara has 24 stickers. She gives a third of them to her brother. How many are left?",
                True,
            ],
            ["The solar system consists of", False],
            ["def factorial(n):\n    ", False],
        ],
        [prompt, question_format],
    )

demo.queue(default_concurrency_limit=1).launch()
