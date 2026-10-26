"""Gradio playground for a hybrid-family checkpoint, on CPU with incremental decoding.

Runs inside a Hugging Face Space assembled by publish.py: the model code is copied
next to this file (model_hybrid/, model_10m/), and the export named in
model-settings.json is downloaded from its private model repo with the Space's
HF_TOKEN secret.
"""

import json
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from model_hybrid import generate
from model_hybrid import tokenizer as glm

HERE = Path(__file__).parent
SETTINGS = json.loads((HERE / "model-settings.json").read_text())
torch.set_num_threads(max(1, torch.get_num_threads()))
EXPORT = Path(snapshot_download(SETTINGS["repo"], allow_patterns=[f"{SETTINGS['export']}/*"]))
MODEL = generate.load_export(EXPORT / SETTINGS["export"])
TOKENIZER = glm.load()


def stream(prompt_ids, temperature, top_p, max_new):
    started, ids = time.time(), []
    generator = torch.Generator().manual_seed(0)
    for token in generate.generate(
        MODEL, prompt_ids, int(max_new), glm.EOS_ID, temperature, top_p, generator
    ):
        if token == glm.EOS_ID:
            break
        ids.append(token)
        yield ids, len(ids) / max(time.time() - started, 1e-6)


def continue_text(prefix, temperature, top_p, max_new):
    prompt = TOKENIZER.encode(prefix, add_special_tokens=False).ids
    for ids, rate in stream(prompt, temperature, top_p, max_new):
        yield prefix + TOKENIZER.decode(ids), f"{len(ids)} tokens · {rate:.1f} tokens/s"


EXAMPLES = [
    "Photosynthesis is the process by which",
    "def is_palindrome(s: str) -> bool:",
    "Question: A train travels 120 km in 2 hours. What is its average speed?\nAnswer:",
    "The history of the printing press",
]

with gr.Blocks(title="Hybrid 720M playground") as demo:
    gr.Markdown(f"## {SETTINGS['name']}\n{SETTINGS['description']}")
    with gr.Row():
        temperature = gr.Slider(0, 1.5, value=0.7, step=0.05, label="Temperature (0 = greedy)")
        top_p = gr.Slider(0.1, 1, value=0.9, step=0.05, label="Top-p")
        max_new = gr.Slider(16, 1024, value=160, step=16, label="Max new tokens")
    prefix = gr.Textbox(lines=6, label="Prompt", value=EXAMPLES[0])
    go = gr.Button("Continue", variant="primary")
    output = gr.Textbox(lines=18, label="Text")
    rate = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=prefix)
    go.click(continue_text, [prefix, temperature, top_p, max_new], [output, rate])

if __name__ == "__main__":
    demo.queue().launch()
