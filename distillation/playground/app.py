"""Gradio playground for a distillation student, served on a CPU Hugging Face Space.

Loads the training checkpoint (``model.pt``) into the repository's decoder
(``model.py``, copied from ``training/10m/model.py``) and generates token by token
with a per-layer key/value cache. Prompts use the training format: an exercise as
``{"problem": ..., "code": ..., "explanation": ...}`` JSON, or any raw prefix.
"""

import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from evalsets import parse_code_string, student_prompt
from model import GPT, ModelConfig, rotary
from tokenizers import Tokenizer
from torch.nn import functional as F

BOS = 1
IM_END = 248046  # native stop token that ends every training document
torch.set_num_threads(os.cpu_count() or 1)

saved = torch.load("model.pt", map_location="cpu", weights_only=True)
model = GPT(ModelConfig(**saved["config"])).eval()
model.load_state_dict(saved["model"])
tokenizer = Tokenizer.from_file("tokenizer.json")
table = np.load("vocab-table.npy")  # native ID -> student ID (2 = UNK)
native = np.concatenate([[-1, -1, -1], np.load("vocab-kept.npy")])  # student -> native
EOS = int(table[IM_END])
SETTINGS = json.loads(Path("model-settings.json").read_text())


def encode(text):
    return [BOS] + table[tokenizer.encode(text, add_special_tokens=False).ids].tolist()


def decode(ids):
    return tokenizer.decode([int(native[i]) for i in ids if native[i] >= 0])


@torch.no_grad()
def step(tokens, cache, offset):
    """Logits for the last of ``tokens`` at positions ``offset:``, extending ``cache``."""
    x = model.token_embedding(torch.tensor([tokens]))
    for layer, block in enumerate(model.blocks):
        attention = block.attention
        h = block.attention_norm(x)
        batch, length, width = h.shape
        heads = attention.n_head
        q, k, v = (
            t.view(batch, length, heads, width // heads).transpose(1, 2)
            for t in attention.qkv(h).chunk(3, -1)
        )
        q, k = rotary(q, offset), rotary(k, offset)
        if cache[layer] is not None:
            k = torch.cat([cache[layer][0], k], 2)
            v = torch.cat([cache[layer][1], v], 2)
        cache[layer] = (k, v)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=length > 1)
        x = x + attention.projection(out.transpose(1, 2).reshape(batch, length, width))
        x = x + block.mlp(block.mlp_norm(x))
    return model.lm_head(model.final_norm(x[0, -1]))


def sample(logits, temperature, top_p):
    if temperature <= 0:
        return int(logits.argmax())
    probs = (logits / temperature).softmax(-1)
    ordered, index = probs.sort(descending=True)
    keep = ordered.cumsum(-1) - ordered < top_p
    choice = torch.multinomial(ordered * keep, 1)
    return int(index[choice])


def generate(prompt_ids, max_new, temperature, top_p, finished):
    """Yield (generated IDs, tokens/s) after each new token."""
    cache = [None] * len(model.blocks)
    max_new = min(max_new, model.config.block_size - len(prompt_ids))
    logits = step(prompt_ids, cache, 0)
    generated, started = [], time.time()
    for position in range(len(prompt_ids), len(prompt_ids) + max_new):
        token = sample(logits, temperature, top_p)
        if token == EOS:
            break
        generated.append(token)
        yield generated, len(generated) / (time.time() - started)
        if finished(generated):
            break
        logits = step([token], cache, position)


def partial_code(text):
    """Code so far from the JSON string body, also while it is still open."""
    code = parse_code_string(text)
    if code is not None:
        return code
    try:
        return json.loads('"' + text.rstrip("\\") + '"')
    except json.JSONDecodeError:
        return text.replace("\\n", "\n").replace('\\"', '"')


def solve(problem, explain, temperature, top_p, max_new):
    prompt = student_prompt(problem.strip())

    def finished(ids):
        return not explain and parse_code_string(decode(ids)) is not None

    for ids, rate in generate(
        encode(prompt), int(max_new), temperature, top_p, finished
    ):
        text = decode(ids)
        yield (
            partial_code(text),
            prompt + text,
            f"{len(ids)} tokens · {rate:.1f} tokens/s",
        )


def continue_text(prefix, temperature, top_p, max_new):
    for ids, rate in generate(
        encode(prefix), int(max_new), temperature, top_p, lambda ids: False
    ):
        yield prefix + decode(ids), f"{len(ids)} tokens · {rate:.1f} tokens/s"


EXAMPLES = [
    (
        "Given a list of integers, return a new list with duplicates removed, keeping the "
        "first occurrence of each value in its original order. Implement it as "
        "`def solve(values):`."
    ),
    (
        "N zones start at 0. Each request [L, R, Q] adds Q to zones L through R - 1. "
        "Return the final counts. Implement it as `def solve(requests, n):`."
    ),
    (
        "Write a function that returns the n-th Fibonacci number, with solve(0) == 0 and "
        "solve(1) == 1. Implement it as `def solve(n):`."
    ),
]

with gr.Blocks(title="Distillation student playground") as demo:
    gr.Markdown(
        f"## {SETTINGS['name']}\n{SETTINGS['description']}\n\n"
        "Greedy decoding (temperature 0) matches the evaluation. Prompts follow the "
        "training format, so describe one Python exercise and the signature to implement."
    )
    with gr.Row():
        temperature = gr.Slider(0, 1.5, value=0, step=0.05, label="Temperature")
        top_p = gr.Slider(0.1, 1, value=0.95, step=0.05, label="Top-p")
        max_new = gr.Slider(16, 1536, value=512, step=16, label="Max new tokens")
    with gr.Tab("Solve a problem"):
        problem = gr.Textbox(lines=6, label="Problem", value=EXAMPLES[0])
        explain = gr.Checkbox(label="Continue with the explanation", value=False)
        run = gr.Button("Generate", variant="primary")
        code = gr.Code(language="python", label="Generated code")
        status = gr.Markdown()
        raw = gr.Textbox(lines=12, label="Raw model text (prompt + output)")
        gr.Examples(EXAMPLES, inputs=problem)
        run.click(
            solve, [problem, explain, temperature, top_p, max_new], [code, raw, status]
        )
    with gr.Tab("Free continuation"):
        prefix = gr.Textbox(lines=6, label="Prefix", value='{\n  "problem": "')
        go = gr.Button("Continue", variant="primary")
        output = gr.Textbox(lines=20, label="Text")
        rate = gr.Markdown()
        go.click(continue_text, [prefix, temperature, top_p, max_new], [output, rate])

if __name__ == "__main__":
    demo.queue().launch()
