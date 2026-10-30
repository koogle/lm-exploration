"""Gradio playground for a hybrid-family checkpoint, decoding on the Space's GPU.

Runs inside a Hugging Face Space assembled by publish.py: the model code is copied
next to this file (model_hybrid/, model_10m/), and the export named in
model-settings.json is downloaded from its private model repo with the Space's
HF_TOKEN secret. Generation is serialized: one request decodes at a time. Chat exports (`"chat": true`) get a chat tab: OpenAI messages and
tools rendered in GLM's chat format, with each reply parsed back into an OpenAI
assistant message (content and tool_calls); tool results are sent as `tool` turns.
"""

import json
import threading
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from model_hybrid import chat, generate, sft_data
from model_hybrid import tokenizer as glm

HERE = Path(__file__).parent
SETTINGS = json.loads((HERE / "model-settings.json").read_text())
EXPORT = Path(snapshot_download(SETTINGS["repo"], allow_patterns=[f"{SETTINGS['export']}/*"]))
MODEL = generate.load_export(EXPORT / SETTINGS["export"], dtype=torch.bfloat16, device="cuda")
TOKENIZER = glm.load()
# Decoders are reused per shape, so requests take turns.
LOCK = threading.Lock()
# Compile prefill attention and capture the decode graph before the first visitor.
for _ in generate.generate(MODEL, TOKENIZER.encode("Hello").ids, 8, ()):
    pass


def stream(prompt_ids, temperature, top_p, max_new, stop_ids=(glm.EOS_ID,)):
    with LOCK:
        started, ids = time.time(), []
        generator = torch.Generator(device="cuda").manual_seed(0)
        for token in generate.generate(
            MODEL, prompt_ids, int(max_new), stop_ids, temperature, top_p, generator
        ):
            if token in stop_ids:
                break
            ids.append(token)
            yield ids, len(ids) / max(time.time() - started, 1e-6)


def continue_text(prefix, temperature, top_p, max_new):
    prompt = TOKENIZER.encode(prefix, add_special_tokens=False).ids
    for ids, rate in stream(prompt, temperature, top_p, max_new):
        yield prefix + TOKENIZER.decode(ids), f"{len(ids)} tokens · {rate:.1f} tokens/s"


def display(messages):
    """OpenAI messages as Gradio chat bubbles (tool calls and results shown as JSON)."""
    out = []
    for message in messages:
        if message["role"] == "tool":
            out.append(
                {
                    "role": "user",
                    "content": message["content"],
                    "metadata": {"title": "Tool result"},
                }
            )
            continue
        if message.get("content"):
            out.append({"role": message["role"], "content": message["content"]})
        for call in message.get("tool_calls") or []:
            function = call["function"]
            out.append(
                {
                    "role": "assistant",
                    "content": f"`{function['name']}({function['arguments']})`",
                    "metadata": {"title": "Tool call"},
                }
            )
    return out


def respond(text, role, system, tools_json, history, temperature, top_p, max_new):
    tools = json.loads(tools_json) if tools_json.strip() else None
    history = [*history, {"role": role, "content": text}]
    system_turn = [{"role": "system", "content": system}] if system.strip() else []
    prompt = chat.render(system_turn + history, tools, add_generation_prompt=True)
    prompt_ids = TOKENIZER.encode(prompt, add_special_tokens=False).ids
    reply, status = "", ""
    for ids, rate in stream(prompt_ids, temperature, top_p, max_new, sft_data.STOP_IDS):
        reply = TOKENIZER.decode(ids, skip_special_tokens=False)
        status = f"{len(ids)} tokens · {rate:.1f} tokens/s"
        partial = display(history) + [{"role": "assistant", "content": reply}]
        yield partial, history, "", None, status
    message = chat.parse_assistant(reply)
    history = [*history, message]
    yield display(history), history, "", message, status


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
        max_new = gr.Slider(16, 1024, value=320, step=16, label="Max new tokens")
    if SETTINGS.get("chat"):
        with gr.Tab("Chat"):
            history = gr.State([])
            with gr.Row():
                system = gr.Textbox(lines=2, label="System prompt")
                tools_json = gr.Code(
                    json.dumps([sft_data.WEATHER_TOOL], indent=1),
                    language="json",
                    label="Tools (OpenAI format; empty for none)",
                )
            chatbot = gr.Chatbot(height=420)
            with gr.Row():
                text = gr.Textbox(label="Message", scale=4)
                role = gr.Radio(["user", "tool"], value="user", label="Send as", scale=1)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
            chat_rate = gr.Markdown()
            parsed = gr.JSON(label="Last reply as an OpenAI assistant message")
            inputs = [text, role, system, tools_json, history, temperature, top_p, max_new]
            outputs = [chatbot, history, text, parsed, chat_rate]
            send.click(respond, inputs, outputs)
            text.submit(respond, inputs, outputs)
            clear.click(lambda: ([], [], None), None, [chatbot, history, parsed])
    with gr.Tab("Continue text"):
        prefix = gr.Textbox(lines=6, label="Prompt", value=EXAMPLES[0])
        go = gr.Button("Continue", variant="primary")
        output = gr.Textbox(lines=18, label="Text")
        rate = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=prefix)
        go.click(continue_text, [prefix, temperature, top_p, max_new], [output, rate])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
