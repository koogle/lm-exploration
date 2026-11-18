"""Gradio playground for hybrid-family checkpoints, decoding on the Space's GPU.

Runs inside a Hugging Face Space assembled by publish.py: the model code is copied
next to this file (model_hybrid/, model_10m/), and every export listed in
model-settings.json is downloaded from its private model repo with the Space's
HF_TOKEN secret and kept on the GPU. Generation is serialized: one request decodes at
a time.

- Compare: the same prompts on several exports, side by side, as text continuation,
  a chat turn, or a chat turn with thinking; the table fills in as each reply
  finishes and can be downloaded as JSON.
- Chat (when an export is a chat model, `"chat": true`): OpenAI messages and tools
  rendered in GLM's chat format, with each reply parsed back into an OpenAI assistant
  message (content and tool_calls); tool results are sent as `tool` turns.
- Continue text: one export continues a prompt.
"""

import json
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from model_hybrid import generate
from training_data import chat, chat_sft

from tokenizer import glm

HERE = Path(__file__).parent
SETTINGS = json.loads((HERE / "model-settings.json").read_text())
TOKENIZER = glm.load()


def load(entry):
    folder = Path(snapshot_download(entry["repo"], allow_patterns=[f"{entry['export']}/*"]))
    model = generate.load_export(folder / entry["export"], dtype=torch.bfloat16, device="cuda")
    # Compile prefill attention and capture the decode graph before the first visitor.
    for _ in generate.generate(model, TOKENIZER.encode("Hello").ids, 8, ()):
        pass
    return model


ENTRIES = {entry["label"]: entry for entry in SETTINGS["models"]}
MODELS = {label: load(entry) for label, entry in ENTRIES.items()}
CHAT_MODELS = [label for label, entry in ENTRIES.items() if entry.get("chat")]
# Decoders are reused per shape, so requests take turns.
LOCK = threading.Lock()
FORMATS = ("Continue text", "Chat", "Chat with thinking")
SEPARATOR = "\n---\n"


def stream(label, prompt_ids, temperature, top_p, max_new, stop_ids=(glm.EOS_ID,), seed=0):
    with LOCK:
        started, ids = time.time(), []
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        for token in generate.generate(
            MODELS[label], prompt_ids, int(max_new), stop_ids, temperature, top_p, generator
        ):
            if token in stop_ids:
                break
            ids.append(token)
            yield ids, len(ids) / max(time.time() - started, 1e-6)


def prompt_ids(text, format_):
    """Token ids of one prompt in a Compare format, and the tokens that end the reply."""
    if format_ == "Continue text":
        return TOKENIZER.encode(text, add_special_tokens=False).ids, (glm.EOS_ID,)
    rendered = chat.render(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        thinking=format_ == "Chat with thinking",
    )
    return TOKENIZER.encode(rendered, add_special_tokens=False).ids, chat_sft.STOP_IDS


def split_prompts(text):
    return [p.strip() for p in text.split(SEPARATOR) if p.strip()]


def compare(prompts_text, labels, format_, temperature, top_p, max_new, seed):
    """Fill a prompts x models table, one reply at a time, then offer it as JSON."""
    prompts, labels = split_prompts(prompts_text), list(labels)
    if not prompts or not labels:
        raise gr.Error("Enter at least one prompt and pick at least one model.")
    table = [[prompt] + [""] * len(labels) for prompt in prompts]
    headers = ["Prompt"] + labels
    results = []
    for row, prompt in enumerate(prompts):
        ids_, stop_ids = prompt_ids(prompt, format_)
        for column, label in enumerate(labels, start=1):
            reply, count, rate, last = "", 0, 0.0, 0.0
            for ids, rate in stream(label, ids_, temperature, top_p, max_new, stop_ids, seed):
                count = len(ids)
                if time.time() - last > 0.5:
                    reply, last = TOKENIZER.decode(ids, skip_special_tokens=False), time.time()
                    table[row][column] = reply
                    yield (
                        gr.update(value={"headers": headers, "data": table}),
                        f"{label}: {count} tokens",
                        None,
                    )
            reply = TOKENIZER.decode(ids, skip_special_tokens=False) if count else ""
            table[row][column] = reply
            results.append(
                {
                    "prompt": prompt,
                    "model": label,
                    "repo": ENTRIES[label]["repo"],
                    "export": ENTRIES[label]["export"],
                    "reply": reply,
                    "tokens": count,
                    "finished": count < int(max_new),
                    "tokens_per_second": round(rate, 1),
                }
            )
            yield (
                gr.update(value={"headers": headers, "data": table}),
                f"{label}: {count} tokens",
                None,
            )
    record = {
        "format": format_,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": int(max_new),
            "seed": int(seed),
        },
        "results": results,
    }
    path = Path(tempfile.mkdtemp()) / "comparison.json"
    path.write_text(json.dumps(record, indent=1))
    yield (
        gr.update(value={"headers": headers, "data": table}),
        f"Done: {len(results)} replies",
        str(path),
    )


def continue_text(label, prefix, temperature, top_p, max_new):
    prompt = TOKENIZER.encode(prefix, add_special_tokens=False).ids
    for ids, rate in stream(label, prompt, temperature, top_p, max_new):
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


def respond(label, text, role, system, tools_json, history, temperature, top_p, max_new):
    tools = json.loads(tools_json) if tools_json.strip() else None
    history = [*history, {"role": role, "content": text}]
    system_turn = [{"role": "system", "content": system}] if system.strip() else []
    prompt = chat.render(system_turn + history, tools, add_generation_prompt=True)
    prompt_ids_ = TOKENIZER.encode(prompt, add_special_tokens=False).ids
    reply, status = "", ""
    for ids, rate in stream(label, prompt_ids_, temperature, top_p, max_new, chat_sft.STOP_IDS):
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
PROMPT_SETS = {
    "Code completion": SEPARATOR.join(
        [
            'def is_palindrome(s: str) -> bool:\n    """Return True if s reads the same '
            'forwards and backwards, ignoring case and non-alphanumeric characters."""\n',
            'def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    """Merge '
            'all overlapping intervals and return them sorted by start."""\n',
            "# Read n and a list of n integers from stdin and print the length of the longest\n"
            "# strictly increasing subsequence.\nimport sys\n",
        ]
    ),
    "Contest problems": SEPARATOR.join(
        [
            "You are given an integer array nums. Return the number of pairs (i, j) with "
            "i < j such that nums[i] + nums[j] is divisible by 3.\n\nWrite a Python program "
            "that reads the array from stdin (n on the first line, then n integers) and "
            "prints the answer.",
            "Given a string s of lowercase letters, find the length of the longest substring "
            "in which every letter appears an even number of times. Read s from stdin and "
            "print the answer.",
            "There are N cities connected by M bidirectional roads with lengths. Find the "
            "shortest distance from city 1 to every city, printing -1 for unreachable "
            "cities. Input: N M, then M lines u v w. Output N integers on one line.",
        ]
    ),
    "General": SEPARATOR.join(
        [
            "Explain in two sentences why the sky is blue.",
            "A train travels 120 km in 2 hours. What is its average speed?",
            "Write a haiku about debugging.",
        ]
    ),
}
LABELS = list(MODELS)

with gr.Blocks(title=SETTINGS["name"]) as demo:
    rows = "\n".join(
        f"| {label} | `{entry['repo']}/{entry['export']}` | {'chat' if entry.get('chat') else 'base'} |"
        for label, entry in ENTRIES.items()
    )
    gr.Markdown(
        f"## {SETTINGS['name']}\n{SETTINGS['description']}\n\n"
        f"| Model | Export | Kind |\n|---|---|---|\n{rows}"
    )
    with gr.Row():
        temperature = gr.Slider(0, 1.5, value=0.0, step=0.05, label="Temperature (0 = greedy)")
        top_p = gr.Slider(0.1, 1, value=0.95, step=0.05, label="Top-p")
        max_new = gr.Slider(16, 16384, value=512, step=16, label="Max new tokens")
        seed = gr.Number(0, precision=0, label="Seed")
    with gr.Tab("Compare"):
        with gr.Row():
            prompt_set = gr.Dropdown(
                list(PROMPT_SETS), value="Contest problems", label="Prompt set"
            )
            format_ = gr.Radio(FORMATS, value="Chat with thinking", label="Format")
        prompts_text = gr.Textbox(
            PROMPT_SETS["Contest problems"],
            lines=10,
            label="Prompts (separate prompts with a line containing only ---)",
        )
        labels = gr.CheckboxGroup(LABELS, value=LABELS, label="Models")
        with gr.Row():
            run = gr.Button("Run all", variant="primary")
            stop = gr.Button("Stop")
        progress = gr.Markdown()
        grid = gr.Dataframe(headers=["Prompt"] + LABELS, wrap=True, interactive=False)
        download = gr.File(label="Results (JSON)")
        prompt_set.change(lambda name: PROMPT_SETS[name], prompt_set, prompts_text)
        event = run.click(
            compare,
            [prompts_text, labels, format_, temperature, top_p, max_new, seed],
            [grid, progress, download],
        )
        stop.click(None, None, None, cancels=[event])
    if CHAT_MODELS:
        with gr.Tab("Chat"):
            chat_model = gr.Dropdown(CHAT_MODELS, value=CHAT_MODELS[-1], label="Model")
            history = gr.State([])
            with gr.Row():
                system = gr.Textbox(lines=2, label="System prompt")
                tools_json = gr.Code(
                    json.dumps([chat_sft.WEATHER_TOOL], indent=1),
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
            inputs = [chat_model, text, role, system, tools_json, history, temperature, top_p]
            inputs.append(max_new)
            outputs = [chatbot, history, text, parsed, chat_rate]
            send.click(respond, inputs, outputs)
            text.submit(respond, inputs, outputs)
            clear.click(lambda: ([], [], None), None, [chatbot, history, parsed])
    with gr.Tab("Continue text"):
        text_model = gr.Dropdown(LABELS, value=LABELS[0], label="Model")
        prefix = gr.Textbox(lines=6, label="Prompt", value=EXAMPLES[0])
        go = gr.Button("Continue", variant="primary")
        output = gr.Textbox(lines=18, label="Text")
        rate = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=prefix)
        go.click(continue_text, [text_model, prefix, temperature, top_p, max_new], [output, rate])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
