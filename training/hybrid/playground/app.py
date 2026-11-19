"""Gradio playground for hybrid-family checkpoints, decoding on the Space's GPU.

Runs inside a Hugging Face Space assembled by publish.py: the model code is copied
next to this file (model_hybrid/, model_10m/), and every export listed in
model-settings.json is downloaded from its private model repo with the Space's
HF_TOKEN secret and kept on the GPU. Generation is serialized: one request decodes at
a time.

- Compare: a table with one row per prompt (a Run box to pick which prompts run, the
  prompt, then one column per export). Ticked prompts run in turn; each prompt's
  replies from the selected exports stream into their columns side by side, as text
  continuation, a chat turn, or a chat turn with thinking. The results download as
  JSON with the sampling settings.
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


def stream_together(labels, prompt_ids_, stop_ids, temperature, top_p, max_new, seed):
    """Decode one prompt on several exports in turn, a token each per round, yielding
    {label: ids so far} and the labels whose reply ended on a stop token. Each export
    has its own decoder and sampler, so every reply is the one it would decode alone."""
    with LOCK:
        live = {
            label: generate.generate(
                MODELS[label],
                prompt_ids_,
                int(max_new),
                stop_ids,
                temperature,
                top_p,
                torch.Generator(device="cuda").manual_seed(int(seed)),
            )
            for label in labels
        }
        ids, stopped = {label: [] for label in labels}, set()
        while live:
            for label in list(live):
                token = next(live[label], None)
                if token is None or token in stop_ids:
                    del live[label]
                    if token is not None:
                        stopped.add(label)
                else:
                    ids[label].append(token)
            yield ids, stopped


def checked(value):
    return value is True or str(value).strip().lower() in ("true", "1", "yes", "x")


def table_rows(prompts):
    return [[True, prompt] + [""] * len(LABELS) for prompt in prompts]


def compare(rows, labels, format_, temperature, top_p, max_new, seed):
    """Stream every selected export's reply to every prompt whose Run box is ticked
    into the table (prompts in turn, exports side by side), then offer it as JSON."""
    rows = [list(row) for row in rows if str(row[1]).strip()]
    labels = [label for label in LABELS if label in labels]
    chosen = [index for index, row in enumerate(rows) if checked(row[0])]
    if not chosen or not labels:
        raise gr.Error("Tick at least one prompt's Run box and pick at least one model.")
    columns = {label: 2 + LABELS.index(label) for label in labels}
    results = []
    for done, index in enumerate(chosen):
        prompt = rows[index][1]
        for column in columns.values():
            rows[index][column] = ""
        ids_, stop_ids = prompt_ids(prompt, format_)
        started, last = time.time(), 0.0
        ids, stopped = {label: [] for label in labels}, set()
        for ids, stopped in stream_together(
            labels, ids_, stop_ids, temperature, top_p, max_new, seed
        ):
            if time.time() - last > 0.25:
                last = time.time()
                for label, column in columns.items():
                    rows[index][column] = TOKENIZER.decode(ids[label], skip_special_tokens=False)
                status = f"Prompt {done + 1} of {len(chosen)} · " + " · ".join(
                    f"{label}: {len(ids[label])}" for label in labels
                )
                yield rows, status, None
        seconds = max(time.time() - started, 1e-6)
        for label, column in columns.items():
            rows[index][column] = TOKENIZER.decode(ids[label], skip_special_tokens=False)
            results.append(
                {
                    "prompt": prompt,
                    "model": label,
                    "repo": ENTRIES[label]["repo"],
                    "export": ENTRIES[label]["export"],
                    "reply": rows[index][column],
                    "tokens": len(ids[label]),
                    "stopped": label in stopped,
                }
            )
        yield rows, f"Prompt {done + 1} of {len(chosen)} done in {seconds:.0f} s", None
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
    yield rows, f"Done: {len(results)} replies", str(path)


def add_prompt(rows, text):
    rows = [list(row) for row in rows if str(row[1]).strip()]
    if text.strip():
        rows.append([True, text] + [""] * len(LABELS))
    return rows, ""


def set_run(rows, value):
    return [[value, *list(row)[1:]] for row in rows if str(row[1]).strip()]


def clear_answers(rows):
    return [list(row)[:2] + [""] * len(LABELS) for row in rows if str(row[1]).strip()]


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
# Prompt sets offered in the Compare tab: {name: [prompt, ...]}.
PROMPT_SETS = json.loads((HERE / "prompt-sets.json").read_text())
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
                list(PROMPT_SETS), value="Contest problems", label="Load prompt set"
            )
            format_ = gr.Radio(FORMATS, value="Chat with thinking", label="Format")
        labels = gr.CheckboxGroup(LABELS, value=LABELS, label="Models to run")
        with gr.Row():
            run = gr.Button("Run ticked prompts", variant="primary")
            stop = gr.Button("Stop")
            tick_all = gr.Button("Tick all")
            tick_none = gr.Button("Untick all")
            clear = gr.Button("Clear answers")
        progress = gr.Markdown()
        grid = gr.Dataframe(
            table_rows(PROMPT_SETS["Contest problems"]),
            headers=["Run", "Prompt", *LABELS],
            datatype=["bool", "str", *["str"] * len(LABELS)],
            static_columns=list(range(2, 2 + len(LABELS))),
            pinned_columns=2,
            column_widths=["70px", "320px", *["420px"] * len(LABELS)],
            type="array",
            wrap=True,
            interactive=True,
            max_height=900,
        )
        with gr.Row():
            new_prompt = gr.Textbox(lines=3, label="New prompt", scale=4)
            add = gr.Button("Add prompt", scale=1)
        download = gr.File(label="Results (JSON)")
        prompt_set.change(lambda name: table_rows(PROMPT_SETS[name]), prompt_set, grid)
        add.click(add_prompt, [grid, new_prompt], [grid, new_prompt])
        tick_all.click(lambda rows: set_run(rows, True), grid, grid)
        tick_none.click(lambda rows: set_run(rows, False), grid, grid)
        clear.click(clear_answers, grid, grid)
        event = run.click(
            compare,
            [grid, labels, format_, temperature, top_p, max_new, seed],
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
