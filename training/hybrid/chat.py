"""OpenAI-style chat messages and tool calls in GLM's chat format.

Conversations are lists of OpenAI chat messages: `{"role": "system" | "user" |
"assistant" | "tool", "content": str}`, where assistant messages may carry
`tool_calls` (`[{"type": "function", "function": {"name", "arguments"}}]`, arguments
as a JSON string or object) and `tools` is the OpenAI list of function schemas.
They are rendered with GLM-4.7-Flash's own chat template (pinned like the
tokenizer), so the token layout is GLM's: tools in a system turn, calls as
`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`, results
as `<|observation|><tool_response>...</tool_response>`. Assistant turns are rendered
without reasoning (`</think>` right after `<|assistant|>`).

`encode` returns token ids and a loss mask that is 1 on what the model must
generate: each assistant turn after its `<|assistant|></think>` prompt, plus the
token that ends the turn (the next role token, or EOS at the end).
`parse_assistant` turns generated text back into an OpenAI assistant message.
"""

import functools
import hashlib
import json
import re
from pathlib import Path

from . import tokenizer as glm

TEMPLATE = {**glm.SOURCE, "file": "chat_template.jinja"}
TEMPLATE_SHA256 = "d63ad536c3c81880043e22ec7fd08db42b4d8fb7c89c7138bc562bfa25281375"
ASSISTANT_PROMPT = "<|assistant|></think>"
TURN_END = ("<|user|>", "<|observation|>", "<|system|>")


@functools.cache
def template():
    """The pinned GLM chat template, compiled with jinja2."""
    import jinja2
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(TEMPLATE["repo"], TEMPLATE["file"], revision=TEMPLATE["revision"]))
    if hashlib.sha256(path.read_bytes()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError(f"{path} does not match the pinned GLM chat template hash")
    environment = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    environment.filters["tojson"] = lambda value, ensure_ascii=False: json.dumps(
        value, ensure_ascii=ensure_ascii
    )
    return environment.from_string(path.read_text())


def _arguments(call):
    function = call.get("function", call)
    arguments = function["arguments"]
    return {
        "name": function["name"],
        "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments,
    }


def normalize(messages):
    """Messages as the template expects them: tool-call arguments as objects."""
    out = []
    for message in messages:
        message = dict(message)
        if message.get("tool_calls"):
            message["tool_calls"] = [_arguments(call) for call in message["tool_calls"]]
        out.append(message)
    return out


def render(messages, tools=None, add_generation_prompt=False, thinking=False):
    """The conversation as GLM chat text; with `thinking`, every assistant turn keeps
    its reasoning (`<think>…</think>` in its content or `reasoning_content`)."""
    text = template().render(
        messages=normalize(messages),
        tools=tools or None,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=thinking,
        clear_thinking=not thinking,
    )
    return text


# Renderers a prepared source can name (`"render"`) to turn message rows into text.
RENDERERS = {"glm_chat_thinking": lambda messages: render(messages, thinking=True)}


def encode(tokenizer, messages, tools=None):
    """Token ids and loss mask (lists) of one conversation, ending with EOS."""
    segments, previous = [], ""
    for index in range(len(messages)):
        text = render(messages[: index + 1], tools)
        if not text.startswith(previous):
            raise ValueError("the chat template is not prefix-stable for these messages")
        segments.append((messages[index]["role"], text[len(previous) :]))
        previous = text
    ids, mask = [], []

    def add(text, trained):
        piece = tokenizer.encode(text, add_special_tokens=False).ids
        ids.extend(piece)
        mask.extend([int(trained)] * len(piece))

    for role, text in segments:
        if mask and mask[-1] and role != "assistant":
            # The token that ends an assistant turn is the next turn's role token.
            head = next(t for t in TURN_END if text.lstrip().startswith(t))
            start = text.index(head)
            add(text[:start], False)
            add(head, True)
            add(text[start + len(head) :], False)
        elif role == "assistant":
            start = text.index(ASSISTANT_PROMPT) + len(ASSISTANT_PROMPT)
            add(text[:start], False)
            add(text[start:], True)
        else:
            add(text, False)
    ids.append(glm.EOS_ID)
    mask.append(int(messages[-1]["role"] == "assistant"))
    return ids, mask


TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ARGUMENT = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)


def parse_assistant(text):
    """An OpenAI assistant message from generated GLM text (up to the turn end)."""
    for end in (*TURN_END, "<|endoftext|>"):
        text = text.split(end)[0]
    calls = []
    for index, match in enumerate(TOOL_CALL.finditer(text)):
        body = match.group(1)
        name = body.split("<arg_key>")[0].strip()
        arguments = {}
        for key, value in ARGUMENT.findall(body):
            try:
                arguments[key.strip()] = json.loads(value)
            except json.JSONDecodeError:
                arguments[key.strip()] = value
        calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        )
    content = TOOL_CALL.sub("", text).strip()
    message = {"role": "assistant", "content": content or None}
    if calls:
        message["tool_calls"] = calls
    return message
