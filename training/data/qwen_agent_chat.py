"""Agent conversations in Qwen3's chat format, for Qwen reference models.

Conversations are OpenAI chat messages with tool calls, as `pr_task_agent.training_row`
gives them. They render in Qwen3's ChatML layout with its tool section: the tool
schemas in the system turn, calls as `<tool_call>\\n{"name": ..., "arguments": ...}\\n
</tool_call>` after the assistant's text, and consecutive tool results in one user
turn as `<tool_response>` blocks. This is the text Qwen3's own chat template produces
for these conversations without reasoning; `chat=True` adds the empty
`<think>\\n\\n</think>\\n\\n` block that the template puts in front of a reply when
thinking is off (earlier turns carry none, as in the template), which Qwen3's
post-trained (chat) models expect.

`encode` returns token ids and a loss mask that is 1 on each assistant turn after its
`<|im_start|>assistant\\n` prompt, through its `<|im_end|>`. Pieces are tokenized
separately, so a generation prompt tokenizes exactly as in training.
`parse_assistant` turns generated text back into an OpenAI assistant message.
"""

import json
import re

TURN_END = "<|im_end|>"
TOOLS_HEADER = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)
EMPTY_THINK = "<think>\n\n</think>\n\n"
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _arguments(call):
    raw = call["function"].get("arguments") or {}
    return json.loads(raw) if isinstance(raw, str) else raw


def _system(content, tools):
    text = f"{content}\n\n" if content else ""
    schemas = "".join("\n" + json.dumps(tool, ensure_ascii=False) for tool in tools)
    return f"<|im_start|>system\n{text}{TOOLS_HEADER}{schemas}{TOOLS_FOOTER}{TURN_END}\n"


def _assistant_body(message):
    text = message.get("content") or ""
    for index, call in enumerate(message.get("tool_calls") or []):
        if index or text:
            text += "\n"
        name = json.dumps(call["function"]["name"], ensure_ascii=False)
        arguments = json.dumps(_arguments(call), ensure_ascii=False)
        text += f'<tool_call>\n{{"name": {name}, "arguments": {arguments}}}\n</tool_call>'
    return text + TURN_END


def pieces(messages, tools, add_generation_prompt=False, chat=False):
    """(text, is_assistant_output) pieces of the rendered conversation."""
    out, index = [], 0
    if messages and messages[0]["role"] == "system":
        out.append((_system(messages[0]["content"], tools), False))
        index = 1
    else:
        out.append((_system("", tools), False))
    prompt = "<|im_start|>assistant\n"
    while index < len(messages):
        message = messages[index]
        role = message["role"]
        if role == "user":
            out.append((f"<|im_start|>user\n{message['content']}{TURN_END}\n", False))
        elif role == "assistant":
            out.append((prompt, False))
            out.append((_assistant_body(message), True))
            out.append(("\n", False))
        elif role == "tool":
            results = []
            while index < len(messages) and messages[index]["role"] == "tool":
                results.append(f"<tool_response>\n{messages[index]['content']}\n</tool_response>")
                index += 1
            out.append((f"<|im_start|>user\n{chr(10).join(results)}{TURN_END}\n", False))
            continue
        else:
            raise ValueError(f"Unexpected role {role}")
        index += 1
    if add_generation_prompt:
        out.append((prompt + (EMPTY_THINK if chat else ""), False))
    return out


def render(messages, tools, add_generation_prompt=False, chat=False):
    return "".join(text for text, _ in pieces(messages, tools, add_generation_prompt, chat))


def encode(encode_text, messages, tools, chat=False):
    """(token ids, loss mask) with `encode_text(str) -> list[int]` (no special-token adds)."""
    ids, mask = [], []
    for text, output in pieces(messages, tools, chat=chat):
        piece = encode_text(text)
        ids.extend(piece)
        mask.extend([int(output)] * len(piece))
    return ids, mask


def parse_assistant(text):
    """Generated text (up to `<|im_end|>`) as an OpenAI assistant message."""
    text = text.split(TURN_END)[0]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    calls = []
    for index, match in enumerate(CALL.finditer(text)):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            continue
        arguments = data.get("arguments")
        calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": data["name"],
                    "arguments": json.dumps(arguments if isinstance(arguments, dict) else {}),
                },
            }
        )
    content = CALL.sub("", text).split("<tool_call>")[0].strip()
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return message
