"""PR-task episodes in the format of NVIDIA's Open-SWE-Traces `minisweagent` subset.

Open-SWE-Traces (`nvidia/Open-SWE-Traces`, CC-BY-4.0) holds mini-swe-agent v2 trajectories
on real SWE-rebench V2 and Scale-SWE tasks: one `bash` tool in OpenAI tool-call format,
results as JSON `{"returncode": N, "output": "..."}`, and submission by running
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`. The system prompt, the
tool schema and the instructions after the PR description are the dataset's own text
(one of each in every row checked); its instructions name `/testbed` as the working
directory, so episodes in our sandboxes run with `/testbed` linked to the repository.

Messages render in Qwen3's tool format (`training_data.qwen_agent_chat`) without the
teachers' reasoning. `training_messages` turns a dataset row into training messages;
`conversation` turns a harness episode (OpenAI messages) into the same layout;
`parse_assistant` maps generated text to harness calls (`bash`, or `submit` for the
submission command).
"""

import json
import re
from pathlib import Path

SYSTEM = (
    "You are a helpful assistant that can interact with a computer shell to solve "
    "programming tasks."
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"}
                },
                "required": ["command"],
            },
        },
    }
]
SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
WORKDIR = "/testbed"
RESULT = re.compile(r"exit code: (\S+)\n(.*)", re.DOTALL)
# The sandbox clips long output in the middle (`pr_task_runner._clip`).
CLIPPED = re.compile(r"(.*)\n\[\.\.\. (\d+) characters omitted \.\.\.\]\n(.*)", re.DOTALL)
# The dataset's JSON escapes these like Go's encoding/json (HTML-safe).
ESCAPES = {"<": "\\u003c", ">": "\\u003e", "&": "\\u0026", "'": "\\u0027"}
TASK = re.compile(r"Repository: [^\n]*\n\n# ([^\n]*)\n\n(.*)", re.DOTALL)


def instructions():
    return Path(__file__).with_name("open_swe_instructions.txt").read_text()


def instance(title, body):
    return (
        "<pr_description>\nConsider the following PR description:\n"
        f"# Title\n{title}\n\n# Description\n{body}\n</pr_description>{instructions()}"
    )


def observation(result):
    """Our bash tool's result (`exit code: N\\n<output>`) as the dataset's JSON: the
    output in full, or, when the sandbox clipped it, head and tail as the dataset shows
    output over 10,000 characters."""
    match = RESULT.match(result)
    code, output = match.groups() if match else ("1", result)
    value = {"returncode": int(code) if code.lstrip("-").isdigit() else code}
    clipped = CLIPPED.fullmatch(output)
    if clipped:
        head, elided, tail = clipped.groups()
        value.update(
            output_head=head, output_tail=tail, elided_chars=int(elided), warning="Output too long."
        )
    else:
        value["output"] = output
    text = json.dumps(value, indent=2)
    for character, escape in ESCAPES.items():
        text = text.replace(character, escape)
    return text


def _message(role, content, tool_calls=None):
    message = {"role": role, "content": content or ""}
    if tool_calls:
        message["tool_calls"] = [
            {"type": "function", "function": dict(c["function"])} for c in tool_calls
        ]
    return message


def training_messages(row):
    """A dataset row's messages without reasoning, in harness layout."""
    return [_message(m["role"], m["content"], m.get("tool_calls")) for m in row["messages"]]


def conversation(messages):
    """A harness episode as Open-SWE messages."""
    task = TASK.match(messages[1]["content"])
    title, body = task.groups() if task else ("", messages[1]["content"])
    out = [_message("system", SYSTEM), _message("user", instance(title, body))]
    for message in messages[2:]:
        if message["role"] == "assistant":
            calls = [
                {"function": {"name": "bash", "arguments": c["function"]["arguments"]}}
                if c["function"]["name"] == "bash"
                else {
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": f"echo {SUBMIT} && cat patch.txt"}),
                    }
                }
                for c in message.get("tool_calls") or []
            ]
            out.append(_message("assistant", message.get("content"), calls))
        elif message["role"] == "tool":
            out.append(_message("tool", observation(message.get("content") or "")))
    return out


def parse_assistant(text):
    """Generated text as a harness assistant message (`bash` calls, `submit` on submission)."""
    from training_data import qwen_agent_chat

    message = qwen_agent_chat.parse_assistant(text)
    for call in message.get("tool_calls") or []:
        if call["function"]["name"] != "bash":
            continue
        command = json.loads(call["function"]["arguments"]).get("command", "")
        if SUBMIT in str(command):
            call["function"] = {"name": "submit", "arguments": "{}"}
    return message
