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
`from_harness` turns a finished harness episode with our five tools into a training
conversation that uses only the bash tool;
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


READ_LINES = 400  # pr_task_runner.READ_LINES
NUMBERED = re.compile(r" *(\d+)  (.*)")


def _bash(command):
    return {"function": {"name": "bash", "arguments": json.dumps({"command": command})}}


def _heredoc(text, word="EOF"):
    while word in text.splitlines():
        word += "_"
    return word


def _as_bash(call, result):
    """(bash call, its result as `exit code: N\\n<output>`) for one of our tool calls."""
    import shlex

    name, arguments = call["function"]["name"], call["function"].get("arguments") or "{}"
    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    if name == "bash":
        return _bash(arguments.get("command", "")), result
    path = shlex.quote(str(arguments.get("path", "")))
    failed = result.startswith("error:")
    if name == "read_file":
        start = int(arguments.get("start_line") or 1)
        end = int(arguments.get("end_line") or start + READ_LINES - 1)
        command = f"nl -ba {path} | sed -n '{start},{end}p'"
        if failed:
            return _bash(
                command
            ), f"exit code: 0\nnl: {arguments.get('path')}: No such file or directory"
        lines = []
        for line in result.split("\n"):
            match = NUMBERED.fullmatch(line)
            if match:
                lines.append(f"{int(match.group(1)):>6}\t{match.group(2)}")
            elif not line.startswith("[") or not line.endswith("lines in total]"):
                lines.append(line)
        return _bash(command), "exit code: 0\n" + "\n".join(lines) + "\n"
    if name == "write_file":
        content = str(arguments.get("content", ""))
        word = _heredoc(content)
        command = f"mkdir -p \"$(dirname {path})\" && cat <<'{word}' > {path}\n{content}\n{word}"
        return _bash(command), "exit code: 0\n"
    if name == "edit_file":
        old, new = str(arguments.get("old", "")), str(arguments.get("new", ""))
        script = (
            "from pathlib import Path\n"
            f"p = Path({arguments.get('path')!r})\n"
            "s = p.read_text()\n"
            f"old = {old!r}\n"
            f"new = {new!r}\n"
            'assert s.count(old) == 1, f"old text occurs {s.count(old)} times"\n'
            "p.write_text(s.replace(old, new))"
        )
        word = _heredoc(script, "PY")
        command = f"python3 - <<'{word}'\n{script}\n{word}"
        if failed:
            count = re.search(r"occurs (\d+) times", result)
            message = f"old text occurs {count.group(1) if count else 0} times"
            return _bash(command), (
                "exit code: 1\nTraceback (most recent call last):\n"
                '  File "<stdin>", line 6, in <module>\n'
                f"AssertionError: {message}\n"
            )
        return _bash(command), "exit code: 0\n"
    raise ValueError(f"Unexpected tool {name}")


def from_harness(messages):
    """Training messages in the dataset's bash-only layout for a resolved harness episode
    with our tools (bash, read_file, edit_file, write_file, submit). Every episode ends
    with the dataset's submission: the diff written to patch.txt, then the submit
    command (episodes that ran out of turns get the same ending)."""
    task = TASK.match(messages[1]["content"])
    title, body = task.groups() if task else ("", messages[1]["content"])
    out = [_message("system", SYSTEM), _message("user", instance(title, body))]
    position = 2
    while position < len(messages):
        message = messages[position]
        if message["role"] != "assistant":
            position += 1
            continue
        calls = [c for c in message.get("tool_calls") or [] if c["function"]["name"] != "submit"]
        following = messages[position + 1 : position + 1 + len(calls)]
        results = [m.get("content", "") for m in following if m["role"] == "tool"]
        converted = [_as_bash(c, r) for c, r in zip(calls, results, strict=False)]
        if converted or message.get("content"):
            out.append(_message("assistant", message.get("content"), [c for c, _ in converted]))
            out.extend(_message("tool", observation(r)) for _, r in converted)
        position += 1 + len(results)
    out.append(_message("assistant", "", [_bash("git diff > patch.txt")]))
    out.append(_message("tool", observation("exit code: 0\n")))
    out.append(_message("assistant", "", [_bash(f"echo {SUBMIT} && cat patch.txt")]))
    return out
