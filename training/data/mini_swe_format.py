"""PR-task agent episodes in mini-swe-agent's bash-only format.

mini-swe-agent (and the `ricdomolm/mini-coder-trajs-400k` trajectories and mini-coder
models trained on them) uses no tool schema: each assistant turn is a THOUGHT and one
```` ```bash ```` block, each command's result comes back as a user turn
`<returncode>N</returncode>\\n<output>\\n...</output>`, and the agent submits by running
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached`. The
system prompt and the instructions after the PR description are the dataset's own text.

Our harness (`pr_task_agent.run_episodes`) keeps OpenAI messages with tool calls.
`conversation` turns such an episode into mini-swe-agent messages (the generated text is
kept as the assistant content, so a turn renders exactly as generated), and
`parse_assistant` turns generated text into an assistant message whose single block is
a `bash` call, or `submit` for the submission command. A reply without exactly one
block has no tool call, which ends the episode.

`render` and `encode` give the plain ChatML text that Qwen3's chat template produces for
these messages (no tools, no reasoning) and a loss mask on the assistant turns.
"""

import re

SYSTEM = (
    "You are a helpful assistant that can interact multiple times with a computer shell "
    "to solve programming tasks.\nYour response must contain exactly ONE bash code block "
    "with ONE command (or commands connected with && or ||).\n\nInclude a THOUGHT section "
    "before your command where you explain your reasoning process.\nFormat your response "
    "as shown in <example>.\n\n<example>\nTHOUGHT: Your reasoning and analysis here\n\n"
    "```bash\nyour_command_here\n```\n</example>\n\nFailure to follow these rules will "
    "cause your response to be rejected."
)
SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
BLOCK = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
RESULT = re.compile(r"exit code: (\S+)\n(.*)", re.DOTALL)
TASK = re.compile(r"Repository: [^\n]*\n\n# ([^\n]*)\n\n(.*)", re.DOTALL)


def instructions():
    """The dataset's instructions after the PR description (one text for every row)."""
    from pathlib import Path

    return (Path(__file__).with_name("mini_swe_instructions.txt")).read_text()


def instance(title, body):
    return (
        "<pr_description>\nConsider the following PR description:\n"
        f"{title}\n{body}\n</pr_description>{instructions()}"
    )


def observation(result):
    """Our bash tool's result (`exit code: N\\n<output>`) as mini-swe-agent shows it."""
    match = RESULT.match(result)
    code, output = match.groups() if match else ("1", result)
    if output and not output.endswith("\n"):
        output += "\n"
    return f"<returncode>{code}</returncode>\n<output>\n{output}</output>"


def conversation(messages):
    """A harness episode (OpenAI messages) as mini-swe-agent messages."""
    task = TASK.match(messages[1]["content"])
    title, body = task.groups() if task else ("", messages[1]["content"])
    out = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instance(title, body)},
    ]
    for message in messages[2:]:
        if message["role"] == "assistant":
            out.append({"role": "assistant", "content": message.get("content") or ""})
        elif message["role"] == "tool":
            out.append({"role": "user", "content": observation(message.get("content") or "")})
    return out


def parse_assistant(text):
    """Generated text as an assistant message with one `bash` or `submit` call, if valid."""
    import json

    text = re.sub(r"<think>.*?</think>\s*", "", text.split("<|im_end|>")[0], flags=re.DOTALL)
    message = {"role": "assistant", "content": text}
    blocks = BLOCK.findall(text)
    if len(blocks) != 1:
        return message
    command = blocks[0].strip()
    if SUBMIT in command:
        call = {"name": "submit", "arguments": "{}"}
    else:
        call = {"name": "bash", "arguments": json.dumps({"command": command})}
    message["tool_calls"] = [{"type": "function", "function": call}]
    return message


def pieces(messages, add_generation_prompt=False):
    """(text, is_assistant_output) pieces in plain ChatML."""
    out = []
    for message in messages:
        if message["role"] == "assistant":
            out.append(("<|im_start|>assistant\n", False))
            out.append((f"{message['content']}<|im_end|>", True))
            out.append(("\n", False))
        else:
            out.append((f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n", False))
    if add_generation_prompt:
        out.append(("<|im_start|>assistant\n", False))
    return out


def render(messages, add_generation_prompt=False):
    return "".join(text for text, _ in pieces(messages, add_generation_prompt))


def encode(encode_text, messages):
    """(token ids, loss mask) with `encode_text(str) -> list[int]` (no special-token adds)."""
    ids, mask = [], []
    for text, output in pieces(messages):
        piece = encode_text(text)
        ids.extend(piece)
        mask.extend([int(output)] * len(piece))
    return ids, mask
