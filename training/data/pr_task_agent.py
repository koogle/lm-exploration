"""Agent episodes on PR tasks: tools, prompts, the held-out split and the turn loop.

An episode is one task: the agent sees the repository at the parent revision with
the commit's tests removed (`pr_task_runner.workspace`) and the task's issue, and
works through the tools below until it calls `submit`, answers without a tool call,
or reaches `MAX_TURNS`. Its patch is then graded against the commit's hidden tests
(`pr_task_runner.grade`): resolved when every fail-to-pass test passes and every
pass-to-pass test still passes.

A policy maps a list of conversations to one assistant message each (or None when a
conversation no longer fits), so a teacher API answers one conversation per call
and a local model answers a whole batch per generation (the hybrid family's
`model_hybrid.agent_policy.LocalPolicy`). `modal_scripts/modal_agentic.py` provides the
sandboxes and runs the stages.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

MAX_TURNS = 40
HELDOUT_SALT = "heldout-v1:"
HELDOUT_FRACTION = 0.25


def _function(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [
    _function(
        "bash",
        "Run a shell command in the repository root (no network; 180 s limit). Returns "
        "the exit code and the combined output, shortened in the middle when long.",
        {"command": {"type": "string", "description": "The command to run"}},
        ["command"],
    ),
    _function(
        "read_file",
        "Read a file with line numbers, at most 400 lines per call.",
        {
            "path": {"type": "string", "description": "Path relative to the repository root"},
            "start_line": {"type": "integer", "description": "First line (1-based)"},
            "end_line": {"type": "integer", "description": "Last line"},
        },
        ["path"],
    ),
    _function(
        "edit_file",
        "Replace the one occurrence of `old` in a file with `new`.",
        {
            "path": {"type": "string", "description": "Path relative to the repository root"},
            "old": {"type": "string", "description": "Exact text to replace; must occur once"},
            "new": {"type": "string", "description": "Replacement text"},
        },
        ["path", "old", "new"],
    ),
    _function(
        "write_file",
        "Create or overwrite a file.",
        {
            "path": {"type": "string", "description": "Path relative to the repository root"},
            "content": {"type": "string", "description": "The whole file"},
        },
        ["path", "content"],
    ),
    _function("submit", "Finish: the current state of the repository is your answer.", {}, []),
]
TEST_COMMANDS = {
    "python": "python -m pytest",
    "go": "go test ./...",
    "javascript": "the project's test runner (see package.json)",
}
SYSTEM = (
    "You are a software engineer working in a {language} repository checked out at "
    "the repository root. Resolve the issue below by changing the code. Explore the "
    "code first, make focused changes, and check them by running code or tests (for "
    "example {tests}); you may add tests. There is no network access. Call `submit` "
    "when the change is complete."
)


def heldout(repository):
    """Whether a repository's tasks are held out from training (by owner, so sibling
    repositories of one owner stay on one side)."""
    owner = repository.split("/")[0]
    digest = hashlib.sha256((HELDOUT_SALT + owner).encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64 < HELDOUT_FRACTION


def first_messages(task, issue):
    language = task["language"]
    system = SYSTEM.format(language=language, tests=TEST_COMMANDS[language])
    user = f"Repository: {task['repository']}\n\n# {issue['title']}\n\n{issue['body']}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def arguments(call):
    """A tool call's arguments as a dict (malformed JSON gives an empty dict)."""
    raw = call["function"].get("arguments") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def run_episodes(episodes, policy, execute, max_turns=MAX_TURNS, workers=32):
    """Advance every episode until it ends; returns the episodes.

    `episodes` are dicts with "messages"; `policy(conversations)` returns an
    assistant message (or None: the conversation no longer fits) per conversation;
    `execute(episode, name, arguments)` returns a tool result. Each episode gets
    "end" (submitted, answered, max_turns, context) and "turns".
    """
    for episode in episodes:
        episode.setdefault("turns", 0)
    with ThreadPoolExecutor(workers) as pool:
        while active := [e for e in episodes if "end" not in e]:
            replies = policy([e["messages"] for e in active])
            calls = []
            for episode, reply in zip(active, replies, strict=True):
                if reply is None:
                    episode["end"] = "context"
                    continue
                episode["messages"].append(reply)
                episode["turns"] += 1
                tool_calls = reply.get("tool_calls") or []
                if not tool_calls:
                    episode["end"] = "answered"
                    continue
                for index, call in enumerate(tool_calls):
                    call.setdefault("id", f"call_{episode['turns']}_{index}")
                    calls.append((episode, call))
                if any(c["function"]["name"] == "submit" for c in tool_calls):
                    episode["end"] = "submitted"
                elif episode["turns"] >= max_turns:
                    episode["end"] = "max_turns"

            def answer(item):
                episode, call = item
                name = call["function"]["name"]
                if name == "submit":
                    return "submitted"
                return execute(episode, name, arguments(call))

            for (episode, call), result in zip(calls, pool.map(answer, calls), strict=True):
                episode["messages"].append(
                    {"role": "tool", "tool_call_id": call["id"], "content": result}
                )
    return episodes


def resolved(grade, verification):
    """Every fail-to-pass test passes and no pass-to-pass test fails after the patch."""
    if grade.get("stage") != "graded":
        return False
    results = grade["results"]
    needed = verification["fail_to_pass"] + verification["pass_to_pass"]
    return bool(needed) and all(results.get(name) == "pass" for name in needed)


def training_row(messages):
    """An episode as an SFT conversation: OpenAI messages and the tool schemas."""
    keep = ("role", "content", "tool_calls")
    rows = []
    for message in messages:
        row = {k: message[k] for k in keep if message.get(k) is not None}
        if row.get("tool_calls"):
            row["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": c["function"]["name"],
                        "arguments": json.dumps(arguments(c), ensure_ascii=False),
                    },
                }
                for c in row["tool_calls"]
            ]
        row.setdefault("content", "")
        rows.append(row)
    return {"messages": rows, "tools": TOOLS}
