"""Terminus 2 trajectories (Harbor's Terminal-Bench baseline harness) for training.

Terminus 2 has no tool schema: the user turns carry the instructions and the terminal
screen, and each assistant turn is JSON (`analysis`, `plan`, `commands` of tmux
keystrokes, `task_complete`). Public corpora in this format store conversations as
`{"from": "human"|"gpt", "value"}` (Terminal-Lego) or `{"role", "content"}`
(Nemotron-Terminal-Corpus). `training_messages` normalizes both and removes the
teachers' `<think>` blocks; the messages render as plain ChatML
(`training_data.mini_swe_format.encode`), which is what Qwen3's template gives for
conversations without tools or reasoning.
"""

import re

ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}
THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def training_messages(conversations):
    """Normalized messages, or None when the conversation has no assistant turn."""
    out = []
    for turn in conversations:
        role = ROLES[turn.get("from") or turn.get("role")]
        content = turn.get("value") if "value" in turn else turn.get("content")
        if role == "assistant":
            content = THINK.sub("", content or "").strip()
        out.append({"role": role, "content": content or ""})
    if not any(m["role"] == "assistant" for m in out):
        return None
    return out
